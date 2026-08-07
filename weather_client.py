"""
Client for the National Weather Service API (https://api.weather.gov/).

The weather counterpart of massive_client.py. Two deliberate differences from
the Massive client:

  1. NO API key. The NWS API is free and open, so there is no secret to fetch -
     but NWS policy REQUIRES a descriptive User-Agent header (ideally with
     contact info) on every request. Override it with the WEATHER_USER_AGENT
     env var for real deployments.
  2. It also exposes `normalize_*` helpers that turn a raw NWS item into a
     `weather_documents` row dict, so the notebook AND a future Flask
     `POST /weather/sync` route can share the same normalization logic.

Endpoints used (see https://api.weather.gov/):
  GET /points/{lat},{lon}                          -> resolve a grid point
  GET /alerts/active?area={state}                  -> active alerts for a state
  GET /gridpoints/{office}/{x},{y}/forecast        -> multi-day narrative forecast
  GET /gridpoints/{office}/{x},{y}/forecast/hourly -> hourly narrative forecast

Pipeline chain (state-driven): area -> points -> forecast (+ hourly)
  1. get_active_alerts(area=STATE)     # area: active alerts for a state
       -> alert_centroid(feature)      #   area FEEDS points: polygon -> (lat, lon)
  2. get_point(lat, lon)               # points: resolve the grid cell
       -> grid_from_point(props)       #   points FEEDS forecast: (office, x, y)
  3. get_forecast(office, x, y)        # multi-day narrative forecast
  4. get_hourly_forecast(office, x, y) # hourly narrative forecast
"""

import hashlib
import json
import os
from datetime import datetime
from typing import Any

import requests

_BASE_URL = os.environ.get("WEATHER_API_BASE_URL", "https://api.weather.gov")
# NWS requires a descriptive User-Agent (with contact info). Override in prod.
_USER_AGENT = os.environ.get(
    "WEATHER_USER_AGENT", "(databricks-lakebase-app-day-2, contact@example.com)"
)

_DEFAULT_TIMEOUT = 30


class WeatherClient:
    """Thin wrapper around the NWS API (no auth; descriptive User-Agent required)."""

    def __init__(
        self,
        base_url: str | None = None,
        user_agent: str | None = None,
        timeout: int = _DEFAULT_TIMEOUT,
    ):
        self.base_url = (base_url or _BASE_URL).rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": user_agent or _USER_AGENT,
                # NWS returns GeoJSON; ask for it explicitly.
                "Accept": "application/geo+json",
            }
        )

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        resp = self._session.get(f"{self.base_url}{path}", params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_point(self, lat: float, lon: float) -> dict:
        """GET /points/{lat},{lon} -> the `properties` object. Contains gridId
        (office), gridX, gridY, the forecast + forecastHourly URLs, and
        relativeLocation (nearest city/state)."""
        data = self.get(f"/points/{lat},{lon}")
        return data.get("properties", {}) or {}

    def get_active_alerts(
        self,
        area: str | None = None,
        point: str | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        """GET /alerts/active. Pass `area='IL'` (a state/marine code) OR
        `point='lat,lon'`. Returns the raw GeoJSON `features` list; each
        feature's `properties` has the free-text description + instruction."""
        params: dict[str, Any] = {}
        if area:
            params["area"] = area
        if point:
            params["point"] = point
        if limit:
            params["limit"] = limit
        data = self.get("/alerts/active", params=params or None)
        return data.get("features", []) or []

    def get_forecast(self, office: str, grid_x: int, grid_y: int) -> list[dict]:
        """GET /gridpoints/{office}/{x},{y}/forecast -> the `periods` list. Each
        period has a narrative `detailedForecast` (e.g. "Sunny, with a high near
        78. Northwest wind around 6 mph.")."""
        data = self.get(f"/gridpoints/{office}/{grid_x},{grid_y}/forecast")
        return (data.get("properties", {}) or {}).get("periods", []) or []

    def get_hourly_forecast(self, office: str, grid_x: int, grid_y: int) -> list[dict]:
        """GET /gridpoints/{office}/{x},{y}/forecast/hourly -> hourly `periods`.

        NOTE: hourly periods have an EMPTY `detailedForecast` and empty `name`;
        the narrative content is in `shortForecast` (+ temperature/wind/precip).
        Use normalize_hourly_period() to turn one into an embeddable document."""
        data = self.get(f"/gridpoints/{office}/{grid_x},{grid_y}/forecast/hourly")
        return (data.get("properties", {}) or {}).get("periods", []) or []


# --------------------------------------------------------------------------- #
# API-call chain plumbing: area -> points -> forecast / forecast-hourly.
# Pure helpers that connect the four calls. An alert from get_active_alerts()
# (area) yields a centroid that feeds get_point() (points), whose grid params
# feed get_forecast() and get_hourly_forecast().
# --------------------------------------------------------------------------- #

def _flatten_positions(coords: Any):
    """Yield every GeoJSON position ([lon, lat] or [lon, lat, elev]) from any
    nesting depth - Point / Polygon / MultiPolygon all reduce to the same leaf
    pairs, so we can centroid them uniformly."""
    if isinstance(coords, (list, tuple)) and coords and all(
        isinstance(c, (int, float)) for c in coords
    ):
        yield coords
        return
    if isinstance(coords, (list, tuple)):
        for c in coords:
            yield from _flatten_positions(c)


def alert_centroid(feature: dict) -> tuple[float, float] | None:
    """Approximate (lat, lon) for an alert so an `area` alert can FEED `/points`.

    NWS geometry is GeoJSON in [lon, lat] order; we average all positions to a
    centroid and RETURN (lat, lon) - the order GET /points/{lat},{lon} expects -
    rounded to 4 dp (NWS rejects more than 4). Many alerts have `geometry: null`
    (they only reference `affectedZones`); for those we return None so the caller
    skips the forecast leg but still keeps the alert document."""
    geom = feature.get("geometry") or None
    if not geom:
        return None
    positions = list(_flatten_positions(geom.get("coordinates")))
    if not positions:
        return None
    avg_lon = sum(p[0] for p in positions) / len(positions)
    avg_lat = sum(p[1] for p in positions) / len(positions)
    return (round(avg_lat, 4), round(avg_lon, 4))


def grid_from_point(point_props: dict) -> tuple[str, int, int] | None:
    """(office, gridX, gridY) from a get_point() result, so `points` can FEED the
    two forecast calls. Also the natural dedup key: many alert centroids in one
    area collapse to the same grid cell, so each cell's forecast is fetched once."""
    office = point_props.get("gridId")
    gx = point_props.get("gridX")
    gy = point_props.get("gridY")
    if not office or gx is None or gy is None:
        return None
    return (office, int(gx), int(gy))


# --------------------------------------------------------------------------- #
# Normalization: raw NWS item -> weather_documents row dict.
# payload is JSON-serialized here so the row is directly insertable (mirrors
# build_news_rows in the news pipeline).
# --------------------------------------------------------------------------- #

def _clean(text: str | None) -> str:
    return (text or "").strip()


def _hour_label(iso_ts: str | None) -> str | None:
    """'2026-08-06T18:00:00-05:00' -> 'Aug 6 6 PM' (cross-platform; no %-I)."""
    if not iso_ts:
        return None
    try:
        dt = datetime.fromisoformat(iso_ts)
    except (ValueError, TypeError):
        return iso_ts
    hour_12 = dt.hour % 12 or 12
    am_pm = "AM" if dt.hour < 12 else "PM"
    return f"{dt.strftime('%b')} {dt.day} {hour_12} {am_pm}"


def normalize_alert(feature: dict) -> dict | None:
    """NWS alert feature -> weather_documents row (source_type='alert').
    narrative_text = description + instruction (the hazard + the "what to do")."""
    p = feature.get("properties", {}) or {}
    description = _clean(p.get("description"))
    instruction = _clean(p.get("instruction"))
    narrative = (description + ("\n\n" + instruction if instruction else "")).strip()
    if not narrative:
        return None
    return {
        "id": p.get("id") or feature.get("id"),
        "location": _clean(p.get("areaDesc")) or "Unknown area",
        "source_type": "alert",
        "headline": p.get("headline") or p.get("event"),
        "event": p.get("event"),
        "narrative_text": narrative,
        "issued_at": p.get("sent") or p.get("effective"),
        "effective_at": p.get("effective") or p.get("onset"),
        "payload": json.dumps(feature),
    }


def normalize_forecast_period(location: str, period: dict) -> dict | None:
    """NWS multi-day forecast period -> weather_documents row
    (source_type='forecast'). Periods have no stable API id, so hash
    location + period number + startTime into a deterministic dedup key."""
    narrative = _clean(period.get("detailedForecast"))
    if not narrative:
        return None
    raw_key = f"forecast|{location}|{period.get('number')}|{period.get('startTime')}"
    return {
        "id": "forecast_" + hashlib.md5(raw_key.encode("utf-8")).hexdigest(),
        "location": location,
        "source_type": "forecast",
        "headline": period.get("name"),
        "event": "Forecast",
        "narrative_text": narrative,
        "issued_at": period.get("startTime"),
        "effective_at": period.get("startTime"),
        "payload": json.dumps(period),
    }


def normalize_hourly_period(location: str, period: dict) -> dict | None:
    """NWS hourly forecast period -> weather_documents row (source_type='hourly').

    Hourly periods have an EMPTY detailedForecast and empty name, so we synthesize
    a short narrative from shortForecast + temperature + wind + precip chance -
    otherwise there'd be nothing meaningful to embed."""
    short = _clean(period.get("shortForecast"))
    if not short:
        return None

    temp = period.get("temperature")
    unit = period.get("temperatureUnit") or "F"
    wind = _clean(period.get("windSpeed"))
    wind_dir = _clean(period.get("windDirection"))
    pop = (period.get("probabilityOfPrecipitation") or {}).get("value")
    start = period.get("startTime")

    parts = [short.rstrip(".") + "."]
    if temp is not None:
        parts.append(f"Temperature {temp}°{unit}.")
    if wind:
        parts.append(f"Wind {wind}{(' ' + wind_dir) if wind_dir else ''}.")
    if pop:
        parts.append(f"Chance of precipitation {pop}%.")
    narrative = " ".join(parts)

    raw_key = f"hourly|{location}|{start}"
    return {
        "id": "hourly_" + hashlib.md5(raw_key.encode("utf-8")).hexdigest(),
        "location": location,
        "source_type": "hourly",
        "headline": _hour_label(start),
        "event": "Hourly Forecast",
        "narrative_text": narrative,
        "issued_at": start,
        "effective_at": start,
        "payload": json.dumps(period),
    }
