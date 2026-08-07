"""
Weather harvest + upsert logic, shared by the Flask POST /weather/sync route
(app.py) and the standalone seeder (seed_weather.py).

This module deliberately does NOT import Flask - it only needs weather_client
(NWS API + normalization) and lakebase (Postgres writes), so seed_weather.py can
run it without pulling in the web framework or a running server.
"""

import logging
import os
import re

import requests

import lakebase
from weather_client import (
    WeatherClient,
    alert_centroid,
    grid_from_point,
    normalize_alert,
    normalize_forecast_period,
    normalize_hourly_period,
)

logger = logging.getLogger("weather-sync")

WEATHER_TABLE_NAME = os.environ.get("WEATHER_TABLE_NAME", "weather_documents")

# States (2-letter NWS area codes) to harvest active weather alerts for by
# default. Kept to 2 states so the embedding step stays manageable - the hourly
# leg alone yields ~150 periods per grid cell, so each extra state multiplies
# the document (and vector) count quickly.
DEFAULT_WEATHER_STATES = [
    s.strip().upper()
    for s in os.environ.get("WEATHER_STATES", "TX,FL").split(",")
    if s.strip()
]

# Cap on distinct forecast grid cells expanded per state, so a state with many
# active alerts doesn't fan out into hundreds of NWS forecast calls per sync.
WEATHER_MAX_GRIDS_PER_STATE = int(os.environ.get("WEATHER_MAX_GRIDS_PER_STATE", 5))

# 2-letter US state / NWS marine area code (e.g. "IL", "TX", "GM").
_STATE_RE = re.compile(r"^[A-Z]{2}$")


def parse_states(raw) -> list[str]:
    """Normalize a request body's states/locations into distinct 2-letter area
    codes. Accepts bare codes ("TX") or "City, ST" strings (takes the trailing
    state code, so the homework's ["Chicago, IL"] example still works). Anything
    that isn't a valid 2-letter code is dropped."""
    out: list[str] = []
    for item in raw or []:
        if not isinstance(item, str):
            continue
        token = item.strip().upper()
        if "," in token:
            token = token.rsplit(",", 1)[-1].strip()
        if _STATE_RE.match(token) and token not in out:
            out.append(token)
    return out


def _point_location_label(point_props: dict, fallback: str) -> str:
    """Human-readable "City, ST" from a point's relativeLocation, else fallback."""
    rel = (point_props.get("relativeLocation") or {}).get("properties") or {}
    city, state = rel.get("city"), rel.get("state")
    return f"{city}, {state}" if city and state else fallback


def ensure_weather_documents_table():
    """
    Create the raw weather-documents table in Lakebase if it doesn't exist yet.
    Mirrors ensure_news_table(): this is the RAW document store the weather
    embedding script reads from to compute vectors. DDL matches
    sql/weather/01_setup_weather_documents_table.sql.
    """
    lakebase.run_write(
        f"""
        CREATE TABLE IF NOT EXISTS {WEATHER_TABLE_NAME} (
            id             TEXT PRIMARY KEY,
            location       TEXT NOT NULL,
            source_type    TEXT NOT NULL,
            headline       TEXT,
            event          TEXT,
            narrative_text TEXT NOT NULL,
            issued_at      TIMESTAMPTZ,
            effective_at   TIMESTAMPTZ,
            payload        JSONB NOT NULL,
            synced_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    lakebase.run_write(
        f"CREATE INDEX IF NOT EXISTS idx_{WEATHER_TABLE_NAME}_location "
        f"ON {WEATHER_TABLE_NAME} (location)"
    )
    lakebase.run_write(
        f"CREATE INDEX IF NOT EXISTS idx_{WEATHER_TABLE_NAME}_source_type "
        f"ON {WEATHER_TABLE_NAME} (source_type)"
    )


def _upsert_weather_batch(rows: list[dict]) -> int:
    """Upsert normalized weather-document rows into the weather documents table.

    Rows come pre-normalized from weather_client (payload already JSON-encoded),
    so this just maps dict keys to columns. ON CONFLICT (id) DO UPDATE keeps
    re-syncs idempotent.
    """
    count = 0
    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            for row in rows:
                cur.execute(
                    f"""
                    INSERT INTO {WEATHER_TABLE_NAME} (
                        id, location, source_type, headline, event,
                        narrative_text, issued_at, effective_at, payload, synced_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (id) DO UPDATE
                        SET location = EXCLUDED.location,
                            source_type = EXCLUDED.source_type,
                            headline = EXCLUDED.headline,
                            event = EXCLUDED.event,
                            narrative_text = EXCLUDED.narrative_text,
                            issued_at = EXCLUDED.issued_at,
                            effective_at = EXCLUDED.effective_at,
                            payload = EXCLUDED.payload,
                            synced_at = EXCLUDED.synced_at
                    """,
                    (
                        row["id"],
                        row["location"],
                        row["source_type"],
                        row.get("headline"),
                        row.get("event"),
                        row["narrative_text"],
                        row.get("issued_at"),
                        row.get("effective_at"),
                        row["payload"],
                    ),
                )
                count += 1
            conn.commit()
    return count


def sync_weather(states: list[str] | None = None, limit: int = 50) -> dict:
    """
    Harvest unstructured weather text from the National Weather Service and
    upsert it into weather_documents, returning a summary dict. State/area-driven
    (mirrors the news sync): for each state we pull active alerts, then follow
    each alert's centroid to its forecast grid cell and pull the multi-day +
    hourly narrative forecasts.

    Importable and runnable WITHOUT a live server (see seed_weather.py) - the
    POST /weather/sync route is just a thin wrapper over this. Falls back to
    DEFAULT_WEATHER_STATES when `states` is empty/None.
    """
    ensure_weather_documents_table()
    client = WeatherClient()

    states = states or DEFAULT_WEATHER_STATES
    logger.info("Weather sync starting for states %s (limit=%d)", states, limit)

    rows: list[dict] = []
    seen_ids: set[str] = set()

    def _add(doc: dict | None):
        if doc and doc.get("id") and doc["id"] not in seen_ids:
            seen_ids.add(doc["id"])
            rows.append(doc)

    for state in states:
        try:
            alerts = client.get_active_alerts(area=state, limit=limit)
        except requests.HTTPError as exc:
            logger.warning("Failed to fetch active alerts for %s: %s", state, exc)
            continue
        logger.info(
            "%s: %d active alerts (expanding up to %d forecast grids)",
            state, len(alerts), WEATHER_MAX_GRIDS_PER_STATE,
        )

        seen_grids: set[tuple] = set()
        for feature in alerts:
            _add(normalize_alert(feature))

            # Forecast leg: alert centroid -> grid cell -> daily + hourly.
            if len(seen_grids) >= WEATHER_MAX_GRIDS_PER_STATE:
                continue
            centroid = alert_centroid(feature)
            if not centroid:
                continue
            try:
                point_props = client.get_point(*centroid)
                grid = grid_from_point(point_props)
                if not grid or grid in seen_grids:
                    continue
                seen_grids.add(grid)
                location = _point_location_label(
                    point_props, f"{centroid[0]},{centroid[1]}"
                )
                for period in client.get_forecast(*grid):
                    _add(normalize_forecast_period(location, period))
                for period in client.get_hourly_forecast(*grid):
                    _add(normalize_hourly_period(location, period))
                logger.info(
                    "  %s grid %s (%s): %d docs so far",
                    state, grid, location, len(rows),
                )
            except requests.HTTPError as exc:
                logger.warning("Forecast leg failed near %s in %s: %s", centroid, state, exc)
                continue

    logger.info("Harvested %d unique docs; upserting into weather_documents ...", len(rows))
    total = _upsert_weather_batch(rows)
    logger.info("Weather sync done: %d rows upserted", total)
    return {"synced": total, "states": states, "documents": len(rows)}
