"""
Databricks App boilerplate:
- Serves a small Flask API
- Reads/writes to Lakebase (Databricks-managed Postgres) via lakebase.py
- Pulls data from the Massive API via massive_client.py and syncs it into Lakebase

Run locally:
    python app.py
Deploy as a Databricks App using app.yaml.
"""

import logging
import os
import re

import requests
from databricks.sdk import WorkspaceClient
from flask import Flask, jsonify, render_template, request

import lakebase
from massive_client import MassiveClient
from weather_client import (
    WeatherClient,
    alert_centroid,
    grid_from_point,
    normalize_alert,
    normalize_forecast_period,
    normalize_hourly_period,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("massive-app")

app = Flask(__name__)
_w = WorkspaceClient()

TABLE_NAME = os.environ.get("MASSIVE_TABLE_NAME", "massive_records")
WATCHLIST_TABLE_NAME = os.environ.get("WATCHLIST_TABLE_NAME", "watchlist")
NEWS_TABLE_NAME = os.environ.get("NEWS_TABLE_NAME", "ticker_news_documents")
WEATHER_TABLE_NAME = os.environ.get("WEATHER_TABLE_NAME", "weather_documents")

# Tickers to fetch news for by default (comma-separated), e.g. "AAPL,MSFT,GOOGL"
DEFAULT_NEWS_TICKERS = [
    t.strip().upper()
    for t in os.environ.get("NEWS_TICKERS", "AAPL,MSFT,GOOGL,AMZN,TSLA").split(",")
    if t.strip()
]

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

# Basic stock ticker shape check: 1-10 uppercase letters, with an optional
# ".X" or ".XX" share-class suffix (e.g. "BRK.B"). This rejects obviously
# malformed input before we even call the Massive API.
_TICKER_RE = re.compile(r"^[A-Z]{1,10}(\.[A-Z]{1,2})?$")

# 2-letter US state / NWS marine area code (e.g. "IL", "TX", "GM").
_STATE_RE = re.compile(r"^[A-Z]{2}$")


def ensure_table():
    """Create the destination table in Lakebase if it doesn't exist yet."""
    lakebase.run_write(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id TEXT PRIMARY KEY,
            payload JSONB NOT NULL,
            synced_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )


def ensure_watchlist_table():
    """Create the watchlist table in Lakebase if it doesn't exist yet."""
    lakebase.run_write(
        f"""
        CREATE TABLE IF NOT EXISTS {WATCHLIST_TABLE_NAME} (
            symbol TEXT NOT NULL,
            email TEXT NOT NULL,
            latest_price NUMERIC,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (symbol, email)
        )
        """
    )


def ensure_news_table():
    """
    Create the raw ticker-news documents table in Lakebase if it doesn't
    exist yet. This is the RAW document store the Spark notebook
    (notebooks/ingest_ticker_news_embeddings.py) reads from to compute
    vector embeddings into a separate `<NEWS_TABLE_NAME>_embeddings` table.
    """
    lakebase.run_write(
        f"""
        CREATE TABLE IF NOT EXISTS {NEWS_TABLE_NAME} (
            id TEXT PRIMARY KEY,
            ticker TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT,
            author TEXT,
            article_url TEXT,
            publisher_name TEXT,
            keywords JSONB,
            sentiment TEXT,
            sentiment_reasoning TEXT,
            published_utc TIMESTAMPTZ,
            payload JSONB NOT NULL,
            synced_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    lakebase.run_write(
        f"CREATE INDEX IF NOT EXISTS idx_{NEWS_TABLE_NAME}_ticker "
        f"ON {NEWS_TABLE_NAME} (ticker)"
    )


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


def _current_user_email() -> str:
    """
    Resolve the current user's email so the watchlist can be personalized.

    Databricks Apps inject the logged-in user's identity via the
    X-Forwarded-Email header on every request. Fall back to the Databricks
    SDK's current_user API for local development where that header isn't set.
    """
    header_email = request.headers.get("X-Forwarded-Email")
    if header_email:
        return header_email
    return _w.current_user.me().user_name


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.errorhandler(Exception)
def handle_exception(err):
    """Ensure all unhandled errors return JSON (not an HTML error page),
    so the frontend's resp.json() call never chokes on HTML."""
    logger.exception("Unhandled exception while processing request")
    status_code = getattr(err, "code", 500)
    if not isinstance(status_code, int):
        status_code = 500
    return jsonify({"error": str(err)}), status_code


@app.route("/")
def index():
    """Simple UI to submit a list of stock symbols to sync from Massive."""
    return render_template("index.html")


@app.route("/records")
def list_records():
    """Read records already synced into Lakebase."""
    limit = int(request.args.get("limit", 100))
    rows = lakebase.run_query(
        f"SELECT id, payload, synced_at FROM {TABLE_NAME} ORDER BY synced_at DESC LIMIT %s",
        (limit,),
    )
    return jsonify(rows)


@app.route("/sync", methods=["POST"])
def sync_from_massive():
    """
    Pull data from the Massive API (paginated, potentially huge dataset) and
    upsert it into Lakebase in batches.
    """
    ensure_table()
    client = MassiveClient()

    path = request.json.get("path", "/records") if request.is_json else "/records"
    batch_size = int(request.args.get("batch_size", 500))

    batch = []
    total = 0
    for item in client.paginated_get(path):
        batch.append(item)
        if len(batch) >= batch_size:
            total += _upsert_batch(batch)
            batch = []

    if batch:
        total += _upsert_batch(batch)

    return jsonify({"synced": total})


@app.route("/news/sync", methods=["POST"])
def sync_news_from_massive():
    """
    Pull recent news articles for a set of tickers from Massive (ONE API
    call per ticker, via MassiveClient.get_news) and upsert them into the
    ticker_news_documents table in Lakebase.

    Body (optional JSON): {"tickers": ["AAPL", "MSFT"], "limit": 50}
    Defaults to DEFAULT_NEWS_TICKERS when no tickers are supplied.
    """
    ensure_news_table()
    client = MassiveClient()

    body = request.json if request.is_json else {}
    tickers = body.get("tickers") or DEFAULT_NEWS_TICKERS
    tickers = [t.strip().upper() for t in tickers if isinstance(t, str) and t.strip()]
    limit = int(body.get("limit", 50))

    total = 0
    for ticker in tickers:
        if not _TICKER_RE.match(ticker):
            continue
        articles = client.get_news(ticker, limit=limit)
        total += _upsert_news_batch(ticker, articles)

    return jsonify({"synced": total, "tickers": tickers})


def _parse_states(raw) -> list[str]:
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

    rows: list[dict] = []
    seen_ids: set[str] = set()

    def _add(doc: dict | None):
        if doc and doc.get("id") and doc["id"] not in seen_ids:
            seen_ids.add(doc["id"])
            rows.append(doc)

    for state in states:
        try:
            alerts = client.get_active_alerts(area=state, limit=limit)
        except requests.HTTPError:
            logger.warning("Failed to fetch active alerts for %s", state)
            continue

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
            except requests.HTTPError:
                logger.warning("Forecast leg failed near %s in %s", centroid, state)
                continue

    total = _upsert_weather_batch(rows)
    return {"synced": total, "states": states, "documents": len(rows)}


@app.route("/weather/sync", methods=["POST"])
def sync_weather_from_nws():
    """
    POST /weather/sync - thin HTTP wrapper over sync_weather().

    Body (optional JSON):
        {"states": ["TX", "FL"], "limit": 50}
        {"locations": ["Chicago, IL"], "limit": 50}   # trailing state parsed
    Defaults to DEFAULT_WEATHER_STATES when none are supplied.
    """
    body = request.json if request.is_json else {}
    states = _parse_states(body.get("states") or body.get("locations"))
    limit = int(body.get("limit", 50))
    return jsonify(sync_weather(states=states, limit=limit))


@app.route("/watchlist", methods=["GET"])
def get_watchlist():
    """Return the current user's watchlist symbols, with their last known price."""
    ensure_watchlist_table()
    email = _current_user_email()
    rows = lakebase.run_query(
        f"SELECT symbol, email, latest_price, updated_at FROM {WATCHLIST_TABLE_NAME} "
        f"WHERE email = %s ORDER BY symbol ASC",
        (email,),
    )
    return jsonify(rows)


@app.route("/watchlist", methods=["POST"])
def add_to_watchlist():
    """
    Fetch the latest price for a single stock symbol from Massive using
    exactly ONE API call (see MassiveClient.get_latest_price), then add/
    update that symbol on the watchlist in Lakebase.
    """
    ensure_watchlist_table()

    if request.is_json:
        symbol = request.json.get("symbol", "")
    else:
        symbol = request.form.get("symbol", "")

    symbol = symbol.strip().upper() if isinstance(symbol, str) else ""

    if not symbol or not _TICKER_RE.match(symbol):
        return jsonify({"error": f"Invalid ticker symbol: {symbol!r}"}), 400

    client = MassiveClient()
    try:
        data = client.get_latest_price(symbol)  # <-- single API call, latest price only
    except requests.HTTPError:
        # Massive returns a 404/4xx for tickers it doesn't recognize.
        return jsonify({"error": f"Unknown ticker symbol: {symbol}"}), 400

    price = _extract_latest_price(data)
    if price is None:
        # No usable price in the response (e.g. delisted/invalid ticker
        # that still 200s with an empty result set) - don't add it.
        return jsonify({"error": f"No price data available for ticker: {symbol}"}), 400

    email = _current_user_email()

    lakebase.run_write(
        f"""
        INSERT INTO {WATCHLIST_TABLE_NAME} (symbol, email, latest_price, updated_at)
        VALUES (%s, %s, %s, now())
        ON CONFLICT (symbol, email) DO UPDATE
            SET latest_price = EXCLUDED.latest_price,
                updated_at = EXCLUDED.updated_at
        """,
        (symbol, email, price),
    )

    return jsonify({"symbol": symbol, "email": email, "latest_price": price})


@app.route("/watchlist/<symbol>", methods=["DELETE"])
def delete_from_watchlist(symbol: str):
    """Remove a single symbol from the current user's watchlist."""
    ensure_watchlist_table()

    symbol = symbol.strip().upper() if isinstance(symbol, str) else ""
    if not symbol or not _TICKER_RE.match(symbol):
        return jsonify({"error": f"Invalid ticker symbol: {symbol!r}"}), 400

    email = _current_user_email()
    deleted = lakebase.run_write(
        f"DELETE FROM {WATCHLIST_TABLE_NAME} WHERE symbol = %s AND email = %s",
        (symbol, email),
    )

    if not deleted:
        return jsonify({"error": f"{symbol} is not on your watchlist"}), 404

    return jsonify({"symbol": symbol, "email": email, "deleted": True})


def _extract_latest_price(data: dict) -> float | None:
    """Pull the trade price out of the Massive 'previous close' response shape.

    The /v2/aggs/ticker/{symbol}/prev endpoint returns "results" as a LIST
    containing a single aggregate bar (not a dict), e.g.:
        {"status": "OK", "resultsCount": 1, "results": [{"c": 148.845, ...}]}
    Previously this code treated "results" as a dict, so isinstance(results, dict)
    was always False for this endpoint's real shape and the price silently
    resolved to None. Unwrap the list here, and check "status"/"resultsCount"
    so invalid tickers (empty results) are detected instead of "succeeding"
    with a null price.

    Adjust the key lookup here if the real Massive API returns a different
    field name for the traded/close price.
    """
    if not isinstance(data, dict):
        return None
    if data.get("status") not in (None, "OK") or data.get("resultsCount") == 0:
        return None
    results = data.get("results", data)
    if isinstance(results, list):
        results = results[0] if results else None
    if isinstance(results, dict):
        for key in ("c", "p", "price", "last_price", "vw"):
            if key in results:
                return results[key]
    return None


def _upsert_batch(items: list[dict]) -> int:
    """Upsert a batch of Massive API items into Lakebase, one statement per row.

    For very large batches, consider psycopg2.extras.execute_values for
    higher throughput instead of per-row execute calls.
    """
    import json as _json

    count = 0
    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            for item in items:
                cur.execute(
                    f"""
                    INSERT INTO {TABLE_NAME} (id, payload, synced_at)
                    VALUES (%s, %s, now())
                    ON CONFLICT (id) DO UPDATE
                        SET payload = EXCLUDED.payload,
                            synced_at = EXCLUDED.synced_at
                    """,
                    (str(item.get("id")), _json.dumps(item)),
                )
                count += 1
            conn.commit()
    return count


def _upsert_news_batch(ticker: str, articles: list[dict]) -> int:
    """Upsert news articles for a single ticker into the news documents table.

    Flattens the top-level "insights" sentiment entry that matches this
    ticker (if present) into its own columns so the Spark notebook can read
    plain text columns instead of parsing JSONB for the common case.
    """
    import json as _json

    count = 0
    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            for article in articles:
                sentiment = None
                sentiment_reasoning = None
                for insight in article.get("insights", []) or []:
                    if insight.get("ticker") == ticker:
                        sentiment = insight.get("sentiment")
                        sentiment_reasoning = insight.get("sentiment_reasoning")
                        break

                publisher = article.get("publisher") or {}
                cur.execute(
                    f"""
                    INSERT INTO {NEWS_TABLE_NAME} (
                        id, ticker, title, description, author, article_url,
                        publisher_name, keywords, sentiment, sentiment_reasoning,
                        published_utc, payload, synced_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (id) DO UPDATE
                        SET ticker = EXCLUDED.ticker,
                            title = EXCLUDED.title,
                            description = EXCLUDED.description,
                            author = EXCLUDED.author,
                            article_url = EXCLUDED.article_url,
                            publisher_name = EXCLUDED.publisher_name,
                            keywords = EXCLUDED.keywords,
                            sentiment = EXCLUDED.sentiment,
                            sentiment_reasoning = EXCLUDED.sentiment_reasoning,
                            published_utc = EXCLUDED.published_utc,
                            payload = EXCLUDED.payload,
                            synced_at = EXCLUDED.synced_at
                    """,
                    (
                        str(article.get("id")),
                        ticker,
                        article.get("title", ""),
                        article.get("description"),
                        article.get("author"),
                        article.get("article_url"),
                        publisher.get("name"),
                        _json.dumps(article.get("keywords", [])),
                        sentiment,
                        sentiment_reasoning,
                        article.get("published_utc"),
                        _json.dumps(article),
                    ),
                )
                count += 1
            conn.commit()
    return count


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


if __name__ == '__main__':
    host = os.getenv('FLASK_RUN_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_RUN_PORT', 8000))
    app.run(debug=True, host=host, port=port)
    print(f"Flask app running on http://{host}:{port}")