# Homework: Weather Intelligence — Unstructured Data → Lakebase Vector Search → REST API

## Overview

In the `databricks-lakebase-app-day-2` reference app, we synced structured records and news articles from the Massive API into Lakebase, chunked/embedded the news text with `sentence-transformers`, and stored the vectors in `pgvector` columns for retrieval-augmented generation.

For this homework, you'll build the same pipeline end-to-end, but for a new unstructured data source: **weather**. You will:

1. Harvest unstructured weather text from a public API.
2. Vectorize that text and load it into Lakebase (Postgres + `pgvector`).
3. Add a retrieval endpoint to the Flask REST API that performs semantic search over the ingested weather documents.

By the end, a user should be able to `POST /weather/search {"query": "flash flood risk this weekend"}` and get back the most semantically relevant weather documents, ranked by vector similarity.

## Learning Objectives

* Harvest unstructured (free-text) data from a real-world API and normalize it into a document schema.
* Design a Postgres/pgvector schema for storing raw documents + embeddings (mirroring the `ticker_news_documents` / `ticker_news_embeddings` pattern).
* Chunk long text and embed it, matching the project's existing embedding dimensionality conventions.
* Write a Python batch job that writes vectors into Lakebase via `psycopg2`.
* Implement a cosine-similarity retrieval endpoint in Flask using pgvector's `<=>` operator.

## Recommended Data Source: National Weather Service API (`api.weather.gov`)

We recommend the NWS API because:

* It's free, requires no API key, and has generous rate limits.
* It returns rich unstructured narrative text — perfect for embedding:
  * `GET /alerts/active?area={state}` → active weather alerts, each with a free-text `description` and `instruction` field (e.g., "A Flash Flood Warning means...").
  * `GET /gridpoints/{office}/{x},{y}/forecast` → multi-day forecast with a narrative `detailedForecast` string per period (e.g., "Sunny, with a high near 78. Northwest wind around 6 mph.").
  * `GET /gridpoints/{office}/{x},{y}/forecast/hourly` → hourly narrative forecasts.
* No API key means you can focus on the harvesting/vectorization/retrieval logic instead of auth plumbing.

Alternative sources (if you want a different flavor of unstructured text): OpenWeatherMap's `weather` + `alerts` fields, or NOAA's Climate Prediction Center discussion text products (`https://www.cpc.ncep.noaa.gov`). Pick one and justify your choice in your README — don't mix sources unless you want extra credit for a multi-source pipeline.

## Part 1 — Harvest (Ingestion)

Build a client module (mirror `massive_client.py`), e.g. `weather_client.py`, that:

* Given a list of locations (city/state, or lat/lon pairs), resolves each to a NWS grid point via `GET /points/{lat},{lon}`.
* Fetches active alerts and forecast discussions for each location.
* Normalizes each item into a document record with at least:
  * `id` (stable dedup key — e.g. alert `id` field, or `location + issued_at` hash for forecasts)
  * `location` (city/state or lat/lon)
  * `source_type` (`"alert"` or `"forecast"`)
  * `headline` / `event` (e.g. "Flash Flood Warning")
  * `narrative_text` (the free-text body to embed — `description`, `instruction`, or `detailedForecast`)
  * `issued_at` / `effective_at` timestamp
  * `payload` (raw JSON, for provenance)
  * `synced_at`

Write these into a new Lakebase table, e.g. `weather_documents`, following the same connection pattern as `lakebase.py` (`get_connection()` context manager, `psycopg2` + `RealDictCursor`).

Add a Flask endpoint to trigger this:

```
POST /weather/sync
Body: {"locations": ["Chicago, IL", "Austin, TX"], "limit": 50}
```

This should behave like `/news/sync` — fetch, normalize, and upsert into `weather_documents`, returning a count of documents synced.

## Part 2 — Vectorize (Embedding Pipeline)

Build a Python ingestion script (mirror `notebooks/ingest_ticker_news_embeddings.py`, but written as a plain Python script/notebook using `psycopg2` — do not use `spark.write.jdbc`, which does not work reliably against Lakebase in this environment) that:

1. Reads unembedded rows from `weather_documents` via `psycopg2` (using the same `get_connection()` helper as `lakebase.py`).
2. Chunks `narrative_text` for any documents longer than your chosen chunk size (reuse the sliding-window pattern: `CHUNK_SIZE=800`, `CHUNK_OVERLAP=100`, or justify your own values — most NWS text is short enough that chunking may only matter for combined alert+instruction text).
3. Embeds each chunk using `sentence-transformers/all-MiniLM-L6-v2` (384-dim) — use the same model as the existing news pipeline so both stay compatible/queryable with the same distance operator conventions. (If you pick a different model, document the dimensionality and update the schema accordingly.)
4. Writes embeddings into a new table, e.g. `weather_embeddings`:
   * `id`, `document_id` (FK to `weather_documents.id`), `chunk_index`, `chunk_text`, `embedding vector(384)`, `model_name`, `created_at`
5. Writes via `psycopg2` — use `execute_values` (from `psycopg2.extras`) or batched `INSERT ... ON CONFLICT` statements for reasonable throughput. Cast the embedding to `%s::vector` in your SQL (pass the embedding as a Python list/stringified array — psycopg2 + pgvector's adapter will handle the cast) rather than relying on Spark's JDBC `stringtype=unspecified` trick, since Spark JDBC writes are not supported against this Lakebase instance.
6. If you need distributed/parallel processing for large batches, use plain Python (e.g. `concurrent.futures.ThreadPoolExecutor`) or simply batch your `psycopg2` inserts — do not reach for Spark for the write path.

You must create the `weather_embeddings` table with a proper `vector(384)` column (requires the `pgvector` extension, already enabled in this Lakebase instance) and an appropriate index (e.g. `CREATE INDEX ... USING hnsw (embedding vector_cosine_ops)` or `ivfflat`) for retrieval performance.

## Part 3 — Retrieve (REST API)

Add a new endpoint to `app.py`:

```
POST /weather/search
Body: {"query": "risk of flooding near rivers", "top_k": 5}
```

This endpoint should:

1. Embed the `query` string using the same model used for ingestion (load it once at module/app level, not per-request).
2. Run a cosine-similarity search against `weather_embeddings` using pgvector's `<=>` operator, executed via `psycopg2`:

```sql
SELECT d.id, d.location, d.headline, d.narrative_text, e.chunk_text,
       1 - (e.embedding <=> %s::vector) AS similarity
FROM weather_embeddings e
JOIN weather_documents d ON d.id = e.document_id
ORDER BY e.embedding <=> %s::vector
LIMIT %s;
```

3. Return the top `top_k` matches as JSON, each with `location`, `headline`, `chunk_text`, and `similarity` score.

Edge cases to handle: empty `weather_embeddings` table (no data synced yet), malformed/missing `query`, `top_k` bounds (clamp to e.g. 1–20).

## Deliverables

1. `weather_client.py` — NWS API client (or your chosen source).
2. Updated `app.py` with `POST /weather/sync` and `POST /weather/search`.
3. Updated `lakebase.py` (or a new module) with DDL/migration for `weather_documents` and `weather_embeddings`.
4. A `psycopg2`-based embedding ingestion script (`notebooks/ingest_weather_embeddings.py` or equivalent scripts).
5. A short `README_WEATHER.md` (or a section in the main README) explaining:
   * Which data source you chose and why.
   * Your schema decisions (columns, chunking parameters, embedding model/dimensions).
   * How to run the sync → embed → search pipeline end-to-end.
   * Any known limitations or things you'd improve given more time.

## Stretch Goals (optional, for extra credit)

* Add a `GET /weather/search?query=...` variant that also returns an LLM-generated natural-language summary of the top results (basic RAG).
* Deduplicate/upsert on `id` so re-running `/weather/sync` doesn't create duplicate rows.
* Add a scheduled Databricks Job (or simple cron) that re-syncs alerts every N minutes.
* Combine two data sources (e.g. alerts + forecast discussions) and let retrieval filter by `source_type`.
* Add a `CREATE INDEX ... USING hnsw` benchmark comparing query latency with vs. without the index.
