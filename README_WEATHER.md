# Weather Intelligence — NWS → Lakebase Vector Search → REST API

This is the weather homework built on top of the `databricks-lakebase-app-day-2`
reference app. It harvests unstructured weather text from the National Weather
Service, chunks and embeds it into Lakebase `pgvector` columns, and exposes a
semantic-search REST endpoint (`POST /weather/search`) that returns the most
relevant weather documents ranked by vector similarity.

It mirrors the existing ticker-news pipeline (raw documents → chunked embeddings
→ pgvector retrieval), and additionally implements the **multi-source extra
credit**: alerts, daily forecasts, and hourly forecasts all live in one table
distinguished by `source_type`, and retrieval can filter by it.

---

## 1. Data source & why

**Source:** the National Weather Service API, [`api.weather.gov`](https://api.weather.gov).

Honestly, I chose it because it was the source suggested in the homework and I
needed to move quickly. All weather APIs are broadly similar — each with its own
pros and cons — so the specific choice wasn't a big deal here. It's also a
convenient fit in practice: it needs **no API key** (only a descriptive
`User-Agent` header), so there's no auth plumbing to get in the way of the
harvesting / vectorization / retrieval work that this assignment is actually
about.

---

## 2. Schema decisions

I **mirrored the ticker-news schema** (a raw documents table plus embeddings
tables), and added a **`source_type`** column to support the multi-source
pipeline / extra credit. `source_type` is one of `alert` | `forecast` | `hourly`.

### Tables

**`weather_documents`** — one row per normalized NWS "document" (an alert or a
forecast period); `narrative_text` is the free-text body we later chunk + embed.

| column | notes |
| --- | --- |
| `id` | PK. Alert `id` for alerts; deterministic md5 hash of location + period + start time for forecast/hourly (no stable API id). |
| `location` | "City, ST" or "lat,lon". |
| `source_type` | `alert` \| `forecast` \| `hourly`. |
| `headline` | Short label, e.g. "Flash Flood Warning" / "Tonight". |
| `event` | Phenomenon, e.g. "Flash Flood Warning". |
| `narrative_text` | The free-text body we embed (semantic-search target). |
| `issued_at`, `effective_at` | Timestamps. |
| `payload` | Raw NWS JSON (provenance / reprocessing). |
| `synced_at` | When harvested into Lakebase. |

**`weather_embeddings`** — whole-document vectors (one embedding per document).
Columns: `id` (FK → `weather_documents.id`), `location`, `source_type`,
`headline`, `issued_at`, `embedding VECTOR(384)`, `model_name`, `embedded_at`.

**`weather_chunk_embeddings`** — chunk-level vectors (the retrieval target for
`/weather/search`). Columns: `id` (`document_id` + `_` + `chunk_index`),
`document_id` (FK → `weather_documents.id`), `location`, `source_type`,
`chunk_index`, `chunk_text`, `embedding VECTOR(384)`, `model_name`,
`embedded_at`.

### Chunking parameters

Long alert narratives (description + instruction can run several paragraphs)
retrieve better when split into smaller passages, so `narrative_text` is split
with a **sliding window**:

- **chunk size:** 800 characters
- **chunk overlap:** 100 characters
- Short forecast/hourly narratives simply produce a single chunk.

### Embedding model / dimensions

- **Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensions:** `VECTOR(384)`
- **Similarity:** cosine — an **HNSW** index (`vector_cosine_ops`) backs the
  `<=>` operator; a secondary btree index on `source_type` backs the filter.

---

## 3. How to run the pipeline end-to-end

The whole flow runs from Databricks (SQL editor + notebooks). Order matters:

1. **Create the tables (DDL).** Run these three files in the Databricks SQL
   editor, as the table **owner** (they `DROP … CASCADE` + recreate):
   - `sql/weather/01_setup_weather_documents_table.sql`
   - `sql/weather/02_setup_weather_embeddings_table.sql`
   - `sql/weather/03_setup_weather_chunk_embeddings_table.sql`

2. **Seed the raw documents.** Run `notebooks/seed_weather.py`, choosing the
   states to harvest via its widgets (defaults to `TX,FL`). This pulls active
   NWS alerts per state, then follows each alert to its forecast grid cell for
   the daily + hourly narratives, and upserts them into `weather_documents`.
   > Pick states that currently have active alerts — the harvest bootstraps off
   > active alerts, so a state with none yields 0 documents.

3. **Compute embeddings.** Run `notebooks/ingest_weather_embeddings.py`. It reads
   `weather_documents`, writes whole-document vectors to `weather_embeddings`,
   and sliding-window chunk vectors (with `source_type`) to
   `weather_chunk_embeddings`.

4. **Search.** Either:
   - **Notebook (quickest):** run `notebooks/search_weather.py`. Type your query
     into the **query** widget, optionally set **source_type** / **top_k**, and
     re-run the search cell. It runs the exact same SQL as the endpoint, so it
     isolates the retrieval logic from app deployment / auth.
   - **REST endpoint:** `POST /weather/search` on the deployed Databricks App:
     ```bash
     curl -X POST "https://<your-app-url>/weather/search" \
       -H "Authorization: Bearer <databricks-token>" \
       -H "Content-Type: application/json" \
       -d '{"query": "flash flood risk this weekend", "top_k": 5, "source_type": "alert"}'
     ```
     Returns the top matches, each with `location`, `headline`, `chunk_text`,
     `source_type`, and `similarity` (= `1 - cosine_distance`).

You can also (re)harvest documents at any time via `POST /weather/sync`
(`{"states": ["TX","FL"], "limit": 50}`), which is the same code path as
`seed_weather.py`.

> **Notebook note:** every notebook here starts by uninstalling the
> `psycopg2`/`psycopg2-binary` pip wheel and calling `restartPython()`. That
> wheel bundles its own OpenSSL, which collides with `databricks-sdk`/grpc and
> crashes the kernel ("The Python kernel is unresponsive"). The uninstall +
> restart makes the runtime's system psycopg2 load instead.

### Re-running is idempotent

Re-seeding the same states does **not** create duplicates. Every row upserts by
`id` (`ON CONFLICT (id) DO UPDATE`):

- **Alerts** use the stable NWS alert `id` → an already-seeded alert is updated
  in place; newly-active alerts are inserted.
- **Forecast/hourly** use a deterministic hash of location + period + start time
  → the same period re-fetched updates in place; as time advances, new periods
  get new ids and are inserted.
- Note: **old / expired rows are not pruned** — they linger until cleaned up
  manually (see limitations).

---

## Extra credit: multi-source retrieval

The base assignment says *"don't mix sources unless you want extra credit for a
multi-source pipeline,"* and the extra-credit goal is *"combine two data sources
and let retrieval filter by `source_type`."* This project does exactly that:

- **Three sources in one table.** `weather_client.py` normalizes NWS **active
  alerts**, **daily forecast periods**, and **hourly forecast periods** into
  `weather_documents`, each tagged with `source_type` (`alert` / `forecast` /
  `hourly`). Hourly periods have no narrative from the API, so their
  `narrative_text` is synthesized from `shortForecast` + temperature + wind +
  precip chance.
- **`source_type` flows all the way to the vectors.** It's denormalized onto
  both `weather_embeddings` and `weather_chunk_embeddings`, so retrieval can
  filter by kind **without joining back** to the documents table.
- **Filtered retrieval.** `POST /weather/search` (and `search_weather.py`) accept
  an optional `source_type` — e.g. search only alerts for "flash flood risk" —
  which adds `WHERE e.source_type = %s` in front of the cosine `ORDER BY`, backed
  by a btree index on `source_type`.

---

## 4. Known limitations / things I'd improve with more time

**What I'd do with more time (my priorities):**

- **Build the full frontend UI** to actually *see* the weather data and the
  search working in action, instead of testing purely through the notebook /
  REST endpoint.
- **Spend more time tuning the chunking** (size / overlap) and digging through
  the ingest notebook to really understand what's happening at each step.

**Other known limitations:**

- **US-only coverage.** NWS only covers the United States; a global app would
  need a different or additional source.
- **Harvest bootstraps off active alerts.** The state → forecast chain starts
  from active alerts, so a state with no current alerts returns 0 documents.
- **Hourly narrative is synthesized**, not authored by NWS — it's assembled from
  structured fields, so it reads more mechanically than alert / daily text.
- **Stale rows aren't pruned.** Expired alerts and past forecast periods remain
  in the tables after re-syncs (upsert never deletes).
- **RAG summary not implemented.** The `GET /weather/search?query=…` stretch goal
  that returns an LLM-generated natural-language summary of the top results
  isn't built yet.

---

## File map

| File | Role |
| --- | --- |
| `weather_client.py` | NWS API client + `normalize_alert` / `normalize_forecast_period` / `normalize_hourly_period`. |
| `weather_sync.py` | Flask-free harvest core: `sync_weather()`, `parse_states()`, upsert. |
| `seed_weather.py` / `notebooks/seed_weather.py` | Seed `weather_documents` (standalone script + Databricks notebook). |
| `notebooks/ingest_weather_embeddings.py` | Embeddings pipeline (psycopg2 + `execute_values`). |
| `notebooks/search_weather.py` | Interactive cosine-search notebook (mirrors the endpoint). |
| `app.py` | Flask API: `POST /weather/sync`, `POST /weather/search`. |
| `sql/weather/01–03_*.sql` | DDL for the three tables. |
