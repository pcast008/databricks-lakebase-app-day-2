# Weather Embeddings Notebook — Design

**Date:** 2026-08-07
**File to create:** `notebooks/ingest_weather_embeddings.py`

## Goal

Mirror `notebooks/ingest_ticker_news_embeddings.py` for the weather pipeline:
read already-seeded `weather_documents`, compute sentence embeddings, and write
them to `weather_embeddings` (one vector per document) and
`weather_chunk_embeddings` (one vector per sliding-window chunk) so the
`POST /weather/search` retrieval exercise can run pgvector similarity search.

## Scope decisions (confirmed with user)

- **Embeddings-only.** The notebook does NOT harvest weather. It assumes
  `weather_documents` is already populated by `notebooks/seed_weather.py` (which
  wraps `weather_sync.sync_weather()`). The notebook reads → embeds → writes.
- **Full mirror of the news pipeline: both embedding tables.** Whole-document
  embeddings into `weather_embeddings` AND sliding-window chunk embeddings into
  `weather_chunk_embeddings`. Both DDL files already exist under `sql/weather/`.

## Non-goals

- No Massive/NWS API calls, no `weather_sync` import, no URL fetching, no
  `trafilatura` (the news notebook needed it to fetch article bodies; weather's
  `narrative_text` IS the body already).
- No changes to `app.py`, `weather_sync.py`, `weather_client.py`, or the SQL DDL.

## Source and destination schemas (already exist)

`weather_documents` (source): `id, location, source_type, headline, event,
narrative_text, issued_at, effective_at, payload, synced_at`.

`weather_embeddings` (whole-doc dest): `id` (= document id, FK), `location`,
`headline`, `issued_at`, `embedding VECTOR(384)`, `model_name`, `embedded_at`.

`weather_chunk_embeddings` (chunk dest): `id` (= `document_id + '_' + chunk_index`),
`document_id` (FK), `location`, `chunk_index INT`, `chunk_text`,
`embedding VECTOR(384)`, `model_name`, `embedded_at`. Unique index on
`(document_id, chunk_index)`.

## Notebook structure (Databricks `# COMMAND ----------` cells)

1. **Install deps** — `%pip uninstall -y psycopg2 psycopg2-binary` then
   `%pip install -q 'databricks-sdk>=0.118.0' sentence-transformers pandas`.
   (Same psycopg2/OpenSSL SIGABRT ritual as the two existing notebooks. No
   `trafilatura`/`requests`.)
2. **`dbutils.library.restartPython()`**
3. **Config widgets** — `documents_table_name` (`weather_documents`),
   `embeddings_table_name` (`weather_embeddings`),
   `chunk_embeddings_table_name` (`weather_chunk_embeddings`),
   `embedding_model` (`sentence-transformers/all-MiniLM-L6-v2`),
   `chunk_size` (800), `chunk_overlap` (100). Reuse the news notebook's
   `match EMBEDDING_MODEL_NAME` block that maps model name → `EMBEDDING_DIM`.
4. **Resolve Lakebase URL** — parse the `database` / `lakebase-url` base64 secret
   into host/port/dbname/user/password, identical to the news notebook.
5. **Test connection** — psycopg2 connect; `SELECT COUNT(*) FROM weather_documents`.
6. **Prereq markdown** — remind the user to run `seed_weather.py` first and to have
   applied `sql/weather/02_setup_weather_embeddings_table.sql` and
   `sql/weather/03_setup_weather_chunk_embeddings_table.sql`.
7. **Load documents** — `pd.read_sql_query` selecting
   `id, location, headline, issued_at, narrative_text` and computing
   `embedding_text = TRIM(CONCAT(COALESCE(headline,''), '. ', narrative_text))`,
   filtered to non-empty `embedding_text`.
8. **Compute whole-doc embeddings** — load `SentenceTransformer` once
   (HF cache under `/tmp/.cache/huggingface`), encode `embedding_text` in
   batches of 32.
9. **Insert whole-doc embeddings** — `execute_values` into `weather_embeddings`
   (`id, location, headline, issued_at, embedding, model_name, embedded_at`),
   `ON CONFLICT (id) DO NOTHING`.
10. **Chunk `narrative_text`** — sliding window (`chunk_size`,`chunk_overlap`),
    reusing the news chunk loop but over in-hand text (no HTTP/trafilatura).
    Produce rows of `document_id, location, chunk_index, chunk_text`.
11. **Compute chunk embeddings** — reuse the loaded model, batch-encode
    `chunk_text`.
12. **Insert chunk embeddings** — into `weather_chunk_embeddings`
    (`id, document_id, location, chunk_index, chunk_text, embedding, model_name,
    embedded_at`), `ON CONFLICT (id) DO NOTHING`.

## Key improvement over the news notebook

The `weather_embeddings` / `weather_chunk_embeddings` tables declare the column
as `VECTOR(384)` directly (the news tables used `double precision[]` and required
a manual `UPDATE ... SET embedding = embedding::vector` afterward). This notebook
formats each vector as a pgvector literal `'[v1,v2,...]'` and casts with
`::vector` in the INSERT itself, so **no manual post-cast SQL step is needed.**

## Error handling / edge cases

- Empty `weather_documents` → both embedding steps print "nothing to embed" and
  no-op rather than erroring.
- Re-runs are idempotent via `ON CONFLICT (id) DO NOTHING`.
- Documents whose `narrative_text` yields no non-empty chunk are skipped.
