# Weather Intelligence — Homework Notes

## Which data source I chose and why

The National Weather Service API, [`api.weather.gov`](https://api.weather.gov).
Honestly, I chose it because it was the source suggested in the homework and I
needed to move quickly. All weather APIs are broadly similar — each with its own
pros and cons — so it wasn't a big deal. It also needs no API key (just a
`User-Agent` header), which kept the focus on harvesting/vectorization/retrieval.

## Schema decisions

I mirrored the ticker-news schema (a raw documents table + embeddings tables) and
added a `source_type` column (`alert` | `forecast` | `hourly`) for the
multi-source extra credit.

- **`weather_documents`** (raw): `id`, `location`, `source_type`, `headline`,
  `event`, `narrative_text`, `issued_at`, `effective_at`, `payload`, `synced_at`.
- **`weather_embeddings`** (one vector per document): `id`, `location`,
  `source_type`, `headline`, `issued_at`, `embedding VECTOR(384)`, `model_name`,
  `embedded_at`.
- **`weather_chunk_embeddings`** (retrieval target): `id`, `document_id`,
  `location`, `source_type`, `chunk_index`, `chunk_text`, `embedding VECTOR(384)`,
  `model_name`, `embedded_at`.
- **Chunking:** sliding window, 800-char chunks with 100-char overlap (short
  narratives produce a single chunk).
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` → `VECTOR(384)`,
  cosine similarity via an HNSW index (`vector_cosine_ops`).

## How to run the pipeline end-to-end

1. **Create tables:** run `sql/weather/01`, `02`, `03_*.sql` in the Databricks SQL
   editor (as table owner — they `DROP … CASCADE` + recreate).
2. **Sync (seed):** run `notebooks/seed_weather.py`, choosing states via its
   widgets (default `TX,FL`). Re-runs are idempotent (`ON CONFLICT (id) DO
   UPDATE`), so states aren't duplicated. Same code path as `POST /weather/sync`.
3. **Embed:** run `notebooks/ingest_weather_embeddings.py` to populate the two
   embeddings tables.
4. **Search:** run `notebooks/search_weather.py` (query / `top_k` / `source_type`
   widgets), or call `POST /weather/search`:
   ```bash
   curl -X POST "https://<app-url>/weather/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "flash flood risk this weekend", "top_k": 5, "source_type": "alert"}'
   ```

## Known limitations / what I'd improve with more time

- I'd build the full frontend UI to see the weather data and search in action.
- I'd spend more time tuning the chunking and digesting the ingest notebook to
  really understand what's happening.
- NWS is US-only, the harvest bootstraps off active alerts (a state with none
  returns 0 documents), hourly narrative is synthesized, expired rows aren't
  pruned, and the LLM RAG-summary stretch goal isn't built.
