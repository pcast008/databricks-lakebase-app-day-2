-- Setup script for weather_embeddings table (WHOLE-DOCUMENT embeddings)
-- Run this manually in your Lakebase Postgres database before running the ingestion script.
--
-- Mirrors sql/02_setup_embeddings_table.sql (ticker_news_embeddings).
-- Stores ONE embedding per weather document: the whole narrative_text embedded
-- as a single vector. For the chunked version (many vectors per document), see
-- 03_setup_weather_chunk_embeddings_table.sql.

-- Enable pgvector extension (already enabled in this Lakebase instance, safe to repeat).
CREATE EXTENSION IF NOT EXISTS vector;

-- VECTOR(384) hardcoded for sentence-transformers/all-MiniLM-L6-v2.
--   - sentence-transformers/all-MiniLM-L6-v2: 384  <-- used here
--   - sentence-transformers/all-mpnet-base-v2: 768
--   - BAAI/bge-small-en-v1.5: 384
--   - BAAI/bge-base-en-v1.5: 768
--   - BAAI/bge-large-en-v1.5: 1024
CREATE TABLE IF NOT EXISTS weather_embeddings (
    -- id == weather_documents.id (one embedding per document).
    id          TEXT PRIMARY KEY REFERENCES weather_documents (id) ON DELETE CASCADE,

    -- Denormalized for convenience at query time (mirrors ticker on the news table).
    location    TEXT NOT NULL,
    headline    TEXT,
    issued_at   TIMESTAMPTZ,

    -- 384-dim embedding of the full narrative_text.
    embedding   VECTOR(384) NOT NULL,

    model_name  TEXT NOT NULL,
    embedded_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- HNSW index for fast cosine similarity search (pgvector's <=> operator).
CREATE INDEX IF NOT EXISTS idx_weather_embeddings_embedding
ON weather_embeddings
USING hnsw (embedding vector_cosine_ops);

-- Verify the table was created
SELECT
    table_name,
    column_name,
    data_type,
    udt_name
FROM information_schema.columns
WHERE table_name = 'weather_embeddings'
ORDER BY ordinal_position;
