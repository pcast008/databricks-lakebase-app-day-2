-- Setup script for weather_embeddings table
-- Run this manually in your Lakebase Postgres database before running the ingestion script.
--
-- Mirrors sql/03_setup_chunk_embeddings_table.sql (ticker_news_chunk_embeddings).
-- Stores one row per (document chunk) with its embedding vector.
--
-- Dimension is HARDCODED to 384 for sentence-transformers/all-MiniLM-L6-v2
-- (the same model as the existing news pipeline). If you switch models,
-- update the VECTOR(...) dimension below to match:
--   - sentence-transformers/all-MiniLM-L6-v2: 384  <-- used here
--   - sentence-transformers/all-mpnet-base-v2: 768
--   - BAAI/bge-small-en-v1.5: 384
--   - BAAI/bge-base-en-v1.5: 768
--   - BAAI/bge-large-en-v1.5: 1024

-- Enable pgvector extension (already enabled in this Lakebase instance, but safe to repeat).
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the weather embeddings table
CREATE TABLE IF NOT EXISTS weather_embeddings (
    id           TEXT PRIMARY KEY,

    -- FK to weather_documents.id (the document this chunk came from).
    document_id  TEXT NOT NULL REFERENCES weather_documents (id) ON DELETE CASCADE,

    -- 0-based index of the chunk within the document.
    chunk_index  INT NOT NULL,

    -- The exact text that was embedded (one sliding-window chunk).
    chunk_text   TEXT NOT NULL,

    -- 384-dim embedding from all-MiniLM-L6-v2.
    embedding    VECTOR(384) NOT NULL,

    -- Model used, for reproducibility / compatibility checks.
    model_name   TEXT NOT NULL,

    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Prevent duplicate chunks for the same document on re-runs.
CREATE UNIQUE INDEX IF NOT EXISTS idx_weather_embeddings_doc_chunk
ON weather_embeddings (document_id, chunk_index);

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
