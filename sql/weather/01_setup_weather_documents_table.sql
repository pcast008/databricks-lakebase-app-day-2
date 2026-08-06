-- Setup script for weather_documents table
-- Run this manually in your Lakebase Postgres database before running the ingestion script.
--
-- Mirrors sql/01_setup_news_table.sql (ticker_news_documents) but for unstructured
-- weather text harvested from the National Weather Service API (api.weather.gov).
-- Each row is one normalized weather "document" (an alert or a forecast period)
-- whose narrative_text is the free-text body we later chunk + embed.

-- Create the weather documents table
CREATE TABLE IF NOT EXISTS weather_documents (
    -- Stable dedup key: alert `id` field for alerts, or a hash of
    -- location + issued_at for forecasts. Used for upsert on re-sync.
    id             TEXT PRIMARY KEY,

    -- City/State (e.g. "Chicago, IL") or "lat,lon" string.
    location       TEXT NOT NULL,

    -- "alert" or "forecast" (lets retrieval optionally filter by kind).
    source_type    TEXT NOT NULL,

    -- Short human-readable label, e.g. "Flash Flood Warning" or "Tonight".
    headline       TEXT,

    -- Event / phenomenon, e.g. "Flash Flood Warning", "Winter Storm Watch".
    event          TEXT,

    -- The free-text body we embed: alert description/instruction or
    -- forecast detailedForecast. This is the semantic search target.
    narrative_text TEXT NOT NULL,

    -- When the alert/forecast was issued or became effective.
    issued_at      TIMESTAMPTZ,
    effective_at   TIMESTAMPTZ,

    -- Raw JSON from the NWS API, kept for provenance / reprocessing.
    payload        JSONB NOT NULL,

    -- When this row was harvested into Lakebase.
    synced_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Index for location lookups (mirrors the ticker index on the news table).
CREATE INDEX IF NOT EXISTS idx_weather_documents_location
ON weather_documents (location);

-- Index for filtering by source_type (alerts vs forecasts).
CREATE INDEX IF NOT EXISTS idx_weather_documents_source_type
ON weather_documents (source_type);

-- Verify the table was created
SELECT
    table_name,
    column_name,
    data_type
FROM information_schema.columns
WHERE table_name = 'weather_documents'
ORDER BY ordinal_position;
