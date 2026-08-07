# Databricks notebook source
# MAGIC %md
# MAGIC # Search Weather Documents (Lakebase pgvector cosine retrieval)
# MAGIC
# MAGIC Interactive, notebook-based counterpart to `POST /weather/search`. It runs
# MAGIC the **exact same retrieval** the Flask endpoint runs, so you can test the
# MAGIC search logic without deploying the app or dealing with app OAuth.
# MAGIC
# MAGIC It:
# MAGIC 1. Reads a query string (+ optional `source_type` filter and `top_k`) from
# MAGIC    notebook widgets.
# MAGIC 2. Embeds the query with the **same** sentence-transformers model used at
# MAGIC    ingestion (`notebooks/ingest_weather_embeddings.py`) so the query vector
# MAGIC    and the stored chunk vectors are comparable.
# MAGIC 3. Runs a pgvector cosine (`<=>`) search over `weather_chunk_embeddings`
# MAGIC    JOINed to `weather_documents`, returning matches ranked by
# MAGIC    `similarity = 1 - cosine_distance`.
# MAGIC
# MAGIC **Prerequisites:** `weather_documents`, `weather_embeddings`, and
# MAGIC `weather_chunk_embeddings` are populated - i.e. you have already run
# MAGIC `notebooks/seed_weather.py` then `notebooks/ingest_weather_embeddings.py`.

# COMMAND ----------

# DBTITLE 1,Install required packages
# MAGIC %pip uninstall -y psycopg2 psycopg2-binary
# MAGIC %pip install -q 'databricks-sdk>=0.118.0' sentence-transformers pandas

# COMMAND ----------

# MAGIC %md
# MAGIC The pip `psycopg2`/`psycopg2-binary` wheel bundles its own OpenSSL, which
# MAGIC collides with the OpenSSL that `databricks-sdk`/grpc uses in a background
# MAGIC credential-refresh thread and SIGABRTs the kernel ("The Python kernel is
# MAGIC unresponsive"). We uninstall it and restart Python so the runtime's system
# MAGIC psycopg2 is what loads (same ritual as the ingest + seed notebooks).

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config
# MAGIC
# MAGIC Type your query into the **query** widget (defaults to a flood example).
# MAGIC `source_type` filters retrieval to one kind of document (the multi-source
# MAGIC extra credit); leave it on `all` to search across alerts + forecasts +
# MAGIC hourly. `embedding_model` MUST match the model used at ingestion.

# COMMAND ----------

dbutils.widgets.text("query", "flash flood risk this weekend", "Search query")
dbutils.widgets.text("top_k", "5", "Number of results (clamped 1-20)")
dbutils.widgets.dropdown(
    "source_type", "all", ["all", "alert", "forecast", "hourly"], "Filter by source_type"
)
dbutils.widgets.text("documents_table_name", "weather_documents", "Documents table")
dbutils.widgets.text(
    "chunk_embeddings_table_name", "weather_chunk_embeddings", "Chunk embeddings table"
)
dbutils.widgets.text(
    "embedding_model", "sentence-transformers/all-MiniLM-L6-v2", "Embedding model (MUST match ingestion)"
)

QUERY = dbutils.widgets.get("query")
TOP_K = max(1, min(int(dbutils.widgets.get("top_k")), 20))
_source_type = dbutils.widgets.get("source_type")
SOURCE_TYPE = None if _source_type == "all" else _source_type
DOCUMENTS_TABLE_NAME = dbutils.widgets.get("documents_table_name")
CHUNK_EMBEDDINGS_TABLE_NAME = dbutils.widgets.get("chunk_embeddings_table_name")
EMBEDDING_MODEL_NAME = dbutils.widgets.get("embedding_model")

print(f"Query      : {QUERY!r}")
print(f"top_k      : {TOP_K}")
print(f"source_type: {SOURCE_TYPE or 'all'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resolve the Lakebase connection URL
# MAGIC
# MAGIC Same secret / decoding scheme as `lakebase.py` and the ingest notebook: a
# MAGIC single base64-encoded Postgres URL stored in a Databricks secret.

# COMMAND ----------

import base64
from urllib.parse import urlparse

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()


def get_lakebase_url() -> str:
    secret = w.secrets.get_secret(scope="database", key="lakebase-url")
    return base64.b64decode(secret.value).decode("utf-8")


parsed = urlparse(get_lakebase_url())
db_host = parsed.hostname
db_port = parsed.port or 5432
db_name = parsed.path.lstrip("/")
db_user = parsed.username
db_password = parsed.password

print(f"Connecting to {db_host}:{db_port}/{db_name} as {db_user}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the embedding model (same as ingestion)
# MAGIC
# MAGIC Loaded once here and reused for every search below - mirrors the Flask
# MAGIC endpoint's load-once module-level singleton.

# COMMAND ----------

import os

from sentence_transformers import SentenceTransformer

os.environ["HF_HOME"] = "/tmp/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/tmp/.cache/huggingface"

print(f"Loading embedding model {EMBEDDING_MODEL_NAME}...")
model = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder="/tmp/.cache/huggingface")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run the search
# MAGIC
# MAGIC `weather_search()` is a faithful copy of `POST /weather/search`: it embeds
# MAGIC the query, formats it as a pgvector literal, and runs the cosine `<=>`
# MAGIC search. The vector literal appears twice (similarity in SELECT + ORDER BY),
# MAGIC with the optional `source_type` filter between them - no extra JOIN needed
# MAGIC because `source_type` is denormalized onto the chunk table.

# COMMAND ----------

import pandas as pd
import psycopg2


def weather_search(query: str, top_k: int = 5, source_type: str | None = None) -> pd.DataFrame:
    vec = model.encode(query).tolist()
    vec_literal = "[" + ",".join(str(float(x)) for x in vec) + "]"

    where = "WHERE e.source_type = %s" if source_type else ""
    sql = f"""
        SELECT d.id, d.location, d.headline, e.chunk_text, e.source_type,
               1 - (e.embedding <=> %s::vector) AS similarity
        FROM {CHUNK_EMBEDDINGS_TABLE_NAME} e
        JOIN {DOCUMENTS_TABLE_NAME} d ON d.id = e.document_id
        {where}
        ORDER BY e.embedding <=> %s::vector
        LIMIT %s
    """
    params = [vec_literal]
    if source_type:
        params.append(source_type)
    params.extend([vec_literal, top_k])

    conn = psycopg2.connect(
        host=db_host, port=db_port, dbname=db_name,
        user=db_user, password=db_password, sslmode="require",
    )
    try:
        return pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()


results = weather_search(QUERY, top_k=TOP_K, source_type=SOURCE_TYPE)
print(
    f"Top {len(results)} matches for {QUERY!r}"
    + (f" (source_type={SOURCE_TYPE})" if SOURCE_TYPE else " (all sources)")
)
display(results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Done
# MAGIC
# MAGIC Change the **query** / **source_type** / **top_k** widgets and re-run the
# MAGIC search cell to explore. `similarity` is `1 - cosine_distance` (closer to 1
# MAGIC is a better match); with a `source_type` filter every row's `source_type`
# MAGIC column should equal the filter you chose.
