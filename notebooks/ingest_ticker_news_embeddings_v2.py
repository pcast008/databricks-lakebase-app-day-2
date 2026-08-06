# Databricks notebook source
# MAGIC %md
# MAGIC # Ingest Ticker News -> Vector Embeddings (Lakebase) - v2
# MAGIC
# MAGIC This notebook is part of the **Context Engineering on Databricks** course.
# MAGIC
# MAGIC **What's different in v2** (vs `ingest_ticker_news_embeddings.py`):
# MAGIC 1. **Writes via `pg8000`** (a pure-Python Postgres driver) instead of
# MAGIC    `psycopg2`. psycopg2's native C extension bundles its own OpenSSL, which
# MAGIC    collides with grpc's/the Databricks SDK's OpenSSL in a background
# MAGIC    credential-refresh thread and aborts the Python kernel with a SIGABRT
# MAGIC    (the "Python kernel is unresponsive" crash). pg8000 has no native code,
# MAGIC    so there is nothing to collide - it is stable on serverless AND classic.
# MAGIC 2. **Embeddings are written straight to `::vector`** (pgvector literal
# MAGIC    `[v1,v2,...]`) instead of a `double precision[]` array, so the manual
# MAGIC    `UPDATE ... SET embedding = embedding::vector` post-step is no longer
# MAGIC    needed.
# MAGIC 3. **Compute-once**: the embedding model and the per-article HTTP fetches
# MAGIC    each run a single time (results are `collect()`ed once and reused),
# MAGIC    instead of being silently re-run by a later `.count()`/`.collect()`.
# MAGIC 4. A browser-ish **User-Agent** on article fetches to dodge publisher
# MAGIC    bot-walls.
# MAGIC
# MAGIC It:
# MAGIC 1. Reads the `watchlist` table in Lakebase to find out which ticker
# MAGIC    symbols are currently being tracked.
# MAGIC 2. Fetches recent news for those tickers directly from the Massive
# MAGIC    `/v2/reference/news` endpoint (see `massive_client.py` for the same
# MAGIC    call shape used by the Flask app's `POST /news/sync` route), rate
# MAGIC    limited to stay within the free Massive API tier's strict quota, and
# MAGIC    upserts the results into the `ticker_news_documents` table.
# MAGIC 3. Computes a sentence embedding for each article (title + description)
# MAGIC    using Spark, distributed across the cluster via a pandas UDF, and
# MAGIC    writes them into a `ticker_news_embeddings` table using the
# MAGIC    `pgvector` Postgres extension so downstream RAG / context-engineering
# MAGIC    exercises can run similarity search directly in Postgres.
# MAGIC 4. Fetches the full article body for each `article_url` (via
# MAGIC    `trafilatura`, which strips nav/ads/boilerplate from the raw HTML),
# MAGIC    splits it into overlapping text chunks, embeds each chunk, and writes
# MAGIC    them into a `ticker_news_chunk_embeddings` table - so RAG exercises can
# MAGIC    retrieve fine-grained passages from article bodies, not just
# MAGIC    title/description.
# MAGIC
# MAGIC It re-uses the SAME Lakebase secret (scope `database`, key `lakebase-url`)
# MAGIC that `lakebase.py` uses in the Flask app, so no extra secrets need to be
# MAGIC created for this notebook.

# COMMAND ----------

# DBTITLE 1,Install all required packages
# MAGIC %pip install -q sentence-transformers trafilatura requests
# MAGIC %pip install -q pg8000

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config
# MAGIC
# MAGIC Widgets let you override the source/destination table names and the
# MAGIC embedding model without editing the notebook - useful when running this
# MAGIC as a scheduled Databricks Job.

# COMMAND ----------

dbutils.widgets.text("watchlist_table_name", "watchlist", "Source table (watchlist symbols)")
dbutils.widgets.text("news_table_name", "ticker_news_documents", "Destination table (raw news)")
dbutils.widgets.text("embeddings_table_name", "ticker_news_embeddings", "Destination table (vectors)")
dbutils.widgets.text("chunk_embeddings_table_name", "ticker_news_chunk_embeddings", "Destination table (chunk vectors)")
dbutils.widgets.text("embedding_model", "sentence-transformers/all-MiniLM-L6-v2", "Embedding model")
dbutils.widgets.text("massive_secret_scope", "massive", "Massive API secret scope")
dbutils.widgets.text("massive_secret_key", "api-key", "Massive API secret key")
dbutils.widgets.text("massive_api_base_url", "https://api.massive.com", "Massive API base URL")
dbutils.widgets.text("news_fetch_limit", "50", "Max articles to fetch per ticker")
dbutils.widgets.text("max_requests_per_minute", "5", "Massive API rate limit (free tier is strict)")
dbutils.widgets.text("chunk_size", "800", "Article content chunk size (chars)")
dbutils.widgets.text("chunk_overlap", "100", "Article content chunk overlap (chars)")

WATCHLIST_TABLE_NAME = dbutils.widgets.get("watchlist_table_name")
NEWS_TABLE_NAME = dbutils.widgets.get("news_table_name")
EMBEDDINGS_TABLE_NAME = dbutils.widgets.get("embeddings_table_name")
CHUNK_EMBEDDINGS_TABLE_NAME = dbutils.widgets.get("chunk_embeddings_table_name")
EMBEDDING_MODEL_NAME = dbutils.widgets.get("embedding_model")
MASSIVE_SECRET_SCOPE = dbutils.widgets.get("massive_secret_scope")
MASSIVE_SECRET_KEY = dbutils.widgets.get("massive_secret_key")
MASSIVE_API_BASE_URL = dbutils.widgets.get("massive_api_base_url")
NEWS_FETCH_LIMIT = int(dbutils.widgets.get("news_fetch_limit"))
MAX_REQUESTS_PER_MINUTE = int(dbutils.widgets.get("max_requests_per_minute"))
CHUNK_SIZE = int(dbutils.widgets.get("chunk_size"))
CHUNK_OVERLAP = int(dbutils.widgets.get("chunk_overlap"))

# Different sentence-transformers models emit different vector sizes, and the
# pgvector column type (VECTOR(N)) must match exactly. Rather than hardcoding
# one dimension, switch on the model name so swapping EMBEDDING_MODEL_NAME via
# the widget above automatically resizes the destination table's vector column.
match EMBEDDING_MODEL_NAME:
    case "sentence-transformers/all-MiniLM-L6-v2":
        EMBEDDING_DIM = 384
    case "sentence-transformers/all-MiniLM-L12-v2":
        EMBEDDING_DIM = 384
    case "sentence-transformers/all-mpnet-base-v2":
        EMBEDDING_DIM = 768
    case "sentence-transformers/paraphrase-multilingual-mpnet-base-v2":
        EMBEDDING_DIM = 768
    case "BAAI/bge-small-en-v1.5":
        EMBEDDING_DIM = 384
    case "BAAI/bge-base-en-v1.5":
        EMBEDDING_DIM = 768
    case "BAAI/bge-large-en-v1.5":
        EMBEDDING_DIM = 1024
    case "text-embedding-3-small":
        EMBEDDING_DIM = 1536
    case "text-embedding-3-large":
        EMBEDDING_DIM = 3072
    case _:
        raise ValueError(
            f"Unknown embedding model {EMBEDDING_MODEL_NAME!r} - add its output "
            "dimension to the match/case block above before running this notebook."
        )

print(f"Using model {EMBEDDING_MODEL_NAME!r} -> {EMBEDDING_DIM}-dim vectors")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resolve the Lakebase connection URL
# MAGIC
# MAGIC Same secret, same decoding scheme as `lakebase.py`: a single base64-encoded
# MAGIC Postgres URL (`postgresql://role:password@host:5432/db?sslmode=require`)
# MAGIC stored in a Databricks secret scope. We parse it into the pieces both
# MAGIC Spark's JDBC reader AND the pg8000 writer below need (host/port/db/user/password).

# COMMAND ----------

# DBTITLE 1,Parse Lakebase Connection Info
import base64
import re
from urllib.parse import urlparse, quote_plus

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()


def get_lakebase_url() -> str:
    secret = w.secrets.get_secret(scope="database", key="lakebase-url")
    return base64.b64decode(secret.value).decode("utf-8")


lakebase_url = get_lakebase_url()
parsed = urlparse(lakebase_url)

# Extract project name and branch name from hostname
# Format: ep-{branch-name}-{random}.{project-name}.{region}.cloud.databricks.com
hostname_parts = parsed.hostname.split('.')
if len(hostname_parts) >= 2:
    # Extract project name (second part)
    project_name = hostname_parts[1]
    # Extract branch name from first part (ep-{branch-name}-{random})
    branch_match = re.match(r'ep-([^-]+)', hostname_parts[0])
    branch_name = branch_match.group(1) if branch_match else 'production'
else:
    raise ValueError(f"Unexpected Lakebase hostname format: {parsed.hostname}")

# Build JDBC URL for reading only (writes use pg8000, below)
jdbc_url = f"jdbc:postgresql://{parsed.hostname}:{parsed.port or 5432}{parsed.path}"
print(f"Connecting to: {parsed.hostname}:{parsed.port or 5432}{parsed.path}")
print(f"Project: {project_name}, Branch: {branch_name}")

# Pass credentials and SSL settings in properties for JDBC reads
jdbc_properties = {
    "user": parsed.username,
    "password": parsed.password,
    "driver": "org.postgresql.Driver",
    "sslmode": "require",
}

db_host = parsed.hostname
db_name = parsed.path.lstrip('/')
print(f"Database: {db_name}")

# COMMAND ----------

# DBTITLE 1,Test JDBC Connection
# Test JDBC connection with embedded credentials
try:
    test_df = spark.read.jdbc(
        url=jdbc_url,
        table=WATCHLIST_TABLE_NAME,
        properties=jdbc_properties
    )
    count = test_df.count()
    print(f"✅ Connection successful! Found {count} rows in {WATCHLIST_TABLE_NAME}")
    test_df.show(5)
except Exception as e:
    print(f"❌ Connection failed: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Database Setup Instructions
# MAGIC
# MAGIC Before running this notebook, you must manually create the required tables
# MAGIC in your Lakebase Postgres database:
# MAGIC
# MAGIC 1. Run `sql/01_setup_news_table.sql` to create `ticker_news_documents`
# MAGIC 2. Run `sql/02_setup_embeddings_table.sql` to create `ticker_news_embeddings`
# MAGIC    - Replace `{{EMBEDDING_DIM}}` with your model's dimension (e.g., 384)
# MAGIC 3. Run `sql/03_setup_chunk_embeddings_table.sql` to create `ticker_news_chunk_embeddings`
# MAGIC    - Replace `{{EMBEDDING_DIM}}` with your model's dimension (e.g., 384)
# MAGIC
# MAGIC This notebook **reads** via Spark JDBC and **writes** via `pg8000` (a
# MAGIC pure-Python Postgres driver) - no psycopg2, so no native-OpenSSL SIGABRT
# MAGIC crash on Databricks compute.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fetch news from Massive for watchlisted tickers
# MAGIC
# MAGIC This ETL is now self-contained: instead of relying on the Flask app's
# MAGIC `POST /news/sync` route to have populated `ticker_news_documents` ahead of
# MAGIC time, the notebook queries the `watchlist` table in Lakebase directly to
# MAGIC find out which tickers are being tracked, then pulls news for exactly
# MAGIC those tickers from Massive itself.
# MAGIC
# MAGIC The free Massive API tier is rate-limited very aggressively, so requests
# MAGIC are made **serially** (not distributed across Spark workers) with a sleep
# MAGIC between calls that enforces `MAX_REQUESTS_PER_MINUTE` (default 5/min).

# COMMAND ----------

# DBTITLE 1,Fetch news and collect rows to upsert
import base64 as _b64
import json as _json
import time
from datetime import datetime

import requests
from pyspark.sql.functions import col, current_timestamp, lit
from pyspark.sql.types import StringType, StructField, StructType


def get_massive_api_key() -> str:
    secret = w.secrets.get_secret(scope=MASSIVE_SECRET_SCOPE, key=MASSIVE_SECRET_KEY)
    return _b64.b64decode(secret.value).decode("utf-8")


def get_watchlist_tickers() -> list[str]:
    """Distinct, uppercased ticker symbols currently tracked across all users
    in the watchlist table - these are the only tickers we fetch news for."""
    watchlist_df = spark.read.jdbc(
        url=jdbc_url, table=WATCHLIST_TABLE_NAME, properties=jdbc_properties
    )
    symbols = watchlist_df.select("symbol").distinct().collect()
    return [row.symbol.strip().upper() for row in symbols if row.symbol]


def fetch_news_for_ticker(session: requests.Session, ticker: str, limit: int) -> list[dict]:
    """Single GET /v2/reference/news call for one ticker (mirrors
    MassiveClient.get_news in massive_client.py)."""
    resp = session.get(
        f"{MASSIVE_API_BASE_URL}/v2/reference/news",
        params={"ticker": ticker, "limit": limit, "order": "desc", "sort": "published_utc"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("results", [])


def build_news_rows(ticker: str, articles: list[dict]) -> list[dict]:
    """Convert a Massive API response into row dicts ready for insert."""
    if not articles:
        return []

    rows = []
    for article in articles:
        sentiment = None
        sentiment_reasoning = None
        for insight in article.get("insights", []) or []:
            if insight.get("ticker") == ticker:
                sentiment = insight.get("sentiment")
                sentiment_reasoning = insight.get("sentiment_reasoning")
                break

        publisher = article.get("publisher") or {}
        rows.append({
            "id": str(article.get("id")),
            "ticker": ticker,
            "title": article.get("title", ""),
            "description": article.get("description"),
            "author": article.get("author"),
            "article_url": article.get("article_url"),
            "publisher_name": publisher.get("name"),
            "keywords": _json.dumps(article.get("keywords", [])),
            "sentiment": sentiment,
            "sentiment_reasoning": sentiment_reasoning,
            "published_utc": article.get("published_utc"),
            "payload": _json.dumps(article),
        })
    return rows


print("NOTE: Before running this cell, ensure you've run sql/01_setup_news_table.sql")
print("      to create the ticker_news_documents table in your Lakebase database.\n")

tickers = get_watchlist_tickers()
print(f"Found {len(tickers)} distinct watchlisted tickers: {tickers}")

# Enforce MAX_REQUESTS_PER_MINUTE by spacing calls evenly across a minute -
# e.g. 5/min -> one request every 12s. Sleeping BEFORE each call after the
# first keeps this correct even if a single request itself takes a while.
_seconds_between_requests = 60.0 / MAX_REQUESTS_PER_MINUTE

_massive_session = requests.Session()
_massive_session.headers.update(
    {"Authorization": f"Bearer {get_massive_api_key()}", "Content-Type": "application/json"}
)

all_news_rows = []  # Collect all rows, then insert via pg8000 in the next cell
for i, ticker in enumerate(tickers):
    if i > 0:
        time.sleep(_seconds_between_requests)
    try:
        articles = fetch_news_for_ticker(_massive_session, ticker, NEWS_FETCH_LIMIT)
        batch_rows = build_news_rows(ticker, articles)
        if batch_rows:
            all_news_rows.extend(batch_rows)
    except Exception as exc:
        print(f"Skipping {ticker}: failed to fetch/sync news ({exc})")
        continue

print(f"\nCollected {len(all_news_rows)} news articles to insert. Run the next cell to write them.")

# COMMAND ----------

# DBTITLE 1,Insert collected news articles using pg8000
# pg8000 (pure Python, no bundled libssl) instead of psycopg2 - avoids the
# SIGABRT crash from psycopg2's bundled OpenSSL colliding with grpc's OpenSSL
# in the Databricks SDK's background credential-refresh thread.
import pg8000.dbapi as dbapi

if all_news_rows:
    print(f"Inserting {len(all_news_rows)} news articles into {NEWS_TABLE_NAME}...")

    conn = dbapi.connect(
        host=db_host,
        port=parsed.port or 5432,
        database=db_name,
        user=parsed.username,
        password=parsed.password,
        ssl_context=True,
    )
    try:
        cur = conn.cursor()
        try:
            insert_data = [
                (
                    row['id'],
                    row['ticker'],
                    row['title'],
                    row['description'],
                    row['author'],
                    row['article_url'],
                    row['publisher_name'],
                    row['keywords'],
                    row['sentiment'],
                    row['sentiment_reasoning'],
                    row['published_utc'],
                    row['payload'],
                )
                for row in all_news_rows
            ]

            # NOW() fills synced_at; ON CONFLICT DO NOTHING dedupes on id.
            insert_sql = f"""
                INSERT INTO {NEWS_TABLE_NAME} (
                    id, ticker, title, description, author, article_url, publisher_name,
                    keywords, sentiment, sentiment_reasoning, published_utc, payload, synced_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (id) DO NOTHING
            """
            cur.executemany(insert_sql, insert_data)
            conn.commit()
            print(f"✅ Inserted up to {len(insert_data)} news articles "
                  f"(duplicates skipped via ON CONFLICT DO NOTHING)")
        finally:
            cur.close()
    finally:
        conn.close()
else:
    print("No news articles to write.")

print(f"\nReady to compute embeddings! Run the cells below to continue.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load raw news documents with Spark
# MAGIC
# MAGIC Reads the whole `ticker_news_documents` table (just synced from Massive
# MAGIC above) via JDBC into a Spark DataFrame so embedding computation can be
# MAGIC distributed across the cluster.

# COMMAND ----------

news_df = (
    spark.read.jdbc(url=jdbc_url, table=NEWS_TABLE_NAME, properties=jdbc_properties)
    .selectExpr(
        "id",
        "ticker",
        "title",
        "description",
        "article_url",
        "published_utc",
        # Embed on title + description together for richer context.
        "trim(concat(coalesce(title, ''), '. ', coalesce(description, ''))) AS embedding_text",
    )
    .filter("embedding_text IS NOT NULL AND embedding_text != ''")
)

print(f"Loaded {news_df.count()} news documents from {NEWS_TABLE_NAME}")
display(news_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute embeddings (distributed pandas UDF)
# MAGIC
# MAGIC Loads the sentence-transformers model once per executor process (not per
# MAGIC row) and applies it in batches via `mapInPandas`, which scales across
# MAGIC however many workers the cluster has. We `collect()` the result **once**
# MAGIC here and reuse it downstream so the model isn't re-run by a later action.

# COMMAND ----------

# DBTITLE 1,Compute embeddings (distributed pandas UDF)
from typing import Iterator

import pandas as pd
from pyspark.sql.types import ArrayType, FloatType, IntegerType, StringType, StructField, StructType

embeddings_schema = StructType(
    [
        StructField("id", StringType(), False),
        StructField("ticker", StringType(), False),
        StructField("title", StringType(), False),
        StructField("published_utc", StringType(), True),
        StructField("embedding", ArrayType(FloatType()), False),
    ]
)


def embed_partitions(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    """Runs once per Spark partition/task: load the model once, then embed
    every batch of rows handed to this partition."""
    import os
    from sentence_transformers import SentenceTransformer

    os.environ["HF_HOME"] = "/tmp/.cache/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = "/tmp/.cache/huggingface"
    os.environ["HF_HUB_CACHE"] = "/tmp/.cache/huggingface"
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder="/tmp/.cache/huggingface")

    for batch in iterator:
        vectors = model.encode(batch["embedding_text"].tolist(), show_progress_bar=False)
        yield pd.DataFrame(
            {
                "id": batch["id"],
                "ticker": batch["ticker"],
                "title": batch["title"],
                "published_utc": batch["published_utc"].astype(str),
                "embedding": [v.tolist() for v in vectors],
            }
        )


# repartition(2) collapses the (few hundred) rows into 2 partitions so the model
# is loaded at most twice, not once per default partition.
embeddings_df = news_df.repartition(2).mapInPandas(embed_partitions, schema=embeddings_schema)

# Collect ONCE here and reuse embeddings_rows downstream. A second action on
# embeddings_df (e.g. calling .count() here then .collect() later) would silently
# re-run embed_partitions and reload/re-encode the whole model a second time.
embeddings_rows = embeddings_df.collect()
print(f"Computed {len(embeddings_rows)} embeddings using {EMBEDDING_MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ensure the pgvector destination table exists
# MAGIC
# MAGIC Each embedding is formatted as a pgvector literal (`[v1,v2,...]`) and cast
# MAGIC to `::vector` on insert via pg8000, so the vectors land as real `vector(N)`
# MAGIC values immediately - no manual post-processing `UPDATE` step is needed.

# COMMAND ----------

# Before running the cells below, ensure you've manually run:
#   sql/02_setup_embeddings_table.sql
# Replace {{EMBEDDING_DIM}} in that file with the value below:
print(f"Required EMBEDDING_DIM for SQL setup: {EMBEDDING_DIM}")
print(f"Table name: {EMBEDDINGS_TABLE_NAME}")
print("\nRun sql/02_setup_embeddings_table.sql in your Lakebase database before continuing.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Upsert embeddings into Lakebase
# MAGIC
# MAGIC Written via pg8000 `executemany`. Each embedding is passed as a pgvector
# MAGIC literal string and cast to Postgres' `vector` type via `::vector`.

# COMMAND ----------

# DBTITLE 1,Insert embeddings using pg8000
import pg8000.dbapi as dbapi

# embeddings_rows was already collected once above - reuse it here rather than
# calling .collect() again (which would re-run the embedding model).
if len(embeddings_rows) > 0:
    print(f"Inserting {len(embeddings_rows)} embeddings into {EMBEDDINGS_TABLE_NAME}...")

    conn = dbapi.connect(
        host=db_host,
        port=parsed.port or 5432,
        database=db_name,
        user=parsed.username,
        password=parsed.password,
        ssl_context=True,
    )
    try:
        cur = conn.cursor()
        try:
            # Format the embedding as a pgvector literal "[v1,v2,...]" and cast
            # it straight to ::vector on insert - no ::double precision[] array
            # and no manual post-processing UPDATE step needed.
            insert_data = [
                (
                    row.id,
                    row.ticker,
                    row.title,
                    str(row.published_utc) if row.published_utc else None,
                    "[" + ",".join(str(float(x)) for x in row.embedding) + "]",
                    EMBEDDING_MODEL_NAME,
                )
                for row in embeddings_rows
            ]

            insert_sql = f"""
                INSERT INTO {EMBEDDINGS_TABLE_NAME} (
                    id, ticker, title, published_utc, embedding, model_name, embedded_at
                ) VALUES (%s, %s, %s, %s, %s::vector, %s, NOW())
                ON CONFLICT (id) DO NOTHING
            """
            cur.executemany(insert_sql, insert_data)
            conn.commit()
            print(f"✅ Inserted up to {len(insert_data)} embeddings "
                  f"(duplicates skipped via ON CONFLICT DO NOTHING)")
        finally:
            cur.close()
    finally:
        conn.close()
else:
    print("No embeddings to write.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fetch and chunk article content
# MAGIC
# MAGIC Title/description only gets you so far - the actual article body lives at
# MAGIC `article_url` on the publisher's site. This step fetches each URL, uses
# MAGIC `trafilatura` to extract just the article text (stripping nav/ads/related
# MAGIC links/etc.), and splits it into overlapping chunks so each chunk can be
# MAGIC embedded and retrieved independently. Fetching is distributed across the
# MAGIC cluster via `mapInPandas`; any URL that fails to fetch/extract (paywall,
# MAGIC timeout, dead link) is skipped rather than failing the whole job. We
# MAGIC `collect()` the chunks **once** (fetching is the expensive part) and
# MAGIC rebuild a DataFrame from them for the embedding step.

# COMMAND ----------

content_df = news_df.select("id", "ticker", "article_url").filter(
    "article_url IS NOT NULL AND article_url != ''"
)

chunks_schema = StructType(
    [
        StructField("article_id", StringType(), False),
        StructField("ticker", StringType(), False),
        StructField("chunk_index", IntegerType(), False),
        StructField("chunk_text", StringType(), False),
    ]
)


def fetch_and_chunk_partitions(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    """Runs once per Spark partition/task: fetch each article's HTML, extract
    the main body text with trafilatura, then split it into overlapping
    chunks of CHUNK_SIZE characters (CHUNK_OVERLAP characters shared between
    consecutive chunks so context isn't lost at chunk boundaries)."""
    import requests
    import trafilatura

    # A browser-ish User-Agent avoids a chunk of publisher 403s/bot-walls that
    # would otherwise drop those articles' bodies entirely.
    headers = {"User-Agent": "Mozilla/5.0 (compatible; FinScoutBot/1.0)"}

    for batch in iterator:
        out_article_ids, out_tickers, out_chunk_indexes, out_chunk_texts = [], [], [], []
        for article_id, ticker, article_url in zip(
            batch["id"], batch["ticker"], batch["article_url"]
        ):
            try:
                resp = requests.get(article_url, timeout=15, headers=headers)
                resp.raise_for_status()
                text = trafilatura.extract(resp.text)
            except Exception:
                # Dead link, paywall, timeout, etc. - skip this article's
                # content chunks rather than failing the whole job.
                continue

            if not text:
                continue

            for chunk_index, start in enumerate(range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP)):
                chunk_text = text[start : start + CHUNK_SIZE].strip()
                if not chunk_text:
                    continue
                out_article_ids.append(article_id)
                out_tickers.append(ticker)
                out_chunk_indexes.append(chunk_index)
                out_chunk_texts.append(chunk_text)
                if start + CHUNK_SIZE >= len(text):
                    break

        yield pd.DataFrame(
            {
                "article_id": out_article_ids,
                "ticker": out_tickers,
                "chunk_index": out_chunk_indexes,
                "chunk_text": out_chunk_texts,
            }
        )


chunks_df = content_df.mapInPandas(fetch_and_chunk_partitions, schema=chunks_schema)

# Collect ONCE - fetching is the expensive part (an HTTP GET per article URL).
# Any later action on chunks_df would silently re-run every fetch again, so we
# materialize the chunks here and rebuild a DataFrame from them downstream.
chunks_rows = chunks_df.collect()
print(f"Extracted {len(chunks_rows)} content chunks from article URLs")
if chunks_rows:
    display(pd.DataFrame([row.asDict() for row in chunks_rows[:5]]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute chunk embeddings
# MAGIC
# MAGIC Same approach as the title/description embeddings above, but one vector
# MAGIC per content chunk instead of per article. We rebuild the DataFrame from
# MAGIC the already-collected `chunks_rows` so we don't re-fetch every URL.

# COMMAND ----------

chunk_embeddings_schema = StructType(
    [
        StructField("article_id", StringType(), False),
        StructField("ticker", StringType(), False),
        StructField("chunk_index", IntegerType(), False),
        StructField("chunk_text", StringType(), False),
        StructField("embedding", ArrayType(FloatType()), False),
    ]
)


def embed_chunk_partitions(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    """Runs once per Spark partition: load the model once, then embed
    every batch of chunks handed to this partition."""
    import os
    from sentence_transformers import SentenceTransformer

    os.environ["HF_HOME"] = "/tmp/.cache/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = "/tmp/.cache/huggingface"
    os.environ["HF_HUB_CACHE"] = "/tmp/.cache/huggingface"
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder="/tmp/.cache/huggingface")

    for batch in iterator:
        vectors = model.encode(batch["chunk_text"].tolist(), show_progress_bar=False)
        yield pd.DataFrame(
            {
                "article_id": batch["article_id"],
                "ticker": batch["ticker"],
                "chunk_index": batch["chunk_index"],
                "chunk_text": batch["chunk_text"],
                "embedding": [v.tolist() for v in vectors],
            }
        )


# Rebuild from chunks_rows (already collected), NOT chunks_df - chunks_df was
# never persisted, so an action on it here would silently re-run every HTTP
# fetch again. repartition(2) keeps the model from reloading per default partition.
if chunks_rows:
    chunks_spark_df = spark.createDataFrame(chunks_rows, schema=chunks_schema)
    chunk_embeddings_df = chunks_spark_df.repartition(2).mapInPandas(
        embed_chunk_partitions, schema=chunk_embeddings_schema
    )
    chunk_embeddings_rows = chunk_embeddings_df.collect()
else:
    chunk_embeddings_rows = []

print(f"Computed {len(chunk_embeddings_rows)} chunk embeddings using {EMBEDDING_MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ensure the chunk embeddings destination table exists

# COMMAND ----------

# Before running the cells below, ensure you've manually run:
#   sql/03_setup_chunk_embeddings_table.sql
# Replace {{EMBEDDING_DIM}} in that file with the value below:
print(f"Required EMBEDDING_DIM for SQL setup: {EMBEDDING_DIM}")
print(f"Table name: {CHUNK_EMBEDDINGS_TABLE_NAME}")
print("\nRun sql/03_setup_chunk_embeddings_table.sql in your Lakebase database before continuing.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Upsert chunk embeddings into Lakebase

# COMMAND ----------

# DBTITLE 1,Insert chunk embeddings using pg8000
import pg8000.dbapi as dbapi

# chunk_embeddings_rows was already collected once above - reuse it here.
if len(chunk_embeddings_rows) > 0:
    print(f"Inserting {len(chunk_embeddings_rows)} chunk embeddings into {CHUNK_EMBEDDINGS_TABLE_NAME}...")

    conn = dbapi.connect(
        host=db_host,
        port=parsed.port or 5432,
        database=db_name,
        user=parsed.username,
        password=parsed.password,
        ssl_context=True,
    )
    try:
        cur = conn.cursor()
        try:
            # id = "{article_id}_{chunk_index}"; embedding cast straight to ::vector.
            insert_data = [
                (
                    f"{row.article_id}_{row.chunk_index}",
                    row.article_id,
                    row.ticker,
                    int(row.chunk_index),
                    row.chunk_text,
                    "[" + ",".join(str(float(x)) for x in row.embedding) + "]",
                    EMBEDDING_MODEL_NAME,
                )
                for row in chunk_embeddings_rows
            ]

            insert_sql = f"""
                INSERT INTO {CHUNK_EMBEDDINGS_TABLE_NAME} (
                    id, article_id, ticker, chunk_index, chunk_text, embedding, model_name, embedded_at
                ) VALUES (%s, %s, %s, %s, %s, %s::vector, %s, NOW())
                ON CONFLICT (id) DO NOTHING
            """
            cur.executemany(insert_sql, insert_data)
            conn.commit()
            print(f"✅ Inserted up to {len(insert_data)} chunk embeddings "
                  f"(duplicates skipped via ON CONFLICT DO NOTHING)")
        finally:
            cur.close()
    finally:
        conn.close()
else:
    print("No chunk embeddings to write.")
