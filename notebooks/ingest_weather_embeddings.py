# Databricks notebook source
# MAGIC %md
# MAGIC # Ingest Weather Documents -> Vector Embeddings (Lakebase)
# MAGIC
# MAGIC This notebook is the weather counterpart of
# MAGIC `ingest_ticker_news_embeddings.py`. It is **embeddings-only**: it assumes
# MAGIC `weather_documents` has already been populated (run
# MAGIC `notebooks/seed_weather.py` first, which wraps `weather_sync.sync_weather()`
# MAGIC and the same code path as `POST /weather/sync`).
# MAGIC
# MAGIC It:
# MAGIC 1. Reads the `weather_documents` table in Lakebase (raw NWS alert +
# MAGIC    forecast narratives harvested from api.weather.gov).
# MAGIC 2. Computes a sentence embedding for each document's whole narrative
# MAGIC    (`headline` + `narrative_text`) and writes them into a
# MAGIC    `weather_embeddings` table (one vector per document) using the
# MAGIC    `pgvector` Postgres extension.
# MAGIC 3. Splits each `narrative_text` into overlapping sliding-window chunks,
# MAGIC    embeds each chunk, and writes them into a `weather_chunk_embeddings`
# MAGIC    table (many vectors per document) - so `POST /weather/search` can do
# MAGIC    fine-grained semantic retrieval, not just whole-document matches.
# MAGIC
# MAGIC Unlike the news notebook there is **no URL fetch / trafilatura leg**: the
# MAGIC weather `narrative_text` IS the document body already, so we chunk the text
# MAGIC we have in hand.
# MAGIC
# MAGIC It re-uses the SAME Lakebase secret (scope `database`, key `lakebase-url`)
# MAGIC that `lakebase.py` uses in the Flask app, so no extra secrets need to be
# MAGIC created for this notebook.

# COMMAND ----------

# DBTITLE 1,Install all required packages
# MAGIC %pip uninstall -y psycopg2 psycopg2-binary
# MAGIC %pip install -q 'databricks-sdk>=0.118.0' sentence-transformers pandas

# COMMAND ----------

# MAGIC %md
# MAGIC The pip `psycopg2`/`psycopg2-binary` wheel bundles its own OpenSSL, which
# MAGIC collides with the OpenSSL that `databricks-sdk`/grpc uses in a background
# MAGIC credential-refresh thread and SIGABRTs the kernel ("The Python kernel is
# MAGIC unresponsive"). We uninstall it and restart Python so the runtime's system
# MAGIC psycopg2 is what loads (same ritual as the news notebook and seed_weather).

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

dbutils.widgets.text("documents_table_name", "weather_documents", "Source table (raw weather docs)")
dbutils.widgets.text("embeddings_table_name", "weather_embeddings", "Destination table (whole-doc vectors)")
dbutils.widgets.text("chunk_embeddings_table_name", "weather_chunk_embeddings", "Destination table (chunk vectors)")
dbutils.widgets.text("embedding_model", "sentence-transformers/all-MiniLM-L6-v2", "Embedding model")
dbutils.widgets.text("chunk_size", "800", "Narrative chunk size (chars)")
dbutils.widgets.text("chunk_overlap", "100", "Narrative chunk overlap (chars)")

DOCUMENTS_TABLE_NAME = dbutils.widgets.get("documents_table_name")
EMBEDDINGS_TABLE_NAME = dbutils.widgets.get("embeddings_table_name")
CHUNK_EMBEDDINGS_TABLE_NAME = dbutils.widgets.get("chunk_embeddings_table_name")
EMBEDDING_MODEL_NAME = dbutils.widgets.get("embedding_model")
CHUNK_SIZE = int(dbutils.widgets.get("chunk_size"))
CHUNK_OVERLAP = int(dbutils.widgets.get("chunk_overlap"))

# Different sentence-transformers models emit different vector sizes, and the
# pgvector column type (VECTOR(N)) must match exactly. Rather than hardcoding
# one dimension, switch on the model name so swapping EMBEDDING_MODEL_NAME via
# the widget above is caught here if it no longer matches the table's VECTOR(N).
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
print(
    f"NOTE: {EMBEDDINGS_TABLE_NAME}/{CHUNK_EMBEDDINGS_TABLE_NAME} declare "
    f"VECTOR(384); if EMBEDDING_DIM != 384 re-create them with the right size."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resolve the Lakebase connection URL
# MAGIC
# MAGIC Same secret, same decoding scheme as `lakebase.py`: a single base64-encoded
# MAGIC Postgres URL (`postgresql://role:password@host:5432/db?sslmode=require`)
# MAGIC stored in a Databricks secret scope. We parse it into the pieces psycopg2
# MAGIC needs for connection (host/port/dbname/user/password).

# COMMAND ----------

# DBTITLE 1,Parse Lakebase Connection Info
import base64
from urllib.parse import urlparse

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()


def get_lakebase_url() -> str:
    secret = w.secrets.get_secret(scope="database", key="lakebase-url")
    return base64.b64decode(secret.value).decode("utf-8")


lakebase_url = get_lakebase_url()
parsed = urlparse(lakebase_url)

# Extract connection details directly from the secret URL
db_host = parsed.hostname
db_port = parsed.port or 5432
db_name = parsed.path.lstrip('/')
db_user = parsed.username
db_password = parsed.password

print(f"Connection details:")
print(f"  Host: {db_host}:{db_port}")
print(f"  Database: {db_name}")
print(f"  User: {db_user}")
print(f"  Using raw credentials from secret (no OAuth)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prerequisites
# MAGIC
# MAGIC Before running the cells below, ensure:
# MAGIC
# MAGIC 1. `weather_documents` is populated - run `notebooks/seed_weather.py`
# MAGIC    (same code path as `POST /weather/sync`).
# MAGIC 2. `sql/weather/02_setup_weather_embeddings_table.sql` has been run
# MAGIC    (creates `weather_embeddings` with a `VECTOR(384)` column + HNSW index).
# MAGIC 3. `sql/weather/03_setup_weather_chunk_embeddings_table.sql` has been run
# MAGIC    (creates `weather_chunk_embeddings` likewise).
# MAGIC
# MAGIC This notebook uses psycopg2 with raw credentials for all DB operations.

# COMMAND ----------

# DBTITLE 1,Test psycopg2 connection
import psycopg2

print(f"Testing connection to {db_host}:{db_port}/{db_name} as {db_user}\n")

try:
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password,
        sslmode='require',
        connect_timeout=10,
    )
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {DOCUMENTS_TABLE_NAME}")
    count = cursor.fetchone()[0]
    print(f"✅ Connection successful! Found {count} rows in {DOCUMENTS_TABLE_NAME}")
    if count == 0:
        print(
            f"⚠️  {DOCUMENTS_TABLE_NAME} is empty - run notebooks/seed_weather.py "
            "first, or the embedding cells below will have nothing to embed."
        )
    cursor.close()
    conn.close()
except Exception as e:
    import traceback
    print(f"❌ Connection failed: {e}")
    traceback.print_exc()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load raw weather documents
# MAGIC
# MAGIC Reads `weather_documents` into a pandas DataFrame, computing the
# MAGIC `embedding_text` we embed for the whole-document vectors: the `headline`
# MAGIC (short label, e.g. "Flash Flood Warning" / "Tonight") followed by the full
# MAGIC `narrative_text`. This is weather's analog of the news title+description.

# COMMAND ----------

import pandas as pd
import psycopg2

conn = psycopg2.connect(
    host=db_host,
    port=db_port,
    dbname=db_name,
    user=db_user,
    password=db_password,
    sslmode='require',
)

try:
    query = f"""
        SELECT
            id,
            location,
            source_type,
            headline,
            issued_at,
            narrative_text,
            TRIM(CONCAT(COALESCE(headline, ''), '. ', COALESCE(narrative_text, ''))) AS embedding_text
        FROM {DOCUMENTS_TABLE_NAME}
        WHERE TRIM(CONCAT(COALESCE(headline, ''), '. ', COALESCE(narrative_text, ''))) IS NOT NULL
          AND TRIM(CONCAT(COALESCE(headline, ''), '. ', COALESCE(narrative_text, ''))) != '.'
          AND TRIM(COALESCE(narrative_text, '')) != ''
    """
    docs_df = pd.read_sql_query(query, conn)
    print(f"Loaded {len(docs_df)} weather documents from {DOCUMENTS_TABLE_NAME}")
    display(docs_df.head(5))
finally:
    conn.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute whole-document embeddings
# MAGIC
# MAGIC Loads the sentence-transformers model once and applies it in batches to
# MAGIC each document's `embedding_text`.

# COMMAND ----------

# DBTITLE 1,Compute whole-document embeddings
import os

import pandas as pd
from sentence_transformers import SentenceTransformer

# Set up HuggingFace cache
os.environ["HF_HOME"] = "/tmp/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/tmp/.cache/huggingface"

print(f"Loading embedding model {EMBEDDING_MODEL_NAME}...")
model = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder="/tmp/.cache/huggingface")

print("Computing whole-document embeddings...")
batch_size = 32
all_embeddings = []

for i in range(0, len(docs_df), batch_size):
    batch = docs_df.iloc[i:i + batch_size]
    vectors = model.encode(batch["embedding_text"].tolist(), show_progress_bar=False)
    all_embeddings.extend(vectors.tolist())
    if (i + batch_size) % 128 == 0:
        print(f"  Processed {min(i + batch_size, len(docs_df))}/{len(docs_df)} documents")

embeddings_df = pd.DataFrame({
    "id": docs_df["id"],
    "location": docs_df["location"],
    "source_type": docs_df["source_type"],
    "headline": docs_df["headline"],
    "issued_at": docs_df["issued_at"],
    "embedding": all_embeddings,
})

print(f"Computed {len(embeddings_df)} whole-document embeddings using {EMBEDDING_MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Upsert whole-document embeddings into Lakebase
# MAGIC
# MAGIC Written in batches via psycopg2's `execute_values`. The
# MAGIC `weather_embeddings.embedding` column is already `VECTOR(384)`, so we pass
# MAGIC each embedding as a pgvector literal (`'[v1,v2,...]'`) and cast with
# MAGIC `::vector` in the INSERT - no manual post-cast SQL step (unlike the news
# MAGIC notebook, whose table used a `double precision[]` staging column).

# COMMAND ----------

# DBTITLE 1,Insert whole-document embeddings
from datetime import datetime

import psycopg2
from psycopg2.extras import execute_values

embeddings_df['model_name'] = EMBEDDING_MODEL_NAME
embeddings_df['embedded_at'] = datetime.now()

embeddings_rows = embeddings_df.to_dict('records')

if len(embeddings_rows) > 0:
    print(f"Inserting {len(embeddings_rows)} embeddings into {EMBEDDINGS_TABLE_NAME}...")

    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password,
        sslmode='require',
    )

    try:
        cursor = conn.cursor()

        # Format embedding as a pgvector literal: '[v1,v2,...]'
        insert_data = [
            (
                row['id'],
                row['location'],
                row['source_type'],
                row['headline'],
                row['issued_at'],
                '[' + ','.join(str(float(x)) for x in row['embedding']) + ']',
                row['model_name'],
                row['embedded_at'],
            )
            for row in embeddings_rows
        ]

        insert_sql = f"""
            INSERT INTO {EMBEDDINGS_TABLE_NAME} (
                id, location, source_type, headline, issued_at, embedding, model_name, embedded_at
            ) VALUES %s
            ON CONFLICT (id) DO NOTHING
        """

        template = "(%s, %s, %s, %s, %s, %s::vector, %s, %s)"
        execute_values(cursor, insert_sql, insert_data, template=template, page_size=100)

        conn.commit()
        print(f"✅ Successfully inserted {cursor.rowcount} new embeddings")
        print(f"   (Duplicates skipped via ON CONFLICT (id) DO NOTHING)")
    finally:
        cursor.close()
        conn.close()
else:
    print("No whole-document embeddings to write.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chunk each narrative into overlapping windows
# MAGIC
# MAGIC Whole-document vectors are great for coarse matching, but long alert
# MAGIC narratives (description + instruction can run to several paragraphs) match
# MAGIC better when split into smaller passages. We slide a `CHUNK_SIZE`-char
# MAGIC window with `CHUNK_OVERLAP`-char overlap over each `narrative_text`. Short
# MAGIC forecast periods simply produce a single chunk.

# COMMAND ----------

import pandas as pd

print(f"Chunking {len(docs_df)} narratives (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")

out_document_ids, out_locations, out_source_types, out_chunk_indexes, out_chunk_texts = [], [], [], [], []

for _, row in docs_df.iterrows():
    document_id = row['id']
    location = row['location']
    source_type = row['source_type']
    text = (row['narrative_text'] or '').strip()
    if not text:
        continue

    for chunk_index, start in enumerate(range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP)):
        chunk_text = text[start:start + CHUNK_SIZE].strip()
        if not chunk_text:
            continue
        out_document_ids.append(document_id)
        out_locations.append(location)
        out_source_types.append(source_type)
        out_chunk_indexes.append(chunk_index)
        out_chunk_texts.append(chunk_text)
        if start + CHUNK_SIZE >= len(text):
            break

chunks_df = pd.DataFrame({
    "document_id": out_document_ids,
    "location": out_locations,
    "source_type": out_source_types,
    "chunk_index": out_chunk_indexes,
    "chunk_text": out_chunk_texts,
})

print(f"Produced {len(chunks_df)} chunks from {len(docs_df)} documents")
display(chunks_df.head(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute chunk embeddings
# MAGIC
# MAGIC Same approach as the whole-document embeddings - reuse the loaded model
# MAGIC and process in batches, but one vector per chunk.

# COMMAND ----------

import os

import pandas as pd
from sentence_transformers import SentenceTransformer

os.environ["HF_HOME"] = "/tmp/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/tmp/.cache/huggingface"

# Reuse the model if still loaded, otherwise load it.
if 'model' not in locals():
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder="/tmp/.cache/huggingface")

print(f"Computing chunk embeddings using {EMBEDDING_MODEL_NAME}...")
batch_size = 32
all_chunk_embeddings = []

for i in range(0, len(chunks_df), batch_size):
    batch = chunks_df.iloc[i:i + batch_size]
    vectors = model.encode(batch["chunk_text"].tolist(), show_progress_bar=False)
    all_chunk_embeddings.extend(vectors.tolist())
    if (i + batch_size) % 128 == 0:
        print(f"  Processed {min(i + batch_size, len(chunks_df))}/{len(chunks_df)} chunks")

chunk_embeddings_df = pd.DataFrame({
    "document_id": chunks_df["document_id"],
    "location": chunks_df["location"],
    "source_type": chunks_df["source_type"],
    "chunk_index": chunks_df["chunk_index"],
    "chunk_text": chunks_df["chunk_text"],
    "embedding": all_chunk_embeddings,
})

print(f"Computed {len(chunk_embeddings_df)} chunk embeddings using {EMBEDDING_MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Upsert chunk embeddings into Lakebase
# MAGIC
# MAGIC `id` is `document_id + '_' + chunk_index` (mirrors the news chunk table).
# MAGIC Same pgvector-literal insert as the whole-document embeddings.

# COMMAND ----------

# DBTITLE 1,Insert chunk embeddings
from datetime import datetime

import psycopg2
from psycopg2.extras import execute_values

chunk_embeddings_df['id'] = (
    chunk_embeddings_df['document_id'] + '_' + chunk_embeddings_df['chunk_index'].astype(str)
)
chunk_embeddings_df['model_name'] = EMBEDDING_MODEL_NAME
chunk_embeddings_df['embedded_at'] = datetime.now()

chunk_embeddings_rows = chunk_embeddings_df.to_dict('records')

if len(chunk_embeddings_rows) > 0:
    print(f"Inserting {len(chunk_embeddings_rows)} chunk embeddings into {CHUNK_EMBEDDINGS_TABLE_NAME}...")

    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password,
        sslmode='require',
    )

    try:
        cursor = conn.cursor()

        insert_data = [
            (
                row['id'],
                row['document_id'],
                row['location'],
                row['source_type'],
                int(row['chunk_index']),
                row['chunk_text'],
                '[' + ','.join(str(float(x)) for x in row['embedding']) + ']',
                row['model_name'],
                row['embedded_at'],
            )
            for row in chunk_embeddings_rows
        ]

        insert_sql = f"""
            INSERT INTO {CHUNK_EMBEDDINGS_TABLE_NAME} (
                id, document_id, location, source_type, chunk_index, chunk_text, embedding, model_name, embedded_at
            ) VALUES %s
            ON CONFLICT (id) DO NOTHING
        """

        template = "(%s, %s, %s, %s, %s, %s, %s::vector, %s, %s)"
        execute_values(cursor, insert_sql, insert_data, template=template, page_size=100)

        conn.commit()
        print(f"✅ Successfully inserted {cursor.rowcount} new chunk embeddings")
        print(f"   (Duplicates skipped via ON CONFLICT (id) DO NOTHING)")
    finally:
        cursor.close()
        conn.close()
else:
    print("No chunk embeddings to write.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Done
# MAGIC
# MAGIC `weather_embeddings` (whole-document) and `weather_chunk_embeddings`
# MAGIC (sliding-window passages) are now populated with `VECTOR(384)` embeddings
# MAGIC and their HNSW cosine indexes are ready. `POST /weather/search` can now run
# MAGIC pgvector similarity search (`embedding <=> %s::vector`) over either table.
