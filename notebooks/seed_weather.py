# Databricks notebook source
# MAGIC %md
# MAGIC # Seed weather_documents from the National Weather Service (Lakebase)
# MAGIC
# MAGIC One runnable unit: run this notebook top-to-bottom and nothing else.
# MAGIC It is the notebook counterpart of the standalone `seed_weather.py` script
# MAGIC — it calls the SAME `weather_sync.sync_weather()` code path the Flask
# MAGIC `POST /weather/sync` route uses — but bundles the psycopg2 fix that a bare
# MAGIC script can't perform on its own.
# MAGIC
# MAGIC **Why a notebook and not just `%run ./seed_weather`:** on Databricks the
# MAGIC pip `psycopg2`/`psycopg2-binary` wheel bundles its own OpenSSL, which
# MAGIC collides with the OpenSSL that `databricks-sdk`/grpc uses in a background
# MAGIC credential-refresh thread and SIGABRTs the kernel ("The Python kernel is
# MAGIC unresponsive") the moment `lakebase.py` imports psycopg2 and builds a
# MAGIC `WorkspaceClient()`. A running script cannot uninstall an already-loaded
# MAGIC C-extension or restart its own kernel, so the uninstall + `restartPython()`
# MAGIC MUST happen in cells BEFORE the seeder imports anything. This notebook does
# MAGIC exactly that (same ritual as `ingest_ticker_news_embeddings.py`).
# MAGIC
# MAGIC It re-uses the SAME Lakebase secret (scope `database`, key `lakebase-url`)
# MAGIC that `lakebase.py` uses in the Flask app — no extra secrets required.

# COMMAND ----------

# DBTITLE 1,Drop the OpenSSL-bundling psycopg2 wheel, install runtime deps
# MAGIC %pip uninstall -y psycopg2 psycopg2-binary
# MAGIC %pip install -q 'databricks-sdk>=0.118.0' requests

# COMMAND ----------

# DBTITLE 1,Restart Python so the runtime's system psycopg2 is what loads
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config
# MAGIC
# MAGIC Widgets mirror `seed_weather.py`'s env vars, so this notebook can also run
# MAGIC as a scheduled Databricks Job without editing code. Pick states that
# MAGIC currently have active NWS alerts (severe-weather states like TX/FL/OK/LA
# MAGIC are good bets) — the whole chain bootstraps off active alerts, so a state
# MAGIC with none yields 0 documents.

# COMMAND ----------

dbutils.widgets.text("states", "TX,FL", "States to harvest (comma-separated, e.g. TX,OK,LA)")
dbutils.widgets.text("sync_limit", "50", "Max alerts fetched per state")
dbutils.widgets.text("max_grids_per_state", "5", "Forecast grid cells expanded per state")

# COMMAND ----------

# DBTITLE 1,Put the repo root on sys.path so weather_sync/lakebase import cleanly
import os
import sys


def _add_repo_root_to_path() -> str | None:
    """Find the dir holding weather_sync.py (this notebook lives in <repo>/notebooks/).

    Databricks sets a notebook's cwd to its own folder, so the repo root is
    usually the parent of cwd — but we walk a few levels up as a fallback so
    this keeps working regardless of exactly where the notebook is attached.
    """
    seen: list[str] = []
    d = os.getcwd()
    for _ in range(6):
        if d and d not in seen:
            seen.append(d)
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    for candidate in seen + [os.path.dirname(os.getcwd())]:
        if candidate and os.path.exists(os.path.join(candidate, "weather_sync.py")):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
            return candidate
    return None


_repo_root = _add_repo_root_to_path()
if not _repo_root:
    raise RuntimeError(
        "Could not locate weather_sync.py — attach this notebook inside the repo "
        "(Git folder / Repo) so the app modules are importable, or add the repo "
        "root to sys.path manually."
    )
print(f"Repo root on sys.path: {_repo_root}")

# COMMAND ----------

# DBTITLE 1,Run the harvest (same code path as POST /weather/sync)
import logging

# weather_sync binds WEATHER_MAX_GRIDS_PER_STATE as a module constant at import
# time, so the widget override MUST be set in the environment before importing it.
os.environ["WEATHER_MAX_GRIDS_PER_STATE"] = dbutils.widgets.get("max_grids_per_state")

from weather_sync import parse_states, sync_weather

# Surface weather_sync's INFO progress logs (otherwise the harvest looks silent).
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

raw_states = [s for s in dbutils.widgets.get("states").split(",") if s.strip()]
states = parse_states(raw_states)
limit = int(dbutils.widgets.get("sync_limit"))

print(f"Harvesting weather for states {states or 'DEFAULT_WEATHER_STATES'} (limit={limit}) ...")
result = sync_weather(states=states, limit=limit)
print(
    f"Seeded {result['synced']} rows into weather_documents "
    f"({result['documents']} harvested) for states {result['states']}"
)
result
