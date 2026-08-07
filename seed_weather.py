"""
Seed weather_documents by running the NWS harvest directly against Lakebase.

Self-contained: this is the ONLY thing you run. It imports sync_weather() from
weather_sync.py (which does NOT import Flask) and calls it in-process, so you do
NOT need `python app.py`, a running server, or Flask installed. Same code path
the POST /weather/sync route uses.

The harvest is state/area-driven: it pulls active NWS alerts for each state,
then follows each alert to its forecast grid cell for the daily + hourly
narratives. Pick states that currently have active alerts (severe-weather
states like TX/FL/OK/LA are good bets).

Usage:
    python seed_weather.py            # seeds DEFAULT_WEATHER_STATES (TX,FL)
    python seed_weather.py TX OK LA   # seed specific states

Env:
    WEATHER_STATES              default states when none passed (e.g. "TX,FL")
    WEATHER_SYNC_LIMIT          max alerts fetched per state (default 50)
    WEATHER_MAX_GRIDS_PER_STATE forecast grid cells expanded per state (default 5)
"""

import importlib.metadata as _md
import logging
import os
import sys


def _check_psycopg2_binary_wheel() -> None:
    """Fail fast if the OpenSSL-bundling psycopg2 wheel is installed.

    On Databricks, the pip `psycopg2-binary` (or a pip-built `psycopg2`) wheel
    bundles its own OpenSSL, which collides with the OpenSSL that
    databricks-sdk/grpc uses in a background credential-refresh thread. The
    collision SIGABRTs the kernel ("The Python kernel is unresponsive") the
    moment lakebase.py imports psycopg2 and constructs WorkspaceClient().

    A running script can't uninstall an already-loaded C extension or restart
    its own kernel, so the only fix is to remove the wheel and restart Python
    BEFORE this runs. Detect it here (via package metadata, without importing
    psycopg2) and print that fix instead of letting the kernel die silently.
    """
    installed = {d.metadata["Name"].lower() for d in _md.distributions()}
    culprit = installed & {"psycopg2-binary"}
    if culprit:
        sys.exit(
            "\n".join(
                [
                    f"Refusing to run: the pip wheel {sorted(culprit)!r} is installed.",
                    "It bundles its own OpenSSL, which collides with databricks-sdk/grpc",
                    "and crashes the kernel ('The Python kernel is unresponsive').",
                    "",
                    "Fix: run these FIRST, in separate notebook cells, then re-run this:",
                    "    %pip uninstall -y psycopg2 psycopg2-binary",
                    "    dbutils.library.restartPython()",
                    "",
                    "This falls back to the Databricks runtime's system psycopg2.",
                ]
            )
        )


_check_psycopg2_binary_wheel()

from weather_sync import sync_weather

# Surface weather_sync's INFO progress logs (the notebook/module path doesn't
# configure logging otherwise, so without this the harvest looks silent/hung).
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

LIMIT = int(os.environ.get("WEATHER_SYNC_LIMIT", 50))


def main(states: list[str]) -> None:
    print(f"Harvesting weather for states {states or 'DEFAULT_WEATHER_STATES'} ...")
    result = sync_weather(states=states, limit=LIMIT)
    print(
        f"Seeded {result['synced']} rows into weather_documents "
        f"({result['documents']} harvested) for states {result['states']}"
    )


if __name__ == "__main__":
    main([s.strip().upper() for s in sys.argv[1:] if s.strip()])
