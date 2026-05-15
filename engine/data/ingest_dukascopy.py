"""Tick ingestion via the Node-based dukascopy-node CLI.

The CLI is invoked exactly as:
    npx dukascopy-node -i <instrument> -from <YYYY-MM-DD> -to <YYYY-MM-DD>
        -t tick -f csv -dir <out_dir>

Resulting CSV schema (per dukascopy-node v4):
    timestamp,askPrice,bidPrice,askVolume,bidVolume

We parse each CSV file in the output directory and insert into the `ticks`
table with source='dukascopy'.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from .duckdb_store import insert_ticks

# Map our internal symbol → dukascopy instrument id.
SYMBOL_TO_INSTRUMENT = {
    "EURUSD": "eurusd",
    "USDJPY": "usdjpy",
    "GBPUSD": "gbpusd",
    "USDCHF": "usdchf",
    "GOLD#":  "xauusd",
    "BTCUSD": "btcusd",
    "ETHUSD": "ethusd",
}


class DukascopyCLIMissing(RuntimeError):
    pass


@dataclass(frozen=True)
class DukascopyJob:
    symbol: str
    date_from: date
    date_to: date
    out_dir: Path


def _check_npx() -> None:
    if shutil.which("npx") is None:
        raise DukascopyCLIMissing(
            "npx not found on PATH. Install Node.js LTS so the dukascopy-node CLI can run."
        )


def run_cli(job: DukascopyJob, *, timeout_s: int = 1800) -> Path:
    """Invoke the dukascopy-node CLI and return the directory it wrote into."""
    _check_npx()
    instrument = SYMBOL_TO_INSTRUMENT.get(job.symbol)
    if instrument is None:
        raise ValueError(f"No dukascopy mapping for symbol {job.symbol!r}")
    job.out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "npx", "--yes", "dukascopy-node",
        "-i", instrument,
        "-from", job.date_from.isoformat(),
        "-to", job.date_to.isoformat(),
        "-t", "tick",
        "-f", "csv",
        "-dir", str(job.out_dir),
    ]
    subprocess.run(cmd, check=True, timeout=timeout_s)
    return job.out_dir


def parse_csv(path: Path, symbol: str) -> pd.DataFrame:
    """Parse a single dukascopy-node CSV into the ticks-table shape."""
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    ts_col = cols.get("timestamp") or cols.get("time")
    bid_col = cols.get("bidprice") or cols.get("bid")
    ask_col = cols.get("askprice") or cols.get("ask")
    bvol = cols.get("bidvolume")
    avol = cols.get("askvolume")
    if not (ts_col and bid_col and ask_col):
        raise ValueError(f"Unexpected CSV columns in {path}: {list(df.columns)}")

    out = pd.DataFrame()
    out["symbol"] = [symbol] * len(df)
    ts = pd.to_numeric(df[ts_col], errors="coerce")
    out["ts"] = pd.to_datetime(ts, unit="ms", utc=True).dt.tz_convert(None)
    out["bid"] = df[bid_col].astype(float)
    out["ask"] = df[ask_col].astype(float)
    if bvol and avol:
        out["volume"] = (df[bvol].astype(float) + df[avol].astype(float))
    else:
        out["volume"] = None
    out["flags"] = None
    out["source"] = "dukascopy"
    out = out.dropna(subset=["ts", "bid", "ask"])
    return out


def ingest_directory(con: duckdb.DuckDBPyConnection, csv_dir: Path, symbol: str) -> int:
    total = 0
    for csv in sorted(csv_dir.glob("*.csv")):
        df = parse_csv(csv, symbol)
        total += insert_ticks(con, df)
    return total


def ingest(
    con: duckdb.DuckDBPyConnection,
    symbol: str,
    date_from: date,
    date_to: date,
    out_dir: Path,
) -> int:
    job = DukascopyJob(symbol=symbol, date_from=date_from, date_to=date_to, out_dir=out_dir)
    run_cli(job)
    return ingest_directory(con, out_dir, symbol)
