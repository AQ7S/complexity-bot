"""HistData M1 OHLCV ingestion.

HistData distributes free M1 data as ZIP archives containing one CSV per month.
The CSVs are semicolon-delimited with the schema (no header):
    YYYYMMDD HHMMSS;OPEN;HIGH;LOW;CLOSE;VOLUME

Timestamps are EST (UTC-5, no DST). We convert to naive UTC for storage.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import duckdb
import pandas as pd

from .duckdb_store import insert_bars

EST_OFFSET_HOURS = 5  # EST is UTC-5; HistData uses EST without DST


def _sniff_and_parse(raw_bytes: bytes) -> pd.DataFrame:
    """Detect ASCII (;-delimited, single dt column) vs MT (,-delimited, date+time)."""
    # Look at first non-blank line.
    head = raw_bytes.split(b"\n", 1)[0].decode("utf-8", errors="replace").strip()
    if ";" in head:
        # ASCII variant: YYYYMMDD HHMMSS;O;H;L;C;V
        df = pd.read_csv(
            io_buffer(raw_bytes),
            sep=";",
            header=None,
            names=["dt", "open", "high", "low", "close", "volume"],
        )
        ts_local = pd.to_datetime(df["dt"], format="%Y%m%d %H%M%S", errors="coerce")
    else:
        # MT variant: YYYY.MM.DD,HH:MM,O,H,L,C,V
        df = pd.read_csv(
            io_buffer(raw_bytes),
            sep=",",
            header=None,
            names=["date", "time", "open", "high", "low", "close", "volume"],
        )
        ts_local = pd.to_datetime(
            df["date"].astype(str) + " " + df["time"].astype(str),
            format="%Y.%m.%d %H:%M",
            errors="coerce",
        )
    return df.assign(_ts=ts_local)


def io_buffer(b: bytes):
    import io as _io
    return _io.BytesIO(b)


def parse_csv(path: Path, symbol: str) -> pd.DataFrame:
    raw = path.read_bytes()
    df = _sniff_and_parse(raw)
    out = pd.DataFrame({
        "symbol": symbol,
        "timeframe": "M1",
        "ts": df["_ts"] + pd.Timedelta(hours=EST_OFFSET_HOURS),
        "open": df["open"].astype(float),
        "high": df["high"].astype(float),
        "low": df["low"].astype(float),
        "close": df["close"].astype(float),
        "volume": df["volume"].astype(float),
        "spread": None,
    })
    return out.dropna(subset=["ts", "open", "high", "low", "close"])


def parse_zip(zip_path: Path, symbol: str) -> pd.DataFrame:
    frames = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".csv"):
                continue
            with zf.open(name) as fh:
                raw = fh.read()
            if not raw.strip():
                continue
            df = _sniff_and_parse(raw)
            out = pd.DataFrame({
                "symbol": symbol,
                "timeframe": "M1",
                "ts": df["_ts"] + pd.Timedelta(hours=EST_OFFSET_HOURS),
                "open": df["open"].astype(float),
                "high": df["high"].astype(float),
                "low": df["low"].astype(float),
                "close": df["close"].astype(float),
                "volume": df["volume"].astype(float),
                "spread": None,
            })
            frames.append(out.dropna(subset=["ts", "open", "high", "low", "close"]))
    if not frames:
        return pd.DataFrame(
            columns=["symbol", "timeframe", "ts", "open", "high", "low", "close", "volume", "spread"]
        )
    return pd.concat(frames, ignore_index=True)


def ingest_path(con: duckdb.DuckDBPyConnection, path: Path, symbol: str) -> int:
    """Ingest a single .csv or .zip file (or a directory of them) into the bars table."""
    if path.is_dir():
        total = 0
        for child in sorted(path.iterdir()):
            if child.suffix.lower() in (".csv", ".zip"):
                total += ingest_path(con, child, symbol)
        return total
    if path.suffix.lower() == ".zip":
        df = parse_zip(path, symbol)
    elif path.suffix.lower() == ".csv":
        df = parse_csv(path, symbol)
    else:
        return 0
    return insert_bars(con, df)
