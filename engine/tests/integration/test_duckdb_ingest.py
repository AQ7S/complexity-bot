"""Integration tests for Phase 2 — DuckDB store + Dukascopy + HistData ingest.

These tests use synthetic fixtures (small canned CSV/ZIP files) so the pipeline
can be exercised offline. The bulk 3-year download is invoked separately via
`scripts/download_history.py`.

Plan target: 6 tests pass (3 symbols × 2 sources).
"""
from __future__ import annotations

import io
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from engine.data import duckdb_store, ingest_dukascopy, ingest_histdata

SYMBOLS = ["EURUSD", "USDJPY", "XAUUSD"]
PRICE_BASE = {"EURUSD": 1.07, "USDJPY": 150.0, "XAUUSD": 2350.0}


def _write_dukascopy_csv(path: Path, symbol: str, n: int = 600) -> None:
    """Write a fake dukascopy-node CSV (timestamp ms, askPrice, bidPrice, askVolume, bidVolume)."""
    base_price = PRICE_BASE[symbol]
    start = datetime(2025, 1, 6, 8, 0, 0)  # Monday, NY morning
    rows = []
    for i in range(n):
        ts_ms = int((start + timedelta(seconds=i)).timestamp() * 1000)
        bid = base_price + (i % 50) * 0.0001
        ask = bid + 0.0002
        rows.append((ts_ms, ask, bid, 1.0, 1.0))
    df = pd.DataFrame(rows, columns=["timestamp", "askPrice", "bidPrice", "askVolume", "bidVolume"])
    df.to_csv(path, index=False)


def _write_histdata_zip(path: Path, symbol: str, n_minutes: int = 600) -> None:
    """Write a fake HistData M1 ZIP with one CSV inside."""
    base = PRICE_BASE[symbol]
    start = datetime(2025, 1, 6, 8, 0, 0)  # naive EST
    lines = []
    for i in range(n_minutes):
        ts = start + timedelta(minutes=i)
        o = base + (i % 30) * 0.0001
        h = o + 0.0005
        low = o - 0.0005
        c = o + 0.0001
        v = 100 + i
        lines.append(f"{ts.strftime('%Y%m%d %H%M%S')};{o};{h};{low};{c};{v}")
    csv_bytes = ("\n".join(lines)).encode("utf-8")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"DAT_ASCII_{symbol}_M1_2025.csv", csv_bytes)


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("duckdb_store") / "market.duckdb"
    con = duckdb_store.connect(db_path)
    fixtures_dir = tmp_path_factory.mktemp("fixtures")

    # Pre-populate fixtures + ingest both sources for all symbols.
    for sym in SYMBOLS:
        sym_dir = fixtures_dir / sym
        sym_dir.mkdir()
        csv_path = sym_dir / f"{sym}_ticks.csv"
        _write_dukascopy_csv(csv_path, sym, n=600)
        ingest_dukascopy.ingest_directory(con, sym_dir, sym)

        zip_path = fixtures_dir / f"HISTDATA_{sym}_M1.zip"
        _write_histdata_zip(zip_path, sym, n_minutes=600)
        ingest_histdata.ingest_path(con, zip_path, sym)

    yield con
    con.close()


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_dukascopy_ticks_ingested(store, symbol):
    n = duckdb_store.row_count(
        store, "ticks", where="symbol=? AND source='dukascopy'", params=[symbol]
    )
    assert n > 0, f"no dukascopy ticks for {symbol}"

    # Spread sanity: ask must be > bid for every row we wrote.
    bad = duckdb_store.row_count(
        store, "ticks", where="symbol=? AND ask <= bid", params=[symbol]
    )
    assert bad == 0, f"{bad} ticks with ask<=bid for {symbol}"


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_histdata_bars_ingested(store, symbol):
    report = duckdb_store.integrity_check(store, symbol, "M1")
    assert report["bars_count"] > 0, f"no M1 bars for {symbol}"
    assert report["ohlc_nulls"] == 0, f"OHLC nulls for {symbol}: {report}"
    assert report["max_gap_minutes"] < 5, f"M1 gap too large for {symbol}: {report}"
