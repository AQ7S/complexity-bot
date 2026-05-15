"""Export M1 bars from DuckDB → CSV for Google Colab upload.

Usage:
    python -m engine.models.export_for_colab --symbol EURUSD --days 90
    python -m engine.models.export_for_colab --symbols EURUSD,USDJPY,XAUUSD --days 1095

Produces one CSV per symbol in engine/models/colab_data/ that you upload to Colab.
"""
from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

from engine.data import duckdb_store

OUTPUT_DIR = Path(__file__).resolve().parent / "colab_data"


def export(symbol: str, days: int, *, db_path: str | None = None) -> Path:
    """Export M1 bars for `symbol` to a CSV file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with duckdb_store.open_store(db_path, read_only=True) as con:
        max_ts = con.execute(
            "SELECT MAX(ts) FROM bars WHERE symbol=? AND timeframe='M1'", [symbol]
        ).fetchone()[0]
        if max_ts is None:
            raise RuntimeError(f"No M1 bars for {symbol} in DuckDB")
        cutoff = max_ts - timedelta(days=days)
        df = con.execute(
            """
            SELECT ts, open, high, low, close, volume
            FROM bars
            WHERE symbol=? AND timeframe='M1' AND ts >= ?
            ORDER BY ts
            """,
            [symbol, cutoff],
        ).fetchdf()
    out_path = OUTPUT_DIR / f"{symbol}_M1_{days}d.csv"
    df.to_csv(out_path, index=False)
    print(f"[OK] Exported {len(df)} M1 bars for {symbol} -> {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return out_path


def main() -> int:
    p = argparse.ArgumentParser(description="Export bars for Colab training")
    p.add_argument("--symbol", default="EURUSD")
    p.add_argument("--symbols", default=None, help="Comma-separated list")
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--db-path", default=None)
    args = p.parse_args()

    symbols = (
        [s.strip() for s in args.symbols.split(",") if s.strip()]
        if args.symbols
        else [args.symbol]
    )
    for sym in symbols:
        export(sym, args.days, db_path=args.db_path)

    print(f"\nAll CSVs saved to: {OUTPUT_DIR}")
    print("Upload these files to Google Colab and run the notebook!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
