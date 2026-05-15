"""Bulk historical data download.

Downloads ~3 years of tick data via dukascopy-node and M1 OHLCV via locally
provided HistData ZIPs, then ingests both into the DuckDB store.

Usage:
    python scripts/download_history.py \
        --symbols EURUSD,USDJPY,XAUUSD \
        --years 3 \
        --histdata-dir C:/path/to/histdata_zips
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.data import duckdb_store, ingest_dukascopy, ingest_histdata  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bulk download Dukascopy ticks + HistData M1 bars")
    p.add_argument("--symbols", default="EURUSD,USDJPY,XAUUSD",
                   help="Comma-separated internal symbols")
    p.add_argument("--years", type=int, default=3)
    p.add_argument("--out-dir", default=str(ROOT / "engine" / "data" / "downloads"),
                   help="Where the dukascopy CLI writes raw CSVs")
    p.add_argument("--histdata-dir", default=None,
                   help="Directory containing per-symbol HistData ZIP archives "
                        "(file names must include the symbol, e.g. HISTDATA_EURUSD_*.zip)")
    p.add_argument("--skip-ticks", action="store_true")
    p.add_argument("--skip-bars", action="store_true")
    return p.parse_args()


def find_histdata_files(directory: Path, symbol: str) -> list[Path]:
    sym_lc = symbol.lower()
    return [
        p for p in directory.glob("*")
        if p.suffix.lower() in (".zip", ".csv") and sym_lc in p.name.lower()
    ]


def main() -> int:
    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    today = date.today()
    date_from = today - timedelta(days=365 * args.years)

    with duckdb_store.open_store() as con:
        for sym in symbols:
            print(f"=== {sym} ===", flush=True)
            if not args.skip_ticks:
                sym_dir = out_dir / sym
                print(f"  ticks: dukascopy {date_from} → {today} → {sym_dir}", flush=True)
                try:
                    n = ingest_dukascopy.ingest(con, sym, date_from, today, sym_dir)
                    print(f"  ticks: inserted {n} rows", flush=True)
                except ingest_dukascopy.DukascopyCLIMissing as e:
                    print(f"  SKIP ticks: {e}", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"  ERROR ticks for {sym}: {e}", flush=True)

            if not args.skip_bars:
                if not args.histdata_dir:
                    print("  SKIP bars: --histdata-dir not provided", flush=True)
                    continue
                hd_dir = Path(args.histdata_dir)
                files = find_histdata_files(hd_dir, sym)
                if not files:
                    print(f"  SKIP bars: no HistData files for {sym} in {hd_dir}", flush=True)
                    continue
                total = 0
                for f in files:
                    total += ingest_histdata.ingest_path(con, f, sym)
                print(f"  bars: inserted {total} rows from {len(files)} file(s)", flush=True)

            report = duckdb_store.integrity_check(con, sym, "M1")
            print(f"  integrity: {report}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
