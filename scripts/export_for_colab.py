"""
Export DuckDB M1 bars to Parquet files for Google Colab GPU training.

Run from repo root with the engine venv active:
    python scripts/export_for_colab.py

Produces:  colab_data/  (one .parquet per symbol that has data)
Then zip it and upload to your Colab session.
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.data import duckdb_store  # noqa: E402

OUT_DIR = ROOT / "colab_data"
OUT_DIR.mkdir(exist_ok=True)

manifest: dict[str, dict] = {}

with duckdb_store.open_store(read_only=True) as con:
    # Auto-detect every symbol that has M1 bars in DuckDB.
    SYMBOLS = [
        r[0]
        for r in con.execute(
            "SELECT DISTINCT symbol FROM bars WHERE timeframe='M1' ORDER BY symbol"
        ).fetchall()
    ]
    print(f"Symbols found in DuckDB: {SYMBOLS}\n")

    for sym in SYMBOLS:
        df = con.execute(
            """
            SELECT ts, open, high, low, close, volume
            FROM bars
            WHERE symbol = ? AND timeframe = 'M1'
            ORDER BY ts
            """,
            [sym],
        ).fetchdf()

        if df.empty:
            print(f"  SKIP {sym}: no M1 bars in DuckDB")
            continue

        out_path = OUT_DIR / f"{sym.replace('#','_hash')}.parquet"
        df.to_parquet(out_path, index=False)
        manifest[sym] = {
            "rows": len(df),
            "from": str(df["ts"].iloc[0]),
            "to": str(df["ts"].iloc[-1]),
            "file": out_path.name,
        }
        print(f"  OK  {sym}: {len(df):,} bars  {df['ts'].iloc[0]} -> {df['ts'].iloc[-1]}")

# Write manifest so Colab knows what was exported.
(OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"\nManifest written: {len(manifest)} symbols exported to {OUT_DIR}")

# Zip everything for easy upload.
zip_path = ROOT / "colab_data.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in OUT_DIR.iterdir():
        zf.write(f, f.name)
print(f"ZIP ready: {zip_path}  ({zip_path.stat().st_size / 1_048_576:.1f} MB)")
print("\nNext step: upload colab_data.zip to Google Colab using the Files panel or files.upload().")
