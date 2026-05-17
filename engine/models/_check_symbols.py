"""Dev-only diagnostic: count M1 bars per symbol in DuckDB.

Run with: python -m engine.models._check_symbols
"""
from __future__ import annotations


def main() -> int:
    from engine.data import duckdb_store
    con = duckdb_store.connect(read_only=True)
    try:
        r = con.execute(
            "SELECT DISTINCT symbol, COUNT(*) as cnt FROM bars "
            "WHERE timeframe='M1' GROUP BY symbol ORDER BY symbol"
        ).fetchdf()
        print(r.to_string())
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
