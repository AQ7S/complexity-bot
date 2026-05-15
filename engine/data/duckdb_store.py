from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import duckdb
import pandas as pd

ENGINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ENGINE_ROOT / "data" / "store" / "market.duckdb"
SCHEMA_PATH = ENGINE_ROOT / "data" / "schemas.sql"
ARCHIVE_DIR = ENGINE_ROOT / "data" / "archive"


def _resolve_db_path(db_path: str | os.PathLike[str] | None) -> Path:
    if db_path is None:
        env = os.environ.get("DUCKDB_PATH")
        if env:
            return Path(env).resolve()
        return DEFAULT_DB_PATH
    return Path(db_path).resolve()


def init_schema(con: duckdb.DuckDBPyConnection) -> None:
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    con.execute(sql)


def connect(db_path: str | os.PathLike[str] | None = None, *, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    path = _resolve_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path), read_only=read_only)
    if not read_only:
        init_schema(con)
    return con


@contextmanager
def open_store(db_path: str | os.PathLike[str] | None = None, *, read_only: bool = False) -> Iterator[duckdb.DuckDBPyConnection]:
    con = connect(db_path, read_only=read_only)
    try:
        yield con
    finally:
        con.close()


TICK_COLUMNS = ("symbol", "ts", "bid", "ask", "volume", "flags", "source")
BAR_COLUMNS = ("symbol", "timeframe", "ts", "open", "high", "low", "close", "volume", "spread")


def insert_ticks(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    frame = df.copy()
    for col in TICK_COLUMNS:
        if col not in frame.columns:
            if col == "source":
                frame[col] = "mt5"
            elif col in ("volume", "flags"):
                frame[col] = None
            else:
                raise ValueError(f"insert_ticks: missing required column {col!r}")
    frame = frame[list(TICK_COLUMNS)]
    frame["ts"] = pd.to_datetime(frame["ts"]).astype("datetime64[ms]")
    con.register("tmp_ticks", frame)
    con.execute("INSERT INTO ticks SELECT * FROM tmp_ticks")
    con.unregister("tmp_ticks")
    return len(frame)


def insert_bars(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    frame = df.copy()
    for col in BAR_COLUMNS:
        if col not in frame.columns:
            if col == "spread":
                frame[col] = None
            else:
                raise ValueError(f"insert_bars: missing required column {col!r}")
    frame = frame[list(BAR_COLUMNS)]
    # Drop intra-batch duplicates so the unique index isn't violated.
    frame = frame.drop_duplicates(subset=["symbol", "timeframe", "ts"], keep="last")
    con.register("tmp_bars", frame)
    # Anti-join against existing bars to avoid violating the unique index.
    # (DuckDB's ON CONFLICT requires a PK/UNIQUE *constraint*, not just an index.)
    con.execute(
        """
        INSERT INTO bars
        SELECT t.* FROM tmp_bars t
        WHERE NOT EXISTS (
          SELECT 1 FROM bars b
          WHERE b.symbol = t.symbol AND b.timeframe = t.timeframe AND b.ts = t.ts
        )
        """
    )
    con.unregister("tmp_bars")
    return len(frame)


def row_count(con: duckdb.DuckDBPyConnection, table: str, *, where: str | None = None, params: Sequence | None = None) -> int:
    sql = f"SELECT COUNT(*) FROM {table}"
    if where:
        sql += f" WHERE {where}"
    return int(con.execute(sql, params or []).fetchone()[0])


def integrity_check(
    con: duckdb.DuckDBPyConnection,
    symbol: str,
    timeframe: str = "M1",
    *,
    max_gap_minutes: int = 5,
) -> dict:
    """Run integrity checks against bars+ticks for one symbol.

    Returns a dict with: bars_count, ticks_count, ohlc_nulls, max_gap_minutes,
    spread_nonzero_pct (ticks where ask>bid).
    """
    bars_count = row_count(con, "bars", where="symbol=? AND timeframe=?", params=[symbol, timeframe])
    ticks_count = row_count(con, "ticks", where="symbol=?", params=[symbol])
    ohlc_nulls = int(
        con.execute(
            """
            SELECT COUNT(*) FROM bars
            WHERE symbol=? AND timeframe=?
              AND (open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL)
            """,
            [symbol, timeframe],
        ).fetchone()[0]
    )
    gap_row = con.execute(
        """
        WITH ordered AS (
          SELECT ts, LAG(ts) OVER (ORDER BY ts) AS prev_ts
          FROM bars WHERE symbol=? AND timeframe=?
        )
        SELECT COALESCE(MAX(date_diff('minute', prev_ts, ts)), 0) FROM ordered WHERE prev_ts IS NOT NULL
        """,
        [symbol, timeframe],
    ).fetchone()
    max_gap = int(gap_row[0]) if gap_row and gap_row[0] is not None else 0

    spread_row = con.execute(
        """
        SELECT
          COUNT(*) FILTER (WHERE ask > bid)::DOUBLE / NULLIF(COUNT(*),0)
        FROM ticks WHERE symbol=?
        """,
        [symbol],
    ).fetchone()
    spread_pct = float(spread_row[0]) if spread_row and spread_row[0] is not None else 0.0

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "bars_count": bars_count,
        "ticks_count": ticks_count,
        "ohlc_nulls": ohlc_nulls,
        "max_gap_minutes": max_gap,
        "max_gap_threshold": max_gap_minutes,
        "spread_nonzero_pct": spread_pct,
    }


def enforce_ring_buffer(
    con: duckdb.DuckDBPyConnection,
    symbols: Iterable[str],
    *,
    max_ticks: int,
    archive: bool = True,
) -> dict[str, int]:
    """Trim ticks table to the latest `max_ticks` rows per symbol.

    Older rows are appended to a per-symbol monthly Parquet archive before deletion.
    """
    deleted: dict[str, int] = {}
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    for symbol in symbols:
        cutoff_row = con.execute(
            "SELECT ts FROM ticks WHERE symbol=? ORDER BY ts DESC LIMIT 1 OFFSET ?",
            [symbol, max_ticks - 1],
        ).fetchone()
        if not cutoff_row:
            deleted[symbol] = 0
            continue
        cutoff = cutoff_row[0]
        if archive:
            sym_dir = ARCHIVE_DIR / symbol
            sym_dir.mkdir(parents=True, exist_ok=True)
            con.execute(
                f"""
                COPY (SELECT * FROM ticks WHERE symbol=? AND ts < ?)
                TO '{(sym_dir / "archive.parquet").as_posix()}'
                (FORMAT 'parquet', APPEND TRUE)
                """,
                [symbol, cutoff],
            )
        n = con.execute(
            "DELETE FROM ticks WHERE symbol=? AND ts < ? RETURNING 1", [symbol, cutoff]
        ).fetchall()
        deleted[symbol] = len(n)
    return deleted
