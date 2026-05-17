from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger


Direction = Literal["BUY", "SELL"]

DEFAULT_SLIPPAGE_PIPS = 0.5
DEFAULT_RISK_PCT = 0.02
DEFAULT_ATR_SL_MULT = 1.5
DEFAULT_TP_RR = 2.0
DEFAULT_TIME_EXIT_BARS = 48
DEFAULT_STARTING_EQUITY = 10_000.0


DEFAULT_SPREAD_PIPS = {
    "EURUSD#": 1.2, "EURUSD": 1.2,
    "USDJPY#": 1.3, "USDJPY": 1.3,
    "GBPUSD#": 1.5, "GBPUSD": 1.5,
    "USDCHF#": 1.6, "USDCHF": 1.6,
    "EURJPY#": 1.7, "EURJPY": 1.7,
    "AUDUSD#": 1.4, "AUDUSD": 1.4,
    "GOLD#":   25.0, "XAUUSD": 25.0, "XAUUSD#": 25.0,
    "BTCUSD#": 2500.0, "BTCUSD": 2500.0,
    "ETHUSD#": 150.0, "ETHUSD": 150.0,
    "AI_INDX#": 5.0, "Crypto_10#": 5.0,
    "TrumpWinners#": 0.5, "HarrisWinners#": 0.5,
}


DEFAULT_SWAP_LONG_PIPS = {
    "EURUSD#": -0.3, "EURUSD": -0.3,
    "USDJPY#": 0.1,  "USDJPY": 0.1,
    "GBPUSD#": -0.5, "GBPUSD": -0.5,
    "USDCHF#": 0.2,  "USDCHF": 0.2,
    "EURJPY#": -0.4, "EURJPY": -0.4,
    "AUDUSD#": -0.2, "AUDUSD": -0.2,
    "GOLD#": -8.0,   "XAUUSD": -8.0, "XAUUSD#": -8.0,
    "BTCUSD#": 0.0,  "ETHUSD#": 0.0,
    "AI_INDX#": 0.0, "Crypto_10#": 0.0,
    "TrumpWinners#": 0.0, "HarrisWinners#": 0.0,
}
DEFAULT_SWAP_SHORT_PIPS = {
    "EURUSD#": 0.1,  "EURUSD": 0.1,
    "USDJPY#": -0.4, "USDJPY": -0.4,
    "GBPUSD#": 0.2,  "GBPUSD": 0.2,
    "USDCHF#": -0.5, "USDCHF": -0.5,
    "EURJPY#": 0.1,  "EURJPY": 0.1,
    "AUDUSD#": -0.1, "AUDUSD": -0.1,
    "GOLD#": 3.0,    "XAUUSD": 3.0, "XAUUSD#": 3.0,
    "BTCUSD#": 0.0,  "ETHUSD#": 0.0,
    "AI_INDX#": 0.0, "Crypto_10#": 0.0,
    "TrumpWinners#": 0.0, "HarrisWinners#": 0.0,
}


@dataclass
class BacktestConfig:
    symbol: str
    from_date: datetime
    to_date: datetime
    timeframe: str = "M5"
    starting_equity: float = DEFAULT_STARTING_EQUITY
    risk_pct: float = DEFAULT_RISK_PCT
    atr_sl_mult: float = DEFAULT_ATR_SL_MULT
    tp_rr: float = DEFAULT_TP_RR
    time_exit_bars: int = DEFAULT_TIME_EXIT_BARS
    slippage_pips: float = DEFAULT_SLIPPAGE_PIPS
    spread_pips: float | None = None
    swap_long_pips: float | None = None
    swap_short_pips: float | None = None
    min_confluence: int = 3


@dataclass
class SimulatedTrade:
    open_time: datetime
    close_time: datetime
    direction: Direction
    entry: float
    exit: float
    sl: float
    tp: float
    bars_held: int
    close_reason: Literal["TP", "SL", "TIME_EXIT"]
    gross_pnl_pips: float
    spread_cost_pips: float
    slippage_cost_pips: float
    swap_cost_pips: float
    net_pnl_pips: float
    net_pnl_usd: float
    r_multiple: float


@dataclass
class BacktestReport:
    config: BacktestConfig
    total_trades: int
    wins: int
    losses: int
    breakeven_or_time_exit: int
    win_rate: float
    gross_pnl_usd: float
    total_costs_usd: float
    net_pnl_usd: float
    avg_r_multiple: float
    sharpe: float
    max_drawdown_pct: float
    profit_factor: float
    spread_pips_used: float
    slippage_pips_used: float
    swap_long_pips_used: float
    swap_short_pips_used: float
    starting_equity: float
    ending_equity: float
    trades: list[SimulatedTrade] = field(default_factory=list)


def _pip_size(symbol: str) -> float:
    if "JPY" in symbol.upper():
        return 0.01
    if any(s in symbol.upper() for s in ("XAU", "GOLD", "BTC", "ETH", "INDX", "CRYPTO", "TRUMP", "HARRIS")):
        return 1.0
    return 0.0001


def _resample_to_m5(m1: pd.DataFrame) -> pd.DataFrame:
    m1 = m1.copy()
    m1["ts"] = pd.to_datetime(m1["ts"])
    m1 = m1.set_index("ts").sort_index()
    out = pd.DataFrame({
        "open":   m1["open"].resample("5min").first(),
        "high":   m1["high"].resample("5min").max(),
        "low":    m1["low"].resample("5min").min(),
        "close":  m1["close"].resample("5min").last(),
        "volume": m1["volume"].resample("5min").sum(),
    }).dropna()
    return out


def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema9"]  = out["close"].ewm(span=9, adjust=False).mean()
    out["ema21"] = out["close"].ewm(span=21, adjust=False).mean()
    out["ema50"] = out["close"].ewm(span=50, adjust=False).mean()
    delta = out["close"].diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean().replace(0, np.nan)
    rs = gain / loss
    out["rsi14"] = 100 - (100 / (1 + rs))
    tr1 = (out["high"] - out["low"]).abs()
    tr2 = (out["high"] - out["close"].shift()).abs()
    tr3 = (out["low"]  - out["close"].shift()).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    out["atr14"] = tr.rolling(14, min_periods=14).mean()
    return out.dropna()


def _signal_for_bar(row: pd.Series, prev: pd.Series) -> tuple[Direction | None, int]:
    votes_buy = 0
    votes_sell = 0
    if row["ema9"] > row["ema21"] > row["ema50"]:
        votes_buy += 1
    elif row["ema9"] < row["ema21"] < row["ema50"]:
        votes_sell += 1
    if row["rsi14"] < 30:
        votes_buy += 1
    elif row["rsi14"] > 70:
        votes_sell += 1
    if prev["ema9"] <= prev["ema21"] and row["ema9"] > row["ema21"]:
        votes_buy += 1
    elif prev["ema9"] >= prev["ema21"] and row["ema9"] < row["ema21"]:
        votes_sell += 1
    if row["close"] > row["ema50"]:
        votes_buy += 1
    elif row["close"] < row["ema50"]:
        votes_sell += 1
    if votes_buy >= votes_sell and votes_buy >= 2:
        return "BUY", votes_buy
    if votes_sell > votes_buy and votes_sell >= 2:
        return "SELL", votes_sell
    return None, 0


def _load_bars(
    symbol: str, from_date: datetime, to_date: datetime,
    db_path: str | None = None,
) -> pd.DataFrame:
    from engine.data import duckdb_store
    candidates = [symbol, symbol.rstrip("#")]
    rows: pd.DataFrame | None = None
    with duckdb_store.open_store(db_path, read_only=True) as con:
        for sym in candidates:
            rows = con.execute(
                "SELECT ts, open, high, low, close, COALESCE(volume,0.0) AS volume "
                "FROM bars WHERE symbol = ? AND timeframe IN ('M1','M5') "
                "AND ts BETWEEN ? AND ? ORDER BY ts",
                [sym, from_date, to_date],
            ).fetchdf()
            if not rows.empty:
                break
    if rows is None or rows.empty:
        raise ValueError(
            f"No bars found in DuckDB for {symbol} between {from_date} and {to_date}. "
            f"Tried {candidates}."
        )
    if rows["ts"].diff().median() < pd.Timedelta(minutes=2):
        rows = _resample_to_m5(rows)
    else:
        rows = rows.copy()
        rows["ts"] = pd.to_datetime(rows["ts"])
        rows = rows.set_index("ts").sort_index()
    return _compute_indicators(rows)


def _resolve_costs(cfg: BacktestConfig) -> tuple[float, float, float]:
    spread = cfg.spread_pips
    if spread is None:
        spread = DEFAULT_SPREAD_PIPS.get(cfg.symbol)
        if spread is None:
            spread = DEFAULT_SPREAD_PIPS.get(cfg.symbol.rstrip("#"), 1.5)
    swap_long = cfg.swap_long_pips
    if swap_long is None:
        swap_long = DEFAULT_SWAP_LONG_PIPS.get(cfg.symbol, DEFAULT_SWAP_LONG_PIPS.get(cfg.symbol.rstrip("#"), 0.0))
    swap_short = cfg.swap_short_pips
    if swap_short is None:
        swap_short = DEFAULT_SWAP_SHORT_PIPS.get(cfg.symbol, DEFAULT_SWAP_SHORT_PIPS.get(cfg.symbol.rstrip("#"), 0.0))
    return float(spread), float(swap_long), float(swap_short)


def _pip_value_usd(symbol: str, lot: float = 1.0) -> float:
    if "JPY" in symbol.upper():
        return 9.0 * lot
    if any(s in symbol.upper() for s in ("XAU", "GOLD")):
        return 10.0 * lot
    if "BTC" in symbol.upper() or "ETH" in symbol.upper():
        return 1.0 * lot
    return 10.0 * lot


def run_backtest(cfg: BacktestConfig, *, db_path: str | None = None) -> BacktestReport:
    bars = _load_bars(cfg.symbol, cfg.from_date, cfg.to_date, db_path)
    if len(bars) < 100:
        raise ValueError(f"Not enough bars ({len(bars)}) to backtest")

    pip = _pip_size(cfg.symbol)
    spread_pips, swap_long_pips, swap_short_pips = _resolve_costs(cfg)
    slippage_pips = float(cfg.slippage_pips)
    pip_value_usd = _pip_value_usd(cfg.symbol)

    bars_list = list(bars.itertuples(index=True, name=None))
    trades: list[SimulatedTrade] = []
    equity = float(cfg.starting_equity)
    equity_curve: list[float] = [equity]
    peak_equity = equity
    max_dd_pct = 0.0
    open_trade: dict | None = None
    bar_idx = 0

    while bar_idx < len(bars_list) - 1:
        ts, o, h, l, c, v, ema9, ema21, ema50, rsi14, atr14 = bars_list[bar_idx]
        if open_trade is None:
            if bar_idx < 1:
                bar_idx += 1
                continue
            prev_row = bars.iloc[bar_idx - 1]
            cur_row = bars.iloc[bar_idx]
            direction, votes = _signal_for_bar(cur_row, prev_row)
            if direction is None or votes < cfg.min_confluence:
                bar_idx += 1
                continue
            entry_raw = c
            slip_adj = slippage_pips * pip
            if direction == "BUY":
                entry = entry_raw + slip_adj
                sl = entry - cfg.atr_sl_mult * atr14
                tp = entry + cfg.atr_sl_mult * atr14 * cfg.tp_rr
            else:
                entry = entry_raw - slip_adj
                sl = entry + cfg.atr_sl_mult * atr14
                tp = entry - cfg.atr_sl_mult * atr14 * cfg.tp_rr
            risk_distance_pips = abs(entry - sl) / pip
            if risk_distance_pips <= 0:
                bar_idx += 1
                continue
            risk_usd = equity * cfg.risk_pct
            lot = max(0.01, risk_usd / (risk_distance_pips * pip_value_usd))
            open_trade = {
                "ts_open": ts, "direction": direction, "entry": entry, "sl": sl,
                "tp": tp, "bar_open_idx": bar_idx, "lot": lot,
            }
            bar_idx += 1
            continue

        ts, o, h, l, c, v, *_ = bars_list[bar_idx]
        d = open_trade["direction"]
        entry = open_trade["entry"]
        sl = open_trade["sl"]
        tp = open_trade["tp"]
        bars_held = bar_idx - open_trade["bar_open_idx"]
        close_reason: str | None = None
        exit_price: float | None = None

        if d == "BUY":
            if l <= sl:
                exit_price = sl
                close_reason = "SL"
            elif h >= tp:
                exit_price = tp
                close_reason = "TP"
        else:
            if h >= sl:
                exit_price = sl
                close_reason = "SL"
            elif l <= tp:
                exit_price = tp
                close_reason = "TP"

        if close_reason is None and bars_held >= cfg.time_exit_bars:
            exit_price = c
            close_reason = "TIME_EXIT"

        if close_reason is None:
            bar_idx += 1
            continue

        gross_pips = (exit_price - entry) / pip if d == "BUY" else (entry - exit_price) / pip

        bars_per_day = (24 * 60) // 5
        nights_held = max(0, bars_held // bars_per_day)
        swap_pips_per_night = swap_long_pips if d == "BUY" else swap_short_pips
        swap_cost_pips = nights_held * swap_pips_per_night

        net_pips = gross_pips - spread_pips - slippage_pips + swap_cost_pips
        lot = open_trade["lot"]
        net_usd = net_pips * pip_value_usd * lot
        spread_cost_pips = spread_pips
        slippage_cost_pips = slippage_pips

        risk_distance_pips = abs(entry - sl) / pip
        r_multiple = net_pips / risk_distance_pips if risk_distance_pips > 0 else 0.0

        trades.append(SimulatedTrade(
            open_time=open_trade["ts_open"], close_time=ts,
            direction=d, entry=entry, exit=exit_price, sl=sl, tp=tp,
            bars_held=bars_held, close_reason=close_reason,
            gross_pnl_pips=gross_pips,
            spread_cost_pips=spread_cost_pips,
            slippage_cost_pips=slippage_cost_pips,
            swap_cost_pips=swap_cost_pips,
            net_pnl_pips=net_pips,
            net_pnl_usd=net_usd,
            r_multiple=r_multiple,
        ))
        equity += net_usd
        equity_curve.append(equity)
        peak_equity = max(peak_equity, equity)
        dd_pct = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0.0
        max_dd_pct = max(max_dd_pct, dd_pct)
        open_trade = None
        bar_idx += 1

    wins = sum(1 for t in trades if t.net_pnl_usd > 0)
    losses = sum(1 for t in trades if t.net_pnl_usd < 0)
    flat = sum(1 for t in trades if t.net_pnl_usd == 0)
    win_rate = wins / len(trades) if trades else 0.0

    def _trade_lot(t: SimulatedTrade) -> float:
        if t.net_pnl_pips == 0 or pip_value_usd == 0:
            return 0.0
        return t.net_pnl_usd / (t.net_pnl_pips * pip_value_usd)

    gross_pnl_usd = sum(t.gross_pnl_pips * pip_value_usd * _trade_lot(t) for t in trades)
    total_costs_usd = sum(
        (t.spread_cost_pips + t.slippage_cost_pips - t.swap_cost_pips)
        * pip_value_usd * _trade_lot(t)
        for t in trades
    )
    net_pnl_usd = equity - cfg.starting_equity
    avg_r = sum(t.r_multiple for t in trades) / len(trades) if trades else 0.0

    returns = pd.Series(equity_curve).pct_change().dropna()
    if len(returns) > 1 and returns.std() > 0:
        sharpe = float(returns.mean() / returns.std() * math.sqrt(252))
    else:
        sharpe = 0.0

    gross_wins = sum(t.net_pnl_usd for t in trades if t.net_pnl_usd > 0)
    gross_losses = abs(sum(t.net_pnl_usd for t in trades if t.net_pnl_usd < 0))
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf") if gross_wins > 0 else 0.0

    return BacktestReport(
        config=cfg,
        total_trades=len(trades),
        wins=wins,
        losses=losses,
        breakeven_or_time_exit=flat,
        win_rate=win_rate,
        gross_pnl_usd=gross_pnl_usd,
        total_costs_usd=total_costs_usd,
        net_pnl_usd=net_pnl_usd,
        avg_r_multiple=avg_r,
        sharpe=sharpe,
        max_drawdown_pct=max_dd_pct,
        profit_factor=profit_factor,
        spread_pips_used=spread_pips,
        slippage_pips_used=slippage_pips,
        swap_long_pips_used=swap_long_pips,
        swap_short_pips_used=swap_short_pips,
        starting_equity=cfg.starting_equity,
        ending_equity=equity,
        trades=trades,
    )


def format_report(report: BacktestReport) -> str:
    lines = [
        f"=== Backtest: {report.config.symbol} {report.config.timeframe} "
        f"{report.config.from_date.date()} → {report.config.to_date.date()} ===",
        f"Costs applied (per trade):",
        f"  spread:    {report.spread_pips_used:.2f} pips",
        f"  slippage:  {report.slippage_pips_used:.2f} pips",
        f"  swap long: {report.swap_long_pips_used:.2f} pips/night",
        f"  swap short:{report.swap_short_pips_used:.2f} pips/night",
        f"Trades:      {report.total_trades} ({report.wins} wins / {report.losses} losses / {report.breakeven_or_time_exit} flat)",
        f"Win rate:    {report.win_rate*100:.2f}%",
        f"Net PnL:     ${report.net_pnl_usd:,.2f}  ({report.starting_equity:,.0f} → {report.ending_equity:,.0f})",
        f"Avg R-mult:  {report.avg_r_multiple:+.3f}",
        f"Sharpe:      {report.sharpe:.3f}",
        f"Profit fact: {report.profit_factor:.3f}",
        f"Max DD:      {report.max_drawdown_pct:.2f}%",
    ]
    if report.sharpe < 0.5:
        lines.append("*** WARNING: Sharpe < 0.5 with costs — strategy edge insufficient to deploy live ***")
    return "\n".join(lines)
