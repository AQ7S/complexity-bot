"""Batch trainer for the CNN-LSTM.

Two tiers (Appendix J):
  * `build`       — EURUSD M5, 90 days, 5 epochs (≤30min CPU). Used by Phase 5 test.
  * `production`  — multi-symbol, 3 years, 50 epochs. Operator runs manually.

Usage:
    python engine/models/train_batch.py --tier build --symbol EURUSD
    python engine/models/train_batch.py --tier production --symbols EURUSD,USDJPY,XAUUSD --years 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

from engine.data import duckdb_store
from engine.models.cnn_lstm import N_CLASSES, build_model
from engine.models.dataset import build_windows

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

TIER_PROFILES: dict[str, dict] = {
    "build":      {"days": 90,        "epochs": 5,  "batch": 64, "lr": 1e-4,
                   "patience": 3,     "min_val_acc": 0.45},
    "production": {"days": 365 * 3,   "epochs": 50, "batch": 64, "lr": 1e-4,
                   "patience": 8,     "min_val_acc": 0.52},
}


def _resample_m5(df_m1: pd.DataFrame) -> pd.DataFrame:
    """Aggregate M1 → M5 OHLCV."""
    return df_m1.resample("5min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()


def load_bars(symbol: str, days: int, *, db_path: str | None = None) -> pd.DataFrame:
    """Load the most recent `days` of M1 bars and resample to M5."""
    with duckdb_store.open_store(db_path, read_only=True) as con:
        max_ts = con.execute(
            "SELECT MAX(ts) FROM bars WHERE symbol=? AND timeframe='M1'", [symbol]
        ).fetchone()[0]
        if max_ts is None:
            raise RuntimeError(f"No M1 bars for {symbol} in DuckDB")
        cutoff = max_ts - timedelta(days=days)
        rows = con.execute(
            """
            SELECT ts, open, high, low, close, volume
            FROM bars
            WHERE symbol=? AND timeframe='M1' AND ts >= ?
            ORDER BY ts
            """,
            [symbol, cutoff],
        ).fetchdf()
    if rows.empty:
        raise RuntimeError(f"No bars in window for {symbol}")
    rows = rows.set_index("ts").sort_index()
    return _resample_m5(rows)


def class_weights(y: np.ndarray, n_classes: int = N_CLASSES) -> torch.Tensor:
    """Inverse-frequency weights for CE loss."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / (n_classes * counts)
    return torch.tensor(w, dtype=torch.float32)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tier", choices=["build", "production"], default="build")
    p.add_argument("--symbol", default="EURUSD",
                   help="Single symbol for build tier")
    p.add_argument("--symbols", default=None,
                   help="Comma-separated symbols for production tier")
    p.add_argument("--days", type=int, default=None,
                   help="Override the tier's day window")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--db-path", default=None)
    p.add_argument("--device", default=None,
                   help="cpu | cuda | mps | dml; default = auto")
    return p.parse_args(argv)


def train(args: argparse.Namespace) -> dict:
    profile = TIER_PROFILES[args.tier]
    days = args.days or profile["days"]
    epochs = args.epochs or profile["epochs"]
    batch_size = args.batch or profile["batch"]

    # --- DirectML / CUDA / CPU device selection ---
    if args.device == "dml":
        import torch_directml  # noqa: PLC0415
        device = torch_directml.device()
    else:
        device = torch.device(
            args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
    n_threads = max(1, (torch.get_num_threads() or 1) - 2)
    torch.set_num_threads(n_threads)

    symbols = (
        [s.strip() for s in args.symbols.split(",") if s.strip()]
        if args.symbols
        else [args.symbol]
    )
    logger.info("training {} on {} ({} days, {} epochs, device={})",
                "cnn_lstm", symbols, days, epochs, device)

    Xs, ys = [], []
    for sym in symbols:
        bars = load_bars(sym, days, db_path=args.db_path)
        logger.info("  {}: {} M5 bars from {} → {}", sym, len(bars),
                    bars.index[0], bars.index[-1])
        X, y = build_windows(bars)
        logger.info("  {}: {} training windows; class counts={}",
                    sym, len(y), np.bincount(y, minlength=3).tolist())
        Xs.append(X); ys.append(y)
    X = np.concatenate(Xs)
    y = np.concatenate(ys)

    # Per-feature stats from training set only (for inference normalization).
    feat_mean = X.reshape(-1, X.shape[-1]).mean(axis=0)
    feat_std = X.reshape(-1, X.shape[-1]).std(axis=0)

    # Chronological 80/20 split — no shuffle across boundary.
    cut = int(len(X) * 0.8)
    X_tr, X_va = X[:cut], X[cut:]
    y_tr, y_va = y[:cut], y[cut:]

    # Reshape to (B, 1, 60, 50)
    def to_tensor(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(arr).unsqueeze(1)

    train_ds = TensorDataset(to_tensor(X_tr), torch.from_numpy(y_tr))
    val_ds = TensorDataset(to_tensor(X_va), torch.from_numpy(y_va))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = build_model(device)
    weights = class_weights(y_tr).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=profile["lr"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_acc = 0.0
    best_state = None
    patience = profile["patience"]
    epochs_no_improve = 0
    history: list[dict] = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_correct = 0
        tr_seen = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * yb.size(0)
            tr_correct += (logits.argmax(-1) == yb).sum().item()
            tr_seen += yb.size(0)
        scheduler.step()

        model.eval()
        va_loss = va_correct = va_seen = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                va_loss += loss.item() * yb.size(0)
                va_correct += (logits.argmax(-1) == yb).sum().item()
                va_seen += yb.size(0)

        tr_acc = tr_correct / max(tr_seen, 1)
        va_acc = va_correct / max(va_seen, 1)
        elapsed = time.time() - t0
        logger.info("  epoch {}/{}  tr_loss={:.4f} tr_acc={:.4f}  va_loss={:.4f} va_acc={:.4f}  ({:.1f}s)",
                    ep, epochs, tr_loss / max(tr_seen, 1), tr_acc,
                    va_loss / max(va_seen, 1), va_acc, elapsed)
        history.append({"epoch": ep, "tr_acc": tr_acc, "va_acc": va_acc,
                        "tr_loss": tr_loss / max(tr_seen, 1), "va_loss": va_loss / max(va_seen, 1)})

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info("  early stop at epoch {}", ep)
                break

    if best_state is None:
        best_state = model.state_dict()

    version = f"v{int(time.time())}_{datetime.utcnow().strftime('%Y%m%d')}"
    ckpt_path = CHECKPOINT_DIR / f"cnn_lstm_{version}.pt"
    torch.save({
        "model_state": best_state,
        "version": version,
        "tier": args.tier,
        "symbols": symbols,
        "feature_mean": feat_mean.tolist(),
        "feature_std": feat_std.tolist(),
        "best_val_acc": best_val_acc,
        "history": history,
    }, ckpt_path)
    logger.info("checkpoint saved → {} (best_val_acc={:.4f})", ckpt_path, best_val_acc)

    return {
        "checkpoint": str(ckpt_path),
        "best_val_acc": best_val_acc,
        "min_val_acc": profile["min_val_acc"],
        "elapsed_s": time.time() - t0,
        "history": history,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = train(args)
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))
    return 0 if result["best_val_acc"] >= result["min_val_acc"] else 2


if __name__ == "__main__":
    sys.exit(main())
