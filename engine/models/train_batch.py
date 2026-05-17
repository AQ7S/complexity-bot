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
from engine.learning.purged_cv import aggregate_folds, purged_kfold_indices
from engine.learning.triple_barrier import build_triple_barrier_dataset
from engine.models.cnn_lstm import N_CLASSES, build_model
from engine.models.dataset import build_feature_frame, build_windows
from engine.features.feature_pipeline import N_FEATURES, SEQUENCE_LEN

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

TIER_PROFILES: dict[str, dict] = {
    "build":      {"days": 90,        "epochs": 5,  "batch": 64, "lr": 1e-4,
                   "patience": 3,     "min_val_acc": 0.45},
    "production": {"days": 365 * 3,   "epochs": 50, "batch": 64, "lr": 1e-4,
                   "patience": 8,     "min_val_acc": 0.52},
}


def build_windows_triple_barrier(
    bars: pd.DataFrame,
    *,
    pt_mult: float,
    sl_mult: float,
    max_h: int,
    sequence_len: int = SEQUENCE_LEN,
    warmup: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (X, y, label_horizons) where labels come from triple-barrier method.

    `label_horizons[i]` is the bar index at which label `i` was determined —
    used by purged CV to drop overlapping training samples.
    """
    feats = build_feature_frame(bars)
    feats = feats.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    feat_arr = feats.to_numpy(dtype=np.float32, copy=False)

    mu = feat_arr[warmup : warmup + sequence_len * 200].mean(axis=0, keepdims=True)
    sd = feat_arr[warmup : warmup + sequence_len * 200].std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-9, 1.0, sd)
    feat_norm = np.clip((feat_arr - mu) / sd, -10.0, 10.0).astype(np.float32, copy=False)

    t0, labels, _ = build_triple_barrier_dataset(
        bars, pt_mult=pt_mult, sl_mult=sl_mult, max_h=max_h,
        skip_first=max(warmup, sequence_len),
    )
    # Need full lookback window available behind each t0.
    valid = t0 >= (sequence_len - 1)
    t0 = t0[valid]
    labels = labels[valid]
    if len(t0) == 0:
        raise ValueError("triple-barrier produced no usable samples")

    X = np.empty((len(t0), sequence_len, N_FEATURES), dtype=np.float32)
    for k, i in enumerate(t0):
        X[k] = feat_norm[i - sequence_len + 1 : i + 1]
    # Label horizon = end-of-trade bar (t0 + max_h, clipped).
    horizons = np.minimum(t0 + max_h, len(feat_norm) - 1).astype(np.int64)
    return X, labels.astype(np.int64), horizons


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


def _run_purged_cv_eval(
    X: np.ndarray,
    y: np.ndarray,
    horizons: np.ndarray | None,
    *,
    n_splits: int,
    embargo_pct: float,
    device: torch.device,
    batch_size: int,
    eval_epochs: int,
    lr: float,
) -> dict[str, float]:
    """Honest OOS validation via purged + embargoed k-fold.

    Trains a fresh model per fold for `eval_epochs` (small) and reports
    per-fold + aggregate accuracy. Does NOT save these models — purely
    measurement.
    """
    fold_scores: list[float] = []
    for k, train_idx, test_idx in purged_kfold_indices(
        n_samples=len(X), n_splits=n_splits,
        label_horizons=horizons, embargo_pct=embargo_pct,
    ):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        tr = TensorDataset(torch.from_numpy(X_tr).unsqueeze(1), torch.from_numpy(y_tr))
        te = TensorDataset(torch.from_numpy(X_te).unsqueeze(1), torch.from_numpy(y_te))
        tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True, drop_last=True)
        te_loader = DataLoader(te, batch_size=batch_size, shuffle=False)
        model = build_model(device)
        weights = class_weights(y_tr).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        for _ in range(eval_epochs):
            model.train()
            for xb, yb in tr_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        model.eval()
        correct = seen = 0
        with torch.no_grad():
            for xb, yb in te_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                correct += (logits.argmax(-1) == yb).sum().item()
                seen += yb.size(0)
        acc = correct / max(seen, 1)
        fold_scores.append(acc)
        logger.info("  fold {}/{}: n_train={} n_test={} acc={:.4f}",
                    k + 1, n_splits, len(train_idx), len(test_idx), acc)
    if not fold_scores:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n_folds": 0}
    arr = np.array(fold_scores, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n_folds": int(arr.size),
    }


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
    p.add_argument("--ewc-lambda", type=float, default=0.0,
                   dest="ewc_lambda",
                   help="EWC regularization strength (0 = disabled). "
                        "Typical: 1000-5000. Requires --ewc-prior-checkpoint.")
    p.add_argument("--ewc-prior-checkpoint", default=None,
                   dest="ewc_prior_checkpoint",
                   help="Path to a prior .pt checkpoint whose weights/Fisher to protect.")
    p.add_argument("--label-method",
                   choices=["next_bar", "triple_barrier"],
                   default="next_bar",
                   dest="label_method",
                   help="Labeling scheme. triple_barrier uses pt/sl/timeout barriers "
                        "(López de Prado AFML ch. 3) for realistic trade-outcome labels.")
    p.add_argument("--pt-mult", type=float, default=2.0, dest="pt_mult",
                   help="Profit-target multiplier for triple_barrier labels.")
    p.add_argument("--sl-mult", type=float, default=1.0, dest="sl_mult",
                   help="Stop-loss multiplier for triple_barrier labels.")
    p.add_argument("--max-h", type=int, default=48, dest="max_h",
                   help="Maximum holding bars (vertical barrier) for triple_barrier.")
    p.add_argument("--cv",
                   choices=["holdout", "purged_kfold"],
                   default="holdout",
                   dest="cv",
                   help="Validation method. purged_kfold runs López-de-Prado purged "
                        "+ embargoed walk-forward k-fold CV.")
    p.add_argument("--n-splits", type=int, default=5, dest="n_splits",
                   help="Number of folds for purged_kfold CV.")
    p.add_argument("--embargo-pct", type=float, default=0.01, dest="embargo_pct",
                   help="Embargo width as fraction of n_samples for purged_kfold.")
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

    label_method = getattr(args, "label_method", "next_bar")
    Xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    horizons_list: list[np.ndarray] = []
    for sym in symbols:
        bars = load_bars(sym, days, db_path=args.db_path)
        logger.info("  {}: {} M5 bars from {} → {}", sym, len(bars),
                    bars.index[0], bars.index[-1])
        if label_method == "triple_barrier":
            Xs_, ys_, horiz_ = build_windows_triple_barrier(
                bars,
                pt_mult=args.pt_mult,
                sl_mult=args.sl_mult,
                max_h=args.max_h,
            )
            horizons_list.append(horiz_)
        else:
            Xs_, ys_ = build_windows(bars)
        logger.info("  {}: {} training windows ({} labels); class counts={}",
                    sym, len(ys_), label_method,
                    np.bincount(ys_, minlength=3).tolist())
        Xs.append(Xs_); ys.append(ys_)
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    horizons = np.concatenate(horizons_list) if horizons_list else None

    # Per-feature stats from training set only (for inference normalization).
    feat_mean = X.reshape(-1, X.shape[-1]).mean(axis=0)
    feat_std = X.reshape(-1, X.shape[-1]).std(axis=0)

    cv_report: dict[str, float] | None = None
    if getattr(args, "cv", "holdout") == "purged_kfold":
        cv_report = _run_purged_cv_eval(
            X, y, horizons,
            n_splits=int(args.n_splits),
            embargo_pct=float(args.embargo_pct),
            device=device,
            batch_size=batch_size,
            eval_epochs=max(2, min(epochs // 3, 5)),
            lr=profile["lr"],
        )
        logger.info("purged_kfold: {}", cv_report)

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

    ewc_snapshot = None
    ewc_lambda = float(getattr(args, "ewc_lambda", 0.0) or 0.0)
    prior_ckpt = getattr(args, "ewc_prior_checkpoint", None)
    if prior_ckpt and ewc_lambda > 0:
        try:
            from engine.models.ewc import compute_fisher_information
            prior_state = torch.load(prior_ckpt, map_location=device, weights_only=False)
            tmp_model = build_model(device)
            tmp_model.load_state_dict(prior_state["model_state"])
            ewc_snapshot = compute_fisher_information(
                tmp_model, train_loader, device=device, max_batches=100,
            )
            logger.info("EWC enabled: lambda={} prior={} fisher_params={}",
                        ewc_lambda, prior_ckpt, len(ewc_snapshot.fisher))
        except Exception as e:  # noqa: BLE001
            logger.warning("EWC setup failed, training without it: {}", e)
            ewc_snapshot = None

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
            task_loss = criterion(logits, yb)
            if ewc_snapshot is not None and ewc_lambda > 0:
                from engine.models.ewc import ewc_penalty
                loss = task_loss + ewc_penalty(model, ewc_snapshot, lambda_ewc=ewc_lambda)
            else:
                loss = task_loss
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

    out = {
        "checkpoint": str(ckpt_path),
        "best_val_acc": best_val_acc,
        "min_val_acc": profile["min_val_acc"],
        "elapsed_s": time.time() - t0,
        "history": history,
        "label_method": label_method,
    }
    if cv_report is not None:
        out["purged_kfold"] = cv_report
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = train(args)
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))
    return 0 if result["best_val_acc"] >= result["min_val_acc"] else 2


if __name__ == "__main__":
    sys.exit(main())
