# Complexity Engine — CNN-LSTM-SE-TemporalAttn-Residual Training (Google Colab T4 GPU)
# ===================================================================================
# HOW TO USE — 5 STEPS:
#
#   1. Open colab.research.google.com → New notebook
#   2. Runtime → Change runtime type → T4 GPU → Save
#   3. Upload THIS FILE to the Colab Files panel (folder icon, left sidebar)
#   4. Upload colab_data.zip to the same Files panel
#   5. In the single empty cell, paste and run:
#          exec(open('/content/ComplexityEngine_Colab_Train.py').read())
#
#   Training takes ~30-50 min on T4. Checkpoint auto-downloads when done.
#   Place it in:  engine\models\checkpoints\
# ===================================================================================

from __future__ import annotations
import subprocess, sys, os, zipfile, json, time, gc
from datetime import datetime, timedelta

MIGRATION_NOTE = """
ENGINE-SIDE MIGRATION REQUIRED before this checkpoint can be loaded by inference.py.

Architecture: CNN-LSTM-SE-TemporalAttn-Residual.
engine/models/cnn_lstm.py must be updated to mirror it.

New modules vs. the original CNNLSTM class:
  - SEBlock(channels, reduction=16) after each Conv block: .se1 (32ch), .se2 (64ch), .se3 (64ch)
  - TemporalAttention(256) replaces x[:, -1, :] after LSTM stack (Linear(256 -> 1, bias=False))
  - Residual skip in Conv block 3 only (no projection, c7 output 64ch matches block-3 input 64ch)

Labels: Triple Barrier (Lopez de Prado 2018) — HOLD=0, BUY=1, SELL=2.
  Note: order CHANGED from original BUY=0/SELL=1/HOLD=2.  Remap softmax column indices.

Features: 45 technical + 5 lag-difference (lag_diff_2/5/10/20/60).
  add_regime() is removed.  engine/features/feature_pipeline.py must produce these 50 columns.

Temperature scaling: checkpoint stores "temperature" (float).
  inference.py must apply:  probs = softmax(logits / T)  before reporting confidence.
"""

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "pyarrow", "loguru"],
    check=True,
)

gpu_info = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
    capture_output=True, text=True,
).stdout.strip()
print(f"GPU : {gpu_info}" if gpu_info else "No GPU — training will be slow")
print(f"Python {sys.version.split()[0]}")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

SEQUENCE_LEN     = 60
N_FEATURES       = 50
N_CLASSES        = 3
EPOCHS           = 100
BATCH_SIZE       = 128
LR               = 8e-4
WEIGHT_DECAY     = 1e-4
PATIENCE         = 20
LABEL_SMOOTH     = 0.05
WARMUP_EPOCHS    = 5
DAYS_WINDOW      = 730
DATA_DIR         = "/content/colab_data"
TRAIN_SYMBOLS    = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
TP_MULT          = 1.5
SL_MULT          = 1.0
HORIZON_BARS     = 10
MIXUP_PROB       = 0.3
MIXUP_ALPHA      = 0.4
USE_MIXUP        = True
TEMP_LBFGS_ITERS = 50
WARMUP_BARS      = 260
PURGE_GAP        = SEQUENCE_LEN

RAM_LIMIT_GB      = 4.5
BYTES_PER_ELEMENT = 4
MAX_ARRAY_BYTES   = int(RAM_LIMIT_GB * 1e9)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}")
if DEVICE.type == "cuda":
    props = torch.cuda.get_device_properties(0)
    print(f"         {props.name}  {props.total_memory/1e9:.1f} GB VRAM")
    torch.backends.cudnn.benchmark        = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.set_float32_matmul_precision("high")


def _ensure_data() -> None:
    already = (
        os.path.isdir(DATA_DIR)
        and any(f.endswith(".parquet") for f in os.listdir(DATA_DIR))
    )
    if already:
        print(f"Data already in {DATA_DIR} — skipping extraction.")
        return
    zip_path = "/content/colab_data.zip"
    if not os.path.exists(zip_path):
        try:
            from google.colab import files as _f
            print("Select colab_data.zip from your PC …")
            up = _f.upload()
            zip_path = next(iter(up))
        except Exception as exc:
            raise FileNotFoundError(
                "colab_data.zip not found. Upload it to the Files panel and re-run."
            ) from exc
    print(f"Extracting {zip_path} …")
    os.makedirs(DATA_DIR, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)
    print(f"Extracted to {DATA_DIR}")


_ensure_data()

_manifest_path = os.path.join(DATA_DIR, "manifest.json")
if os.path.exists(_manifest_path):
    _m = json.load(open(_manifest_path))
    print("Symbols in manifest:")
    for _sym, _info in _m.items():
        print(f"  {_sym}: {_info['rows']:,} bars  {_info['from'][:10]} → {_info['to'][:10]}")


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        s = x.mean(dim=[2, 3])
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s)).view(b, c, 1, 1)
        return x * s


class TemporalAttention(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.score(x).squeeze(-1), dim=1)
        return (w.unsqueeze(-1) * x).sum(dim=1)


class CNNLSTM(nn.Module):
    def __init__(self, n_classes: int = N_CLASSES, dropout: float = 0.4) -> None:
        super().__init__()
        self.act = nn.LeakyReLU(0.01)
        self.c1  = nn.Conv2d(1,  32, 3, padding=1)
        self.c2  = nn.Conv2d(32, 32, 3, padding=1)
        self.p1  = nn.MaxPool2d(2)
        self.se1 = SEBlock(32)
        self.c3  = nn.Conv2d(32, 64, 3, padding=1)
        self.c4  = nn.Conv2d(64, 64, 3, padding=1)
        self.p2  = nn.MaxPool2d(2)
        self.se2 = SEBlock(64)
        self.c5  = nn.Conv2d(64,  128, 3, padding=1)
        self.c6  = nn.Conv2d(128, 128, 3, padding=1)
        self.c7  = nn.Conv2d(128,  64, 1)
        self.se3 = SEBlock(64)
        self.lstm1     = nn.LSTM(64 * 12, 256, batch_first=True)
        self.drop_lstm = nn.Dropout(0.3)
        self.lstm2     = nn.LSTM(256, 256, batch_first=True)
        self.attn      = TemporalAttention(256)
        self.fc1     = nn.Linear(256, 64)
        self.drop_fc = nn.Dropout(dropout)
        self.fc2     = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.c1(x))
        x = self.act(self.c2(x))
        x = self.p1(x)
        x = self.se1(x)
        x = self.act(self.c3(x))
        x = self.act(self.c4(x))
        x = self.p2(x)
        x = self.se2(x)
        res = x
        x = self.act(self.c5(x))
        x = self.act(self.c6(x))
        x = self.act(self.c7(x))
        x = x + res
        x = self.se3(x)
        b, c, h, w = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b, h, c * w)
        x, _ = self.lstm1(x)
        x = self.drop_lstm(x)
        x, _ = self.lstm2(x)
        x = self.attn(x)
        x = F.relu(self.fc1(x))
        x = self.drop_fc(x)
        return self.fc2(x)


_probe = CNNLSTM().to(DEVICE)
_n_params = sum(p.numel() for p in _probe.parameters())
print(f"Model : {_n_params:,} parameters  [arch: CNN-LSTM-SE-TemporalAttn-Residual]")
del _probe
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()


FEATURE_COLUMNS = (
    "ret_1", "ret_5", "log_ret_1",
    "candle_body", "candle_upper_wick", "candle_lower_wick", "candle_range_pct",
    "sma_20", "ema_9", "ema_21", "ema_50", "ema_200",
    "ema_9_21_diff", "ema_21_50_diff", "ema_50_200_diff",
    "vwap", "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_pctb",
    "atr_14", "atr_pct",
    "adx_14", "dmp_14", "dmn_14",
    "stoch_k", "stoch_d",
    "cci_20", "willr_14",
    "obv", "obv_slope",
    "mfi_14", "roc_10",
    "donchian_upper", "donchian_lower", "donchian_pct",
    "kama_30", "psar",
    "volume_z", "hour_of_day",
    "lag_diff_2", "lag_diff_5", "lag_diff_10", "lag_diff_20", "lag_diff_60",
)
assert len(FEATURE_COLUMNS) == N_FEATURES


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> pd.Series:
    tr = pd.concat(
        [h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()


def _rsi(c: pd.Series, n: int = 14) -> pd.Series:
    d = c.diff()
    return 100 - 100 / (
        1 + d.clip(lower=0).rolling(n).mean()
        / (-d.clip(upper=0)).rolling(n).mean().replace(0, np.nan)
    )


def _macd(c: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9):
    line = _ema(c, fast) - _ema(c, slow)
    s    = _ema(line, sig)
    return line, s, line - s


def _bbands(c: pd.Series, n: int = 20, sd: float = 2.0):
    mid = c.rolling(n).mean()
    st  = c.rolling(n).std()
    return mid + sd * st, mid, mid - sd * st


def _adx(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14):
    tr = pd.concat(
        [h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1
    ).max(axis=1)
    up, dn = h.diff(), -l.diff()
    dmp = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=c.index)
    dmn = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=c.index)
    an  = tr.ewm(span=n, adjust=False).mean()
    dip = 100 * dmp.ewm(span=n, adjust=False).mean() / an.replace(0, np.nan)
    din = 100 * dmn.ewm(span=n, adjust=False).mean() / an.replace(0, np.nan)
    dx  = 100 * (dip - din).abs() / (dip + din).replace(0, np.nan)
    return dx.ewm(span=n, adjust=False).mean(), dip, din


def _stoch(h: pd.Series, l: pd.Series, c: pd.Series, k: int = 14, d: int = 3, sk: int = 3):
    lo, hi = l.rolling(k).min(), h.rolling(k).max()
    s = 100 * (c - lo) / (hi - lo).replace(0, np.nan)
    if sk > 1:
        s = s.rolling(sk).mean()
    return s, s.rolling(d).mean()


def _cci(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 20) -> pd.Series:
    tp  = (h + l + c) / 3
    dev = tp.rolling(n).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - tp.rolling(n).mean()) / (0.015 * dev.replace(0, np.nan))


def _mfi(h: pd.Series, l: pd.Series, c: pd.Series, v: pd.Series, n: int = 14) -> pd.Series:
    tp  = (h + l + c) / 3
    rmf = tp * v
    pos = rmf.where(tp > tp.shift(), 0.0)
    neg = rmf.where(tp < tp.shift(), 0.0)
    return 100 - 100 / (1 + pos.rolling(n).sum() / neg.rolling(n).sum().replace(0, np.nan))


def _kama(c: pd.Series, n: int = 30, fast: int = 2, slow: int = 30) -> pd.Series:
    fsc, ssc = 2.0 / (fast + 1), 2.0 / (slow + 1)
    vals = c.to_numpy(dtype=float, copy=True)
    out  = np.full_like(vals, np.nan)
    if len(vals) < n:
        return pd.Series(out, index=c.index)
    out[n - 1] = vals[n - 1]
    for i in range(n, len(vals)):
        direction  = abs(vals[i] - vals[i - n])
        volatility = np.sum(np.abs(np.diff(vals[i - n : i + 1])))
        er  = direction / volatility if volatility else 0.0
        sc  = (er * (fsc - ssc) + ssc) ** 2
        out[i] = out[i - 1] + sc * (vals[i] - out[i - 1])
    return pd.Series(out, index=c.index)


def _psar(h: pd.Series, l: pd.Series, step: float = 0.02, mx: float = 0.2) -> pd.Series:
    hv, lv = h.to_numpy(dtype=float), l.to_numpy(dtype=float)
    out    = np.full(len(hv), np.nan)
    if len(hv) < 2:
        return pd.Series(out, index=h.index)
    bull = True; iaf = step; ep = lv[0]; out[0] = lv[0]
    for i in range(1, len(hv)):
        prev = out[i - 1]
        if bull:
            out[i] = min(prev + iaf * (ep - prev), lv[i - 1], lv[i - 2] if i >= 2 else lv[i - 1])
            if lv[i] < out[i]:   bull = False; iaf = step; ep = hv[i]; out[i] = ep
            elif hv[i] > ep:     ep = hv[i]; iaf = min(iaf + step, mx)
        else:
            out[i] = max(prev + iaf * (ep - prev), hv[i - 1], hv[i - 2] if i >= 2 else hv[i - 1])
            if hv[i] > out[i]:   bull = True; iaf = step; ep = lv[i]; out[i] = ep
            elif lv[i] < ep:     ep = lv[i]; iaf = min(iaf + step, mx)
    return pd.Series(out, index=h.index)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    out = pd.DataFrame(index=df.index)
    rng = (h - l).replace(0, np.nan)

    out["ret_1"]             = c.pct_change()
    out["ret_5"]             = c.pct_change(5)
    out["log_ret_1"]         = np.log(c / c.shift(1))
    out["candle_body"]       = (c - o) / rng
    out["candle_upper_wick"] = (h - np.maximum(o, c)) / rng
    out["candle_lower_wick"] = (np.minimum(o, c) - l) / rng
    out["candle_range_pct"]  = rng / c

    out["sma_20"]          = c.rolling(20).mean()
    out["ema_9"]           = _ema(c, 9)
    out["ema_21"]          = _ema(c, 21)
    out["ema_50"]          = _ema(c, 50)
    out["ema_200"]         = _ema(c, 200)
    out["ema_9_21_diff"]   = (out["ema_9"]   - out["ema_21"])  / c
    out["ema_21_50_diff"]  = (out["ema_21"]  - out["ema_50"])  / c
    out["ema_50_200_diff"] = (out["ema_50"]  - out["ema_200"]) / c

    tp    = (h + l + c) / 3.0
    dates = (
        df.index.normalize()
        if isinstance(df.index, pd.DatetimeIndex)
        else pd.to_datetime(df.index).normalize()
    )
    out["vwap"] = (tp * v).groupby(dates).cumsum() / v.groupby(dates).cumsum().replace(0, np.nan)

    out["rsi_14"] = _rsi(c, 14)
    ml, ms, mh    = _macd(c)
    out["macd"] = ml; out["macd_signal"] = ms; out["macd_hist"] = mh

    bh, bm, bl   = _bbands(c)
    out["bb_upper"] = bh; out["bb_middle"] = bm; out["bb_lower"] = bl
    out["bb_width"] = (bh - bl) / bm.replace(0, np.nan)
    out["bb_pctb"]  = (c  - bl) / (bh - bl).replace(0, np.nan)

    out["atr_14"]  = _atr(h, l, c, 14)
    out["atr_pct"] = out["atr_14"] / c

    adx, dip, din = _adx(h, l, c, 14)
    out["adx_14"] = adx; out["dmp_14"] = dip; out["dmn_14"] = din

    sk, sd        = _stoch(h, l, c)
    out["stoch_k"] = sk; out["stoch_d"] = sd

    out["cci_20"]   = _cci(h, l, c, 20)
    out["willr_14"] = (
        -100
        * (h.rolling(14).max() - c)
        / (h.rolling(14).max() - l.rolling(14).min()).replace(0, np.nan)
    )

    out["obv"]       = (v * np.sign(c.diff().fillna(0))).cumsum()
    out["obv_slope"] = out["obv"].diff(5)
    out["mfi_14"]    = _mfi(h, l, c, v, 14)
    out["roc_10"]    = 100 * (c / c.shift(10) - 1)

    dlo = l.rolling(20).min(); dhi = h.rolling(20).max()
    out["donchian_lower"] = dlo; out["donchian_upper"] = dhi
    out["donchian_pct"]   = (c - dlo) / (dhi - dlo).replace(0, np.nan)

    out["kama_30"] = _kama(c, 30)
    out["psar"]    = _psar(h, l)

    vst             = v.rolling(50).std().replace(0, np.nan)
    out["volume_z"] = (v - v.rolling(50).mean()) / vst
    out["hour_of_day"] = (
        df.index.hour.astype(float) if isinstance(df.index, pd.DatetimeIndex) else 0.0
    )

    out["lag_diff_2"]  = c.diff(2)  / c.shift(2)
    out["lag_diff_5"]  = c.diff(5)  / c.shift(5)
    out["lag_diff_10"] = c.diff(10) / c.shift(10)
    out["lag_diff_20"] = c.diff(20) / c.shift(20)
    out["lag_diff_60"] = c.diff(60) / c.shift(60)

    return out[list(FEATURE_COLUMNS)]


def _triple_barrier_labels(
    close: np.ndarray,
    atr:   np.ndarray,
    tp_mult: float = TP_MULT,
    sl_mult: float = SL_MULT,
    horizon: int   = HORIZON_BARS,
) -> np.ndarray:
    n = len(close)
    labels = np.zeros(n, dtype=np.int64)
    for i in range(n - horizon):
        a = atr[i]
        if np.isnan(a) or a <= 0:
            continue
        tp = close[i] + tp_mult * a
        sl = close[i] - sl_mult * a
        for j in range(i + 1, i + horizon + 1):
            if close[j] >= tp:
                labels[i] = 1
                break
            if close[j] <= sl:
                labels[i] = 2
                break
    labels[-horizon:] = -1
    return labels


def _resample_m5(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    return (
        df.set_index("ts").sort_index()
        .resample("5min")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )


def _symbol_path(sym: str) -> str:
    return os.path.join(DATA_DIR, sym.replace("#", "_hash") + ".parquet")


def _count_windows(sym: str, days_window: int) -> int:
    fpath = _symbol_path(sym)
    if not os.path.exists(fpath):
        return 0
    raw = pd.read_parquet(fpath, columns=["ts", "close"])
    raw["ts"] = pd.to_datetime(raw["ts"])
    cutoff = raw["ts"].max() - timedelta(days=days_window)
    raw = raw[raw["ts"] >= cutoff]
    raw_idx = raw.set_index("ts").sort_index()
    m5 = raw_idx["close"].resample("5min").last().dropna()
    n  = len(m5)
    valid = max(0, n - max(WARMUP_BARS, SEQUENCE_LEN - 1) - HORIZON_BARS)
    del raw, raw_idx, m5
    gc.collect()
    return valid


def _build_symbol_data(sym: str, days_window: int) -> tuple[np.ndarray, np.ndarray]:
    fpath = _symbol_path(sym)
    raw = pd.read_parquet(fpath)
    raw["ts"] = pd.to_datetime(raw["ts"])
    cutoff = raw["ts"].max() - timedelta(days=days_window)
    raw = raw[raw["ts"] >= cutoff]
    m5  = _resample_m5(raw)
    del raw
    gc.collect()

    feats = compute_indicators(m5)
    feats = feats.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    fa    = feats.to_numpy(dtype=np.float32)
    close = m5["close"].to_numpy(dtype=np.float64)
    atr_v = feats["atr_14"].to_numpy(dtype=np.float64)
    del m5, feats
    gc.collect()

    lbl = _triple_barrier_labels(close, atr_v)
    del atr_v
    gc.collect()

    ref_end = min(WARMUP_BARS + SEQUENCE_LEN * 200, len(fa))
    ref = fa[WARMUP_BARS:ref_end] if ref_end > WARMUP_BARS else fa[WARMUP_BARS:]
    if len(ref) == 0:
        return np.empty((0, SEQUENCE_LEN, N_FEATURES), dtype=np.float32), np.empty(0, dtype=np.int64)
    mu  = ref.mean(axis=0, keepdims=True)
    sd  = ref.std(axis=0,  keepdims=True)
    sd  = np.where(sd < 1e-9, 1.0, sd)
    fn  = np.clip((fa - mu) / sd, -10.0, 10.0).astype(np.float32)
    del fa, ref
    gc.collect()

    start = max(WARMUP_BARS, SEQUENCE_LEN - 1)
    end   = len(fn) - HORIZON_BARS
    n     = end - start
    if n <= 0:
        return np.empty((0, SEQUENCE_LEN, N_FEATURES), dtype=np.float32), np.empty(0, dtype=np.int64)

    X = np.empty((n, SEQUENCE_LEN, N_FEATURES), dtype=np.float32)
    for k, i in enumerate(range(start, end)):
        X[k] = fn[i - SEQUENCE_LEN + 1 : i + 1, :N_FEATURES]
    y    = lbl[start:end].astype(np.int64)
    mask = y >= 0
    X    = X[mask]
    y    = y[mask]
    del fn, lbl, close
    gc.collect()
    return X, y


parquet_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".parquet"))
print(f"\nFound {len(parquet_files)} parquet files: {parquet_files}")

available_symbols = [s for s in TRAIN_SYMBOLS if os.path.exists(_symbol_path(s))]
if not available_symbols:
    raise RuntimeError("None of the configured TRAIN_SYMBOLS were found in colab_data/.")
print(f"Training on: {available_symbols}")

print("\nPhase 1: counting windows per symbol …")
counts = {}
for sym in available_symbols:
    n = _count_windows(sym, DAYS_WINDOW)
    counts[sym] = n
    print(f"  {sym}: {n:,} windows  (days={DAYS_WINDOW})")
total_windows = sum(counts.values())
est_bytes = total_windows * SEQUENCE_LEN * N_FEATURES * BYTES_PER_ELEMENT
print(f"Estimated X_all : {est_bytes/1e9:.2f} GB")

while est_bytes > MAX_ARRAY_BYTES and DAYS_WINDOW > 180:
    DAYS_WINDOW -= 90
    counts = {sym: _count_windows(sym, DAYS_WINDOW) for sym in available_symbols}
    total_windows = sum(counts.values())
    est_bytes = total_windows * SEQUENCE_LEN * N_FEATURES * BYTES_PER_ELEMENT
    print(f"  RAM guard: reduced DAYS_WINDOW to {DAYS_WINDOW}  →  {est_bytes/1e9:.2f} GB")

print(f"\nPhase 2: pre-allocating X_all ({total_windows:,}, {SEQUENCE_LEN}, {N_FEATURES}) …")
X_all = np.empty((total_windows, SEQUENCE_LEN, N_FEATURES), dtype=np.float32)
y_all = np.empty(total_windows, dtype=np.int64)
ram_used_gb = X_all.nbytes / 1e9
print(f"X_all RAM : {ram_used_gb:.2f} GB")
assert ram_used_gb < RAM_LIMIT_GB, f"RAM exceeded: {ram_used_gb:.2f} GB > {RAM_LIMIT_GB} GB"

print("\nPhase 3: building per-symbol windows and filling X_all …")
offset = 0
used_symbols: list[str] = []
for sym in available_symbols:
    if counts[sym] == 0:
        print(f"  {sym}: 0 windows — skipped")
        continue
    print(f"  {sym} → building …", end="", flush=True)
    X_sym, y_sym = _build_symbol_data(sym, DAYS_WINDOW)
    n = len(X_sym)
    X_all[offset:offset + n] = X_sym
    y_all[offset:offset + n] = y_sym
    offset += n
    used_symbols.append(sym)
    print(f"  done ({n:,}, classes={np.bincount(y_sym, minlength=3).tolist()})")
    del X_sym, y_sym
    gc.collect()

if offset < total_windows:
    X_all = X_all[:offset]
    y_all = y_all[:offset]

print(f"\nTotal : {len(X_all):,} windows   symbols : {used_symbols}")
print(f"Classes : {np.bincount(y_all, minlength=3).tolist()}  (0=HOLD 1=BUY 2=SELL)")

cut   = int(len(X_all) * 0.80)
purge = PURGE_GAP
X_tr  = X_all[: cut - purge].copy()
y_tr  = y_all[: cut - purge].copy()
X_va  = X_all[cut:].copy()
y_va  = y_all[cut:].copy()

feat_mean = X_all.reshape(-1, N_FEATURES).mean(axis=0)
feat_std  = X_all.reshape(-1, N_FEATURES).std(axis=0)

del X_all, y_all
gc.collect()
print(f"\nWalk-forward split — train: {len(X_tr):,}  val: {len(X_va):,}  (purge gap: {purge})")


def _to_tensor(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(a).unsqueeze(1)


train_dl = DataLoader(
    TensorDataset(_to_tensor(X_tr), torch.from_numpy(y_tr)),
    batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
    num_workers=2, pin_memory=(DEVICE.type == "cuda"),
)
val_dl = DataLoader(
    TensorDataset(_to_tensor(X_va), torch.from_numpy(y_va)),
    batch_size=BATCH_SIZE, shuffle=False,
    num_workers=2, pin_memory=(DEVICE.type == "cuda"),
)
print(f"Train batches : {len(train_dl)}   Val batches : {len(val_dl)}")

counts_arr = np.bincount(y_tr, minlength=N_CLASSES).astype(np.float64)
counts_arr[counts_arr == 0] = 1.0
weights = torch.tensor(counts_arr.sum() / (N_CLASSES * counts_arr), dtype=torch.float32).to(DEVICE)
print(f"Class weights : {[f'{w:.3f}' for w in weights.tolist()]}")


class Lookahead:
    def __init__(self, opt, k: int = 5, alpha: float = 0.5) -> None:
        self.opt = opt
        self.k = k
        self.alpha = alpha
        self._t = 0
        self.slow = [
            [p.clone().detach() for p in g["params"]] for g in opt.param_groups
        ]

    def step(self) -> None:
        self.opt.step()
        self._t += 1
        if self._t % self.k == 0:
            for g, slow in zip(self.opt.param_groups, self.slow):
                for p, s in zip(g["params"], slow):
                    s.data.add_(self.alpha * (p.data - s.data))
                    p.data.copy_(s.data)

    def zero_grad(self, **kw) -> None:
        self.opt.zero_grad(**kw)

    @property
    def param_groups(self):
        return self.opt.param_groups

    def state_dict(self):
        return self.opt.state_dict()

    def load_state_dict(self, d) -> None:
        self.opt.load_state_dict(d)


def _mcc(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = N_CLASSES) -> float:
    cm = np.zeros((n_classes, n_classes), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    t_sum = cm.sum(axis=1)
    p_sum = cm.sum(axis=0)
    total = float(cm.sum())
    if total == 0.0:
        return 0.0
    cov_yy = float(t_sum.sum() * total - (t_sum ** 2).sum())
    cov_xx = float(p_sum.sum() * total - (p_sum ** 2).sum())
    cov_xy = float(total * np.trace(cm) - (t_sum * p_sum).sum())
    denom  = (cov_yy * cov_xx) ** 0.5
    return float(cov_xy / denom) if denom > 0 else 0.0


def _dir_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = (y_true != 0)
    if mask.sum() == 0:
        return 0.0
    return float((y_true[mask] == y_pred[mask]).mean())


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


model = CNNLSTM().to(DEVICE)
try:
    if int(torch.__version__.split(".")[0]) >= 2:
        model = torch.compile(model, mode="reduce-overhead")
        print("torch.compile: enabled (reduce-overhead mode)")
except Exception as exc:
    print(f"torch.compile skipped: {exc}")

criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTH)
base_opt  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
optimizer = Lookahead(base_opt, k=5, alpha=0.5)

warmup_sched = torch.optim.lr_scheduler.LinearLR(
    base_opt, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS
)
cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
    base_opt, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6
)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    base_opt, schedulers=[warmup_sched, cosine_sched], milestones=[WARMUP_EPOCHS]
)
scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

best_composite  = -1.0
best_val_acc    = 0.0
best_mcc        = -1.0
best_dir_acc    = 0.0
best_state: dict | None = None
no_improve      = 0
history: list[dict] = []
t0 = time.time()

print(f"\nStarting training on {DEVICE} …")
print("=" * 80)

for ep in range(1, EPOCHS + 1):
    model.train()
    tl = tc = ts = 0
    for xb, yb in train_dl:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)
        base_opt.zero_grad(set_to_none=True)

        if USE_MIXUP and np.random.random() < MIXUP_PROB:
            lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
            if 0.1 < lam < 0.9:
                idx   = torch.randperm(xb.size(0), device=DEVICE)
                x_mix = lam * xb + (1.0 - lam) * xb[idx]
                with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                    lg   = model(x_mix)
                    loss = lam * criterion(lg, yb) + (1.0 - lam) * criterion(lg, yb[idx])
            else:
                with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                    lg   = model(xb)
                    loss = criterion(lg, yb)
        else:
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                lg   = model(xb)
                loss = criterion(lg, yb)

        scaler.scale(loss).backward()
        scaler.unscale_(base_opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(base_opt)
        scaler.update()
        optimizer._t += 1
        if optimizer._t % optimizer.k == 0:
            for g, slow in zip(optimizer.opt.param_groups, optimizer.slow):
                for p, s in zip(g["params"], slow):
                    s.data.add_(optimizer.alpha * (p.data - s.data))
                    p.data.copy_(s.data)

        tl += loss.item() * yb.size(0)
        tc += (lg.argmax(-1) == yb).sum().item()
        ts += yb.size(0)
    scheduler.step()

    model.eval()
    vl = vs = 0
    va_pred_list: list[np.ndarray] = []
    va_true_list: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in val_dl:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                lg   = model(xb)
                loss = criterion(lg, yb)
            vl += loss.item() * yb.size(0)
            vs += yb.size(0)
            va_pred_list.append(lg.argmax(-1).cpu().numpy())
            va_true_list.append(yb.cpu().numpy())

    va_preds  = np.concatenate(va_pred_list)
    va_true   = np.concatenate(va_true_list)
    ta        = tc / max(ts, 1)
    va        = float((va_preds == va_true).mean())
    mcc       = _mcc(va_true, va_preds)
    dir_acc   = _dir_acc(va_true, va_preds)
    composite = 0.40 * va + 0.40 * max(mcc, 0.0) + 0.20 * dir_acc

    row = {
        "epoch":   ep,
        "tr_acc":  float(ta),  "va_acc":  float(va),
        "tr_loss": tl / max(ts, 1),
        "va_loss": vl / max(vs, 1),
        "mcc":     float(mcc), "dir_acc": float(dir_acc),
        "comp":    float(composite),
    }
    history.append(row)

    if composite > best_composite:
        best_composite = composite
        best_val_acc   = va
        best_mcc       = mcc
        best_dir_acc   = dir_acc
        raw_model      = model._orig_mod if hasattr(model, "_orig_mod") else model
        best_state     = {k: v.detach().cpu().clone() for k, v in raw_model.state_dict().items()}
        no_improve     = 0
        star           = " ★"
    else:
        no_improve += 1
        star        = ""

    print(
        f"  ep {ep:3d}/{EPOCHS}  tr={ta:.4f}  va={va:.4f}  "
        f"MCC={mcc:+.3f}  dir={dir_acc:.3f}  comp={composite:.4f}  "
        f"loss={row['va_loss']:.4f}  {(time.time() - t0) / 60:.1f}m{star}"
    )

    if no_improve >= PATIENCE:
        print(f"  Early stop at epoch {ep}  (patience={PATIENCE})")
        break

print("=" * 80)
elapsed = (time.time() - t0) / 60


class _TempScale(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.T = nn.Parameter(torch.ones(1, device=DEVICE))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.T.clamp(min=0.1)


raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
if best_state is not None:
    raw_model.load_state_dict(best_state)
raw_model.eval()

ts_module = _TempScale()
ts_opt    = torch.optim.LBFGS([ts_module.T], lr=0.01, max_iter=TEMP_LBFGS_ITERS)
ts_crit   = nn.CrossEntropyLoss()

all_logits: list[torch.Tensor] = []
all_labels: list[torch.Tensor] = []
with torch.no_grad():
    for xb, yb in val_dl:
        xb = xb.to(DEVICE, non_blocking=True)
        lg = raw_model(xb)
        all_logits.append(lg.float().detach())
        all_labels.append(yb.to(DEVICE))
logits_val = torch.cat(all_logits)
labels_val = torch.cat(all_labels)
del all_logits, all_labels
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()

def _ts_closure():
    ts_opt.zero_grad()
    loss = ts_crit(ts_module(logits_val), labels_val)
    loss.backward()
    return loss

ts_opt.step(_ts_closure)
temperature = float(ts_module.T.detach().cpu().item())

final_preds: list[np.ndarray] = []
final_true:  list[np.ndarray] = []
with torch.no_grad():
    for xb, yb in val_dl:
        xb = xb.to(DEVICE, non_blocking=True)
        lg = raw_model(xb)
        final_preds.append(lg.argmax(-1).cpu().numpy())
        final_true.append(yb.numpy())
final_pred_arr = np.concatenate(final_preds)
final_true_arr = np.concatenate(final_true)
conf_matrix    = _confusion_matrix(final_true_arr, final_pred_arr)

print(f"Best val accuracy  : {best_val_acc:.4f}")
print(f"Best MCC           : {best_mcc:+.3f}")
print(f"Best directional   : {best_dir_acc:.4f}  (BUY+SELL only)")
print(f"Best composite     : {best_composite:.4f}")
print(f"Temperature T      : {temperature:.4f}")
print(f"Days window used   : {DAYS_WINDOW}")
print(f"Total time         : {elapsed:.1f} min")
print(f"\nConfusion Matrix (rows=true, cols=pred, order: HOLD/BUY/SELL):")
print(conf_matrix)

del logits_val, labels_val
gc.collect()
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()


version   = f"v_colab_{datetime.now().strftime('%Y%m%d_%H%M')}"
ckpt_name = f"cnn_lstm_{version}.pt"
ckpt_path = f"/content/{ckpt_name}"

torch.save(
    {
        "model_state":           best_state,
        "version":                version,
        "tier":                   "production",
        "architecture":           "CNN-LSTM-SE-TemporalAttn-Residual",
        "symbols":                used_symbols,
        "feature_mean":           feat_mean.tolist(),
        "feature_std":            feat_std.tolist(),
        "best_val_acc":           best_val_acc,
        "best_mcc":               best_mcc,
        "best_dir_acc":           best_dir_acc,
        "best_composite":         best_composite,
        "temperature":            temperature,
        "label_method":           "triple_barrier_atr_adaptive",
        "label_encoding":         {"HOLD": 0, "BUY": 1, "SELL": 2},
        "triple_barrier_config":  {"tp_mult": TP_MULT, "sl_mult": SL_MULT, "horizon": HORIZON_BARS},
        "sequence_len":           SEQUENCE_LEN,
        "n_features":             N_FEATURES,
        "feature_columns":        list(FEATURE_COLUMNS),
        "days_window_used":       DAYS_WINDOW,
        "history":                history,
        "confusion_matrix":       conf_matrix.tolist(),
        "training_config": {
            "epochs":       EPOCHS,
            "batch_size":   BATCH_SIZE,
            "lr":           LR,
            "weight_decay": WEIGHT_DECAY,
            "optimizer":    "Lookahead(AdamW)",
            "scheduler":    "LinearWarmup+CosineAnnealing",
            "augmentation": f"mixup_p{MIXUP_PROB}_alpha{MIXUP_ALPHA}",
            "label_smooth": LABEL_SMOOTH,
            "warmup":       WARMUP_EPOCHS,
        },
        "MIGRATION_NOTE": MIGRATION_NOTE,
    },
    ckpt_path,
)
print(f"\nCheckpoint saved : {ckpt_path}")
print("After downloading, place in :")
print(r"  C:\Users\leade\Desktop\FXbot\engine\models\checkpoints" + "\\")
print("The engine auto-loads the newest .pt by modification time on next start.")

try:
    from google.colab import files as _colab_files
    _colab_files.download(ckpt_path)
    print("Download started — check your browser's download bar.")
except Exception:
    print("Auto-download unavailable — right-click the .pt file in the Files panel and choose Download.")


try:
    import matplotlib.pyplot as plt
    eps = [r["epoch"] for r in history]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    axes[0].plot(eps, [r["tr_acc"]  for r in history], label="Train")
    axes[0].plot(eps, [r["va_acc"]  for r in history], label="Val")
    axes[0].axhline(0.52, color="red", linestyle="--", label="0.52 target")
    axes[0].set_title("Accuracy"); axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(eps, [r["mcc"]     for r in history], label="MCC",       color="purple")
    axes[1].plot(eps, [r["dir_acc"] for r in history], label="Dir acc",   color="green")
    axes[1].plot(eps, [r["comp"]    for r in history], label="Composite", color="orange")
    axes[1].set_title("MCC / Dir-Acc / Composite"); axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(True)
    axes[2].plot(eps, [r["tr_loss"] for r in history], label="Train")
    axes[2].plot(eps, [r["va_loss"] for r in history], label="Val")
    axes[2].set_title("Loss"); axes[2].set_xlabel("Epoch"); axes[2].legend(); axes[2].grid(True)
    plt.tight_layout()
    plt.savefig("/content/training_curve.png", dpi=150)
    plt.show()
    print("Plot saved : /content/training_curve.png")
except Exception as exc:
    print(f"Plot skipped: {exc}")

print("\nDone.")
