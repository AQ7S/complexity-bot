"""Generate the Colab training notebook (FXbot_Train_Colab.ipynb).

Run once:  python engine/models/generate_colab_notebook.py
Then open the .ipynb in Google Colab.
"""
import json, pathlib, textwrap

def md(src): return {"cell_type":"markdown","metadata":{},"source":textwrap.dedent(src).strip().splitlines(True)}
def code(src): return {"cell_type":"code","metadata":{},"source":textwrap.dedent(src).strip().splitlines(True),"outputs":[],"execution_count":None}

cells = [
# ── Cell 1: Title ──
md("""
# 🚀 FXbot CNN-LSTM Training — Google Colab (GPU)

**Workflow:**
1. Run the install cell
2. Upload your CSV(s) exported by `export_for_colab.py`
3. Hit **Runtime → Run all**
4. Download the `.pt` checkpoint when done
"""),

# ── Cell 2: GPU check + installs ──
code("""
# ── 1. Verify GPU & install deps ──
import subprocess, sys
!nvidia-smi
subprocess.check_call([sys.executable, "-m", "pip", "-q", "install",
    "pandas_ta_classic==0.5.44", "loguru==0.7.2"])
print("✅ Dependencies installed")
"""),

# ── Cell 3: Upload CSV ──
code("""
# ── 2. Upload your CSV file(s) ──
from google.colab import files
uploaded = files.upload()
csv_names = [k for k in uploaded.keys() if k.endswith('.csv')]
print(f"📁 Uploaded: {csv_names}")
"""),

# ── Cell 4: All-in-one source code ──
code('''
# ── 3. Self-contained model + feature code ──
# (Bundled so Colab needs zero local imports)

import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger
import pandas_ta_classic as ta
import time, json as _json
from datetime import datetime
from pathlib import Path

# ─── Device ───
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Training device: {device}")

# ─── Constants ───
CLASSES = ("BUY", "SELL", "HOLD")
N_CLASSES = 3
SEQUENCE_LEN = 60
N_FEATURES = 50

FEATURE_COLUMNS = (
    "ret_1","ret_5","log_ret_1",
    "candle_body","candle_upper_wick","candle_lower_wick","candle_range_pct",
    "sma_20","ema_9","ema_21","ema_50","ema_200",
    "ema_9_21_diff","ema_21_50_diff","ema_50_200_diff",
    "vwap","rsi_14","macd","macd_signal","macd_hist",
    "bb_upper","bb_middle","bb_lower","bb_width","bb_pctb",
    "atr_14","atr_pct","adx_14","dmp_14","dmn_14",
    "stoch_k","stoch_d","cci_20","willr_14",
    "obv","obv_slope","mfi_14","roc_10",
    "donchian_upper","donchian_lower","donchian_pct",
    "kama_30","psar","volume_z","hour_of_day",
    "regime_trending_up","regime_trending_down","regime_ranging","regime_high_volatility",
    "kill_zone_flag",
)

# ─── Indicators (45 cols) ───
def compute_all(df):
    o,h,l,c,v = df["open"],df["high"],df["low"],df["close"],df["volume"]
    out = pd.DataFrame(index=df.index)
    out["ret_1"]=c.pct_change(); out["ret_5"]=c.pct_change(5)
    out["log_ret_1"]=np.log(c/c.shift(1))
    body=c-o; rng=(h-l).replace(0,np.nan)
    out["candle_body"]=body/rng; out["candle_upper_wick"]=(h-np.maximum(o,c))/rng
    out["candle_lower_wick"]=(np.minimum(o,c)-l)/rng; out["candle_range_pct"]=rng/c
    out["sma_20"]=ta.sma(c,length=20)
    out["ema_9"]=ta.ema(c,length=9); out["ema_21"]=ta.ema(c,length=21)
    out["ema_50"]=ta.ema(c,length=50); out["ema_200"]=ta.ema(c,length=200)
    out["ema_9_21_diff"]=(out["ema_9"]-out["ema_21"])/c
    out["ema_21_50_diff"]=(out["ema_21"]-out["ema_50"])/c
    out["ema_50_200_diff"]=(out["ema_50"]-out["ema_200"])/c
    try: out["vwap"]=ta.vwap(h,l,c,v)
    except: tp=(h+l+c)/3.0; out["vwap"]=(tp*v).cumsum()/v.cumsum().replace(0,np.nan)
    out["rsi_14"]=ta.rsi(c,length=14)
    macd=ta.macd(c,fast=12,slow=26,signal=9)
    if macd is not None and not macd.empty:
        out["macd"]=macd.iloc[:,0]; out["macd_hist"]=macd.iloc[:,1]; out["macd_signal"]=macd.iloc[:,2]
    else: out["macd"]=out["macd_hist"]=out["macd_signal"]=np.nan
    bb=ta.bbands(c,length=20,std=2.0)
    if bb is not None and not bb.empty:
        out["bb_lower"]=bb.iloc[:,0]; out["bb_middle"]=bb.iloc[:,1]; out["bb_upper"]=bb.iloc[:,2]
        out["bb_width"]=(out["bb_upper"]-out["bb_lower"])/out["bb_middle"]
        out["bb_pctb"]=(c-out["bb_lower"])/(out["bb_upper"]-out["bb_lower"])
    else:
        for k in ("bb_lower","bb_middle","bb_upper","bb_width","bb_pctb"): out[k]=np.nan
    out["atr_14"]=ta.atr(h,l,c,length=14); out["atr_pct"]=out["atr_14"]/c
    adx=ta.adx(h,l,c,length=14)
    if adx is not None and not adx.empty:
        out["adx_14"]=adx.iloc[:,0]; out["dmp_14"]=adx.iloc[:,1]; out["dmn_14"]=adx.iloc[:,2]
    else: out["adx_14"]=out["dmp_14"]=out["dmn_14"]=np.nan
    st=ta.stoch(h,l,c,k=14,d=3,smooth_k=3)
    if st is not None and not st.empty: out["stoch_k"]=st.iloc[:,0]; out["stoch_d"]=st.iloc[:,1]
    else: out["stoch_k"]=out["stoch_d"]=np.nan
    out["cci_20"]=ta.cci(h,l,c,length=20); out["willr_14"]=ta.willr(h,l,c,length=14)
    out["obv"]=ta.obv(c,v); out["obv_slope"]=out["obv"].diff(5)
    out["mfi_14"]=ta.mfi(h,l,c,v,length=14); out["roc_10"]=ta.roc(c,length=10)
    don=ta.donchian(h,l,lower_length=20,upper_length=20)
    if don is not None and not don.empty:
        out["donchian_lower"]=don.iloc[:,0]; out["donchian_upper"]=don.iloc[:,2]
        out["donchian_pct"]=(c-out["donchian_lower"])/((out["donchian_upper"]-out["donchian_lower"]).replace(0,np.nan))
    else: out["donchian_lower"]=out["donchian_upper"]=out["donchian_pct"]=np.nan
    out["kama_30"]=ta.kama(c,length=30)
    psar=ta.psar(h,l,c)
    if psar is not None and not psar.empty:
        lc=next((col for col in psar.columns if col.startswith("PSARl_")),None)
        sc=next((col for col in psar.columns if col.startswith("PSARs_")),None)
        lv=psar[lc] if lc else pd.Series(np.nan,index=psar.index)
        sv=psar[sc] if sc else pd.Series(np.nan,index=psar.index)
        out["psar"]=lv.fillna(sv)
    else: out["psar"]=np.nan
    vol_mean=v.rolling(50).mean(); vol_std=v.rolling(50).std().replace(0,np.nan)
    out["volume_z"]=(v-vol_mean)/vol_std
    out["hour_of_day"]=df.index.hour.astype(float) if isinstance(df.index,pd.DatetimeIndex) else 0.0
    return out[list(FEATURE_COLUMNS[:45])]

# ─── Regime (vectorized) ───
def vectorized_regime(feats):
    adx=feats["adx_14"]; atr_pct=feats["atr_pct"]; ema21=feats["ema_21"]; ema50=feats["ema_50"]
    atr_pctl=atr_pct.rolling(200,min_periods=20).rank(pct=True)
    is_high_vol=(atr_pctl>=0.85)&(adx<25.0); is_trend=adx>=25.0
    is_up=is_trend&(ema21>ema50); is_down=is_trend&(ema21<=ema50); is_range=~(is_high_vol|is_trend)
    return pd.DataFrame({"regime_trending_up":is_up.astype(float),"regime_trending_down":is_down.astype(float),
        "regime_ranging":is_range.astype(float),"regime_high_volatility":is_high_vol.astype(float)},index=feats.index).fillna(0.0)

# ─── Feature frame builder ───
def build_feature_frame(bars):
    feats=compute_all(bars); regime_oh=vectorized_regime(feats)
    feats=pd.concat([feats,regime_oh],axis=1); feats["kill_zone_flag"]=0.0
    return feats[list(FEATURE_COLUMNS)]

# ─── Labels ───
def make_labels(close, threshold_bps=1.0):
    next_ret=close.shift(-1)/close-1.0; th=threshold_bps/10_000.0
    labels=np.full(len(close),2,dtype=np.int64)
    labels[next_ret.values>th]=0; labels[next_ret.values<-th]=1; labels[-1]=-1
    return labels

# ─── Windows ───
def build_windows(bars, sequence_len=60, warmup=200, label_threshold_bps=1.0):
    feats=build_feature_frame(bars)
    feats=feats.replace([np.inf,-np.inf],np.nan).ffill().bfill().fillna(0.0)
    feat_arr=feats.to_numpy(dtype=np.float32,copy=False)
    labels=make_labels(bars["close"],threshold_bps=label_threshold_bps)
    mu=feat_arr[warmup:warmup+sequence_len*200].mean(axis=0,keepdims=True)
    sd=feat_arr[warmup:warmup+sequence_len*200].std(axis=0,keepdims=True)
    sd=np.where(sd<1e-9,1.0,sd)
    feat_norm=(feat_arr-mu)/sd; feat_norm=np.clip(feat_norm,-10.0,10.0).astype(np.float32,copy=False)
    start=max(warmup,sequence_len-1); end=len(feat_norm)-1; n=end-start
    if n<=0: raise ValueError(f"not enough bars: T={len(feat_norm)}")
    X=np.empty((n,sequence_len,N_FEATURES),dtype=np.float32)
    for k,i in enumerate(range(start,end)): X[k]=feat_norm[i-sequence_len+1:i+1]
    y=labels[start:end].astype(np.int64); mask=y>=0
    return X[mask],y[mask]

# ─── Model ───
class _LSTMCellLoop(nn.Module):
    def __init__(self,input_size,hidden_size):
        super().__init__(); self.hidden_size=hidden_size
        self.W_x=nn.Linear(input_size,4*hidden_size,bias=True)
        self.W_h=nn.Linear(hidden_size,4*hidden_size,bias=False)
    def forward(self,x):
        b,t,_=x.shape
        h=torch.zeros(b,self.hidden_size,device=x.device,dtype=x.dtype)
        c=torch.zeros_like(h); outputs=[]
        for step in range(t):
            gates=self.W_x(x[:,step,:])+self.W_h(h); i,f,g,o=gates.chunk(4,dim=1)
            i=torch.sigmoid(i); f=torch.sigmoid(f); g=torch.tanh(g); o=torch.sigmoid(o)
            c=f*c+i*g; h=o*torch.tanh(c); outputs.append(h)
        return torch.stack(outputs,dim=1),(h,c)

class CNNLSTM(nn.Module):
    def __init__(self,n_classes=3,dropout=0.3):
        super().__init__(); self.act=nn.LeakyReLU(0.01)
        self.c1=nn.Conv2d(1,32,3,padding=1); self.c2=nn.Conv2d(32,32,3,padding=1); self.p1=nn.MaxPool2d(2)
        self.c3=nn.Conv2d(32,64,3,padding=1); self.c4=nn.Conv2d(64,64,3,padding=1); self.p2=nn.MaxPool2d(2)
        self.c5=nn.Conv2d(64,128,3,padding=1); self.c6=nn.Conv2d(128,128,3,padding=1); self.c7=nn.Conv2d(128,64,1)
        self.lstm1=_LSTMCellLoop(64*12,256); self.dropout_lstm=nn.Dropout(0.2)
        self.lstm2=_LSTMCellLoop(256,256)
        self.fc1=nn.Linear(256,64); self.dropout_fc=nn.Dropout(dropout); self.fc2=nn.Linear(64,n_classes)
    def forward(self,x):
        x=self.act(self.c1(x)); x=self.act(self.c2(x)); x=self.p1(x)
        x=self.act(self.c3(x)); x=self.act(self.c4(x)); x=self.p2(x)
        x=self.act(self.c5(x)); x=self.act(self.c6(x)); x=self.act(self.c7(x))
        b,c,h,w=x.shape; x=x.permute(0,2,1,3).reshape(b,h,c*w)
        x,_=self.lstm1(x); x=self.dropout_lstm(x); x,_=self.lstm2(x); x=x[:,-1,:]
        x=F.relu(self.fc1(x)); x=self.dropout_fc(x); return self.fc2(x)

print("✅ Model + feature pipeline defined")
'''),

# ── Cell 5: Load CSV + resample ──
code("""
# ── 4. Load CSV data & resample M1 → M5 ──
all_X, all_y = [], []
for csv_name in csv_names:
    print(f"\\n📊 Processing {csv_name}...")
    df = pd.read_csv(csv_name, parse_dates=["ts"]).set_index("ts").sort_index()
    # Resample M1 → M5
    bars = df.resample("5min").agg(
        {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    ).dropna()
    print(f"  {len(bars)} M5 bars from {bars.index[0]} → {bars.index[-1]}")
    X, y = build_windows(bars)
    print(f"  {len(y)} training windows; class counts={np.bincount(y, minlength=3).tolist()}")
    all_X.append(X); all_y.append(y)

X = np.concatenate(all_X); y = np.concatenate(all_y)
feat_mean = X.reshape(-1, X.shape[-1]).mean(axis=0)
feat_std  = X.reshape(-1, X.shape[-1]).std(axis=0)
print(f"\\n✅ Total: {len(y)} windows, shape={X.shape}")
"""),

# ── Cell 6: Training loop ──
code("""
# ── 5. Train on GPU! ──
# Config — edit these for build vs production
TIER = "build"  # "build" or "production"
TIER_CFG = {
    "build":      {"epochs":5,  "batch":64, "lr":1e-4, "patience":3,  "min_val_acc":0.45},
    "production": {"epochs":50, "batch":64, "lr":1e-4, "patience":8,  "min_val_acc":0.52},
}
cfg = TIER_CFG[TIER]
epochs, batch_size, lr = cfg["epochs"], cfg["batch"], cfg["lr"]
patience = cfg["patience"]

# 80/20 chronological split
cut = int(len(X) * 0.8)
X_tr, X_va = X[:cut], X[cut:]
y_tr, y_va = y[:cut], y[cut:]

to_t = lambda a: torch.from_numpy(a).unsqueeze(1)
train_ds = TensorDataset(to_t(X_tr), torch.from_numpy(y_tr))
val_ds   = TensorDataset(to_t(X_va), torch.from_numpy(y_va))
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

model = CNNLSTM().to(device)
counts = np.bincount(y_tr, minlength=3).astype(np.float64)
counts[counts==0] = 1.0
weights = torch.tensor(counts.sum()/(3*counts), dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

best_val_acc, best_state, epochs_no_improve = 0.0, None, 0
history = []
t0 = time.time()

for ep in range(1, epochs+1):
    model.train(); tr_loss=tr_correct=tr_seen=0
    for xb,yb in train_loader:
        xb,yb = xb.to(device),yb.to(device)
        optimizer.zero_grad(); logits=model(xb); loss=criterion(logits,yb)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
        tr_loss+=loss.item()*yb.size(0); tr_correct+=(logits.argmax(-1)==yb).sum().item(); tr_seen+=yb.size(0)
    scheduler.step()

    model.eval(); va_loss=va_correct=va_seen=0
    with torch.no_grad():
        for xb,yb in val_loader:
            xb,yb=xb.to(device),yb.to(device); logits=model(xb); loss=criterion(logits,yb)
            va_loss+=loss.item()*yb.size(0); va_correct+=(logits.argmax(-1)==yb).sum().item(); va_seen+=yb.size(0)

    tr_acc=tr_correct/max(tr_seen,1); va_acc=va_correct/max(va_seen,1)
    elapsed=time.time()-t0
    print(f"  epoch {ep}/{epochs}  tr_loss={tr_loss/max(tr_seen,1):.4f} tr_acc={tr_acc:.4f}  "
          f"va_loss={va_loss/max(va_seen,1):.4f} va_acc={va_acc:.4f}  ({elapsed:.1f}s)")
    history.append({"epoch":ep,"tr_acc":tr_acc,"va_acc":va_acc})

    if va_acc > best_val_acc:
        best_val_acc=va_acc; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; epochs_no_improve=0
    else:
        epochs_no_improve+=1
        if epochs_no_improve>=patience: print(f"  ⏹ Early stop at epoch {ep}"); break

if best_state is None: best_state=model.state_dict()
print(f"\\n🏆 Best val accuracy: {best_val_acc:.4f} (elapsed {time.time()-t0:.1f}s)")
"""),

# ── Cell 7: Save & download ──
code("""
# ── 6. Save checkpoint & download ──
version = f"v{int(time.time())}_{datetime.utcnow().strftime('%Y%m%d')}"
ckpt_name = f"cnn_lstm_{version}.pt"
symbols = [n.split("_M1_")[0] for n in csv_names]

torch.save({
    "model_state": best_state,
    "version": version,
    "tier": TIER,
    "symbols": symbols,
    "feature_mean": feat_mean.tolist(),
    "feature_std": feat_std.tolist(),
    "best_val_acc": best_val_acc,
    "history": history,
}, ckpt_name)

print(f"✅ Checkpoint saved: {ckpt_name}")
print(f"📥 Downloading to your PC...")
files.download(ckpt_name)
print("\\n👉 Place the downloaded .pt file in: engine/models/checkpoints/")
"""),
]

nb = {
    "nbformat": 4, "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance":[], "gpuType":"T4"},
        "kernelspec": {"name":"python3","display_name":"Python 3"},
        "accelerator": "GPU"
    },
    "cells": cells,
}

out = pathlib.Path(__file__).resolve().parent / "FXbot_Train_Colab.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"[OK] Notebook generated: {out}")
print("Upload this file to https://colab.research.google.com")
