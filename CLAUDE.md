# Complexity Engine — Project Memory

## What This Is
Local Electron + Python AI trading system. XM MT5 demo account. 13 symbols.
Engine = background Python asyncio service. UI = Electron + React 18 + Vite + TypeScript.
IPC bridge: asyncio WebSocket server on ws://localhost:8765 (Python) ↔ Electron client.

## Stack
- Electron + React 18 + Vite + TypeScript + TailwindCSS + shadcn/ui + Framer Motion
- Python 3.11: MetaTrader5, aiomql, pytorch, pandas-ta-classic, TA-Lib, smartmoneyconcepts, fredapi
- DuckDB: tick + OHLCV history  |  SQLite: trade journal + audit log + structured event log
- Supabase: remote sync  |  TradingView Lightweight Charts v4: all candlestick rendering

## Symbols (exact XM strings — never change)
EURUSD#, USDJPY#, GBPUSD#, USDCHF#, GOLD#, BTCUSD#, ETHUSD#,
AI_INDX#, Crypto_10#, TrumpWinners#, HarrisWinners#, EURJPY#, AUDUSD#

## Design System (enforce everywhere)
### Colors (CSS vars in ui/src/index.css)
--void:#050810  --panel:#080d1a  --surface:#0b1220  --card:#0f1827
--card-hover:#131e30  --elevated:#172035  --border:#1c2d45  --border-sub:#111d32
--cyan:#00d4ff  --cyan-dim:#00a8cc  --purple:#7c3aed  --purple-dim:#5b2db0
--green:#10b981  --green-dim:#0a8f63  --amber:#f59e0b  --amber-dim:#c27d08
--red:#ef4444  --red-dim:#c03030  --gold:#d4a017
--text:#e2e8f0  --text-2:#94a3b8  --text-3:#4a5568

### Typography (loaded via Google Fonts @import)
- Prices/numbers/metrics: JetBrains Mono
- UI labels/buttons/headers: Space Grotesk
- Body text: DM Sans
- App wordmark + hero stat ONLY: Orbitron

### CSS animation classes
.price-up / .price-down (tick flash)
.signal-buy / .signal-sell (pulse glow)
.live-dot (blink), .font-mono/.font-ui/.font-body/.font-hero

## Immutable Rules
- ZERO inline comments in any file
- ZERO placeholders, stubs, TODOs
- Never change IPC message keys without updating both sides
- Never change CNN input tensor shape (batch, 1, 60, 50)
- Never change checkpoint key "model_state"
- Max risk per trade: 2% — hardcoded
- All secrets in .env only
- forex-calendar and jblanked keys: guard every call with `if key not in (None, "", "unset")`
- No print() calls — use loguru logger everywhere

## Known State
- CNN-LSTM val accuracy: 0.4309 (below threshold — do NOT enable live execution)
- MT5: login 168514183, server XMGlobal-MT5 2
- Discord webhook + Anthropic API + Supabase: all configured
- forex-calendar + jblanked: unset

## Real Capability State (update after every major test)
- CNN accuracy (truly OOS, walk-forward split with purge gap): 0.4309 — BELOW THRESHOLD
- Triple-barrier labels applied: NO — pending next Colab run
- Shadow mode: ACTIVE (default ON via SHADOW_MODE env, gate is `engine.execution.order_router.send_order`)
- ECE last measured: NOT YET (need ≥50 closed shadow trades; recomputed every 50 thereafter)
- Backtest Sharpe with costs (EURUSD M5 2023-01-01 → 2024-12-31, simple 3-vote EMA+RSI baseline):
  -0.298 — strategy edge insufficient to deploy live
  Costs subtracted: spread 1.20 pips, slippage 0.50 pips, swap long −0.30 / short +0.10 pips/night
- Live execution: DISABLED — do not enable until shadow WR > 50% over 100+ trades AND
  shadow Sharpe > current_model_sharpe × 1.10
- Last honest OOS validation date: 2026-05-17 (initial backtest run via `engine.strategy.backtest`)

## New modules (v1.0.9+)
- engine/execution/vwap_slicer.py — 3-slice VWAP entry
- engine/features/order_flow.py — compute_ofi + ofi_vote
- engine/strategy/po3_detector.py — Power-of-Three sweep+reclaim
- engine/features/supplementary.py — TA-Lib candle patterns + Supertrend/Squeeze/Hull/Fisher/QQE
- engine/news/macro_data.py — FRED yield curve + crypto fear/greed
- engine/watchdog.py — auto-restart supervisor with /health ping
- engine/data/event_log.py — structured DuckDB event log
- engine/models/ewc.py — Elastic Weight Consolidation regularizer
- engine/models/replay_buffer.py — experience replay ring (10k trades)

## Retrain (verifiable end-to-end)
- UI button (`/ai`) → `cmd_manual_retrain` → engine spawns worker via multiprocessing
- Engine broadcasts `model_update` (`retrain_starting_v{N}` then final version) + Notification
- AI Engine card flips to purple TRAINING pulse, version visible via `data-testid="model-version-{name}"`

## Test Commands
pnpm run dev (UI port 5173)  |  pnpm run build (prod)  |  pnpm run dist (.exe)
python engine.py             |  pytest engine/tests
