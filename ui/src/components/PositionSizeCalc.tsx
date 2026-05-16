import { useState } from 'react';
import { X, ChevronDown, ChevronUp } from 'lucide-react';
import { useEngineStore } from '@/store/engineStore';
import { SYMBOLS_13 } from '@/lib/constants';

type Preset = { label: string; riskPct: number; symbol: string; tickSize: number; tickValue: number; step: number };

const PRESETS: Preset[] = [
  { label: 'EURUSD 2%', riskPct: 2,   symbol: 'EURUSD', tickSize: 0.00001, tickValue: 1.0,   step: 0.01 },
  { label: 'XAUUSD 1%', riskPct: 1,   symbol: 'XAUUSD', tickSize: 0.01,    tickValue: 1.0,   step: 0.01 },
  { label: 'USDJPY 2%', riskPct: 2,   symbol: 'USDJPY', tickSize: 0.001,   tickValue: 0.91,  step: 0.01 },
  { label: 'Scalp 0.5%', riskPct: 0.5, symbol: 'EURUSD', tickSize: 0.00001, tickValue: 1.0,  step: 0.01 },
];

const SYMBOL_DEFAULTS: Record<string, { tickSize: number; tickValue: number; step: number }> = {
  EURUSD:         { tickSize: 0.00001, tickValue: 1.0,  step: 0.01 },
  USDJPY:         { tickSize: 0.001,   tickValue: 0.91, step: 0.01 },
  GBPUSD:         { tickSize: 0.00001, tickValue: 1.0,  step: 0.01 },
  USDCHF:         { tickSize: 0.00001, tickValue: 1.0,  step: 0.01 },
  'EURUSD#':      { tickSize: 0.00001, tickValue: 1.0,  step: 0.01 },
  'USDJPY#':      { tickSize: 0.001,   tickValue: 0.91, step: 0.01 },
  XAUUSD:         { tickSize: 0.01,    tickValue: 1.0,  step: 0.01 },
  'BTCUSD#':      { tickSize: 0.01,    tickValue: 1.0,  step: 0.01 },
  'ETHUSD#':      { tickSize: 0.01,    tickValue: 1.0,  step: 0.01 },
  'AI_INDX#':     { tickSize: 0.01,    tickValue: 1.0,  step: 0.01 },
  'Crypto_10#':   { tickSize: 0.01,    tickValue: 1.0,  step: 0.01 },
  'TrumpWinners#':{ tickSize: 0.01,    tickValue: 1.0,  step: 0.01 },
  'HarrisWinners#':{ tickSize: 0.01,   tickValue: 1.0,  step: 0.01 },
};

export default function PositionSizeCalc() {
  const account = useEngineStore((s) => s.account);
  const ticks   = useEngineStore((s) => s.ticks);

  const equity  = account?.equity  ?? 10_000;
  const balance = account?.balance ?? 10_000;

  const [open, setOpen]         = useState(false);
  const [symbol, setSymbol]     = useState('EURUSD');
  const [riskPct, setRiskPct]   = useState(2.0);
  const [entry, setEntry]       = useState(1.07300);
  const [sl, setSl]             = useState(1.07200);
  const [tickSize, setTickSize] = useState(0.00001);
  const [tickValue, setTickValue] = useState(1.0);
  const [step, setStep]         = useState(0.01);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Live-fill entry from ticks when symbol changes
  const applySymbol = (sym: string) => {
    setSymbol(sym);
    const d = SYMBOL_DEFAULTS[sym];
    if (d) { setTickSize(d.tickSize); setTickValue(d.tickValue); setStep(d.step); }
    const tick = ticks[sym];
    if (tick) setEntry(parseFloat(((tick.bid + tick.ask) / 2).toFixed(5)));
  };

  const applyPreset = (p: Preset) => {
    setRiskPct(p.riskPct);
    applySymbol(p.symbol);
    setTickSize(p.tickSize);
    setTickValue(p.tickValue);
    setStep(p.step);
  };

  const sld        = Math.abs(entry - sl);
  const ticks_     = tickSize > 0 ? sld / tickSize : 0;
  const lossPerLot = ticks_ * tickValue;
  const riskUsd    = equity * (riskPct / 100);
  const raw        = lossPerLot > 0 ? riskUsd / lossPerLot : 0;
  const lot        = Math.floor(raw / step) * step;
  const pipsToSl   = tickSize >= 0.01 ? sld / 1 : sld / 0.0001;
  const reject     = raw > 0 && raw < step;
  const valid      = sld > 0 && lossPerLot > 0 && !reject;

  // Pip value display: estimated USD value of 1 pip per lot
  const pipValue   = tickSize >= 0.01 ? tickValue : tickValue * 10;

  const liveMid    = ticks[symbol] ? (ticks[symbol].bid + ticks[symbol].ask) / 2 : null;

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        data-testid="position-size-toggle"
        className="fixed bottom-4 right-4 z-40 rounded-full bg-accent-cyan/90 px-4 py-2 text-xs font-bold text-bg-primary shadow-lg hover:bg-accent-cyan transition-colors"
      >
        ⌗ Lot Calc
      </button>

      {open && (
        <div
          data-testid="position-size-calc"
          className="fixed bottom-16 right-4 z-40 w-80 rounded-lg border border-white/10 bg-bg-secondary shadow-2xl"
        >
          {/* Header */}
          <div className="flex items-center justify-between border-b border-white/5 px-3 py-2">
            <span className="text-xs font-bold uppercase tracking-wider text-white/70">Position Size Calculator</span>
            <button type="button" onClick={() => setOpen(false)} className="text-white/30 hover:text-white">
              <X size={13} />
            </button>
          </div>

          <div className="p-3 space-y-3">
            {/* Account summary */}
            <div className="grid grid-cols-2 gap-2 rounded bg-bg-tertiary p-2 text-[10px]">
              <div>
                <span className="text-white/40 uppercase tracking-wider">Equity</span>
                <div className="font-mono text-accent-green font-bold">${equity.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</div>
              </div>
              <div>
                <span className="text-white/40 uppercase tracking-wider">Balance</span>
                <div className="font-mono text-white font-bold">${balance.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</div>
              </div>
            </div>

            {/* Quick Risk Presets */}
            <div>
              <p className="mb-1 text-[9px] uppercase tracking-wider text-white/40">Quick Presets</p>
              <div className="grid grid-cols-2 gap-1">
                {PRESETS.map((p) => (
                  <button
                    key={p.label}
                    type="button"
                    onClick={() => applyPreset(p)}
                    className="rounded bg-bg-tertiary px-2 py-1 text-[10px] font-mono text-white/60 hover:bg-accent-cyan/20 hover:text-accent-cyan transition-colors text-left"
                  >
                    {p.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Symbol + Risk */}
            <div className="grid grid-cols-2 gap-2 text-xs">
              <label className="flex flex-col">
                <span className="text-[9px] uppercase text-white/40 mb-0.5">Symbol</span>
                <select
                  value={symbol}
                  onChange={(e) => applySymbol(e.target.value)}
                  className="rounded bg-bg-tertiary px-2 py-1 font-mono text-white focus:outline-none focus:ring-1 focus:ring-accent-cyan"
                >
                  {SYMBOLS_13.map(({ name }) => <option key={name} value={name}>{name}</option>)}
                </select>
              </label>
              <Field label="Risk %" value={riskPct} onChange={setRiskPct} step={0.1} min={0.1} max={5} />
            </div>

            {/* Live price chip */}
            {liveMid != null && (
              <div className="flex items-center gap-2 text-[10px]">
                <span className="text-white/30">Live mid:</span>
                <span className="font-mono text-accent-cyan">{liveMid.toFixed(liveMid > 100 ? 2 : 5)}</span>
                <button
                  type="button"
                  onClick={() => setEntry(parseFloat(liveMid.toFixed(5)))}
                  className="ml-auto rounded bg-accent-cyan/10 px-2 py-0.5 text-accent-cyan hover:bg-accent-cyan/20 transition-colors"
                >
                  Use as Entry
                </button>
              </div>
            )}

            {/* Entry + SL */}
            <div className="grid grid-cols-2 gap-2 text-xs">
              <Field label="Entry" value={entry} onChange={setEntry} step={0.00001} />
              <Field label="Stop Loss" value={sl}    onChange={setSl}   step={0.00001} />
            </div>

            {/* Advanced toggle */}
            <button
              type="button"
              onClick={() => setShowAdvanced((v) => !v)}
              className="flex w-full items-center gap-1 text-[9px] uppercase tracking-wider text-white/30 hover:text-white/60"
            >
              {showAdvanced ? <ChevronUp size={10} /> : <ChevronDown size={10} />}
              Advanced (tick size / value / vol step)
            </button>

            {showAdvanced && (
              <div className="grid grid-cols-3 gap-2 text-xs">
                <Field label="Tick sz"  value={tickSize}  onChange={setTickSize}  step={0.00001} />
                <Field label="Tick val" value={tickValue} onChange={setTickValue} step={0.01} />
                <Field label="Vol step" value={step}      onChange={setStep}      step={0.01} />
              </div>
            )}

            {/* Results */}
            <div className={`rounded p-2.5 text-xs space-y-1 ${reject ? 'bg-accent-red/10 border border-accent-red/20' : valid ? 'bg-bg-tertiary' : 'bg-bg-tertiary'}`}>
              <Row label="SL distance"   value={sld > 0 ? sld.toFixed(5) : '—'} />
              <Row label="Pips to SL"    value={sld > 0 ? pipsToSl.toFixed(1) : '—'} />
              <Row label="Pip value/lot" value={`$${pipValue.toFixed(2)}`} />
              <Row label="Risk USD"      value={`$${riskUsd.toFixed(2)}`} accent="gold" />
              {reject ? (
                <p className="pt-1 text-[10px] font-bold text-accent-red">
                  INSUFFICIENT EQUITY — raw lot {raw.toFixed(4)} &lt; min {step.toFixed(2)}
                </p>
              ) : (
                <>
                  <Row label="Raw lot"       value={raw > 0 ? raw.toFixed(4) : '—'} />
                  <Row label="Lot (quantised)" value={valid ? lot.toFixed(2) : '—'} accent={valid ? 'green' : undefined}
                    testId="calc-lot" bold />
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
}

function Field({
  label, value, onChange, step = 1, min, max,
}: {
  label: string; value: number; onChange: (n: number) => void;
  step?: number; min?: number; max?: number;
}) {
  return (
    <label className="flex flex-col">
      <span className="text-[9px] uppercase text-white/40 mb-0.5">{label}</span>
      <input
        type="number" step={step} value={value} min={min} max={max}
        onChange={(e) => onChange(Number(e.target.value))}
        className="rounded bg-bg-tertiary px-2 py-1 text-xs font-mono text-white outline-none focus:ring-1 focus:ring-accent-cyan"
      />
    </label>
  );
}

function Row({
  label, value, accent, bold, testId,
}: {
  label: string; value: string; accent?: 'green' | 'gold' | 'red'; bold?: boolean; testId?: string;
}) {
  const color = accent === 'green' ? 'text-accent-green' : accent === 'gold' ? 'text-accent-gold' : accent === 'red' ? 'text-accent-red' : 'text-white';
  return (
    <div className="flex items-center justify-between">
      <span className="text-white/40">{label}</span>
      <span data-testid={testId} className={`font-mono ${color} ${bold ? 'font-bold text-sm' : ''}`}>{value}</span>
    </div>
  );
}
