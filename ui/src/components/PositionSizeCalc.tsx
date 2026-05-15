import { useState } from 'react';
import { useEngineStore } from '@/store/engineStore';

/**
 * Floating Appendix-E lot calculator. Pure JS — same formula the engine
 * uses, so traders can sanity-check a setup before any signal fires.
 */
export default function PositionSizeCalc() {
  const equity = useEngineStore((s) => s.account?.equity ?? 10_000);
  const [open, setOpen] = useState(false);
  const [riskPct, setRiskPct] = useState(2.0);
  const [entry, setEntry] = useState(1.07300);
  const [sl, setSl] = useState(1.07200);
  const [tickSize, setTickSize] = useState(0.00001);
  const [tickValue, setTickValue] = useState(1.0);
  const [step, setStep] = useState(0.01);

  const sld = Math.abs(entry - sl);
  const ticks = sld / tickSize;
  const lossPerLot = ticks * tickValue;
  const riskUsd = equity * (riskPct / 100);
  const raw = lossPerLot > 0 ? riskUsd / lossPerLot : 0;
  const lot = Math.floor(raw / step) * step;

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        data-testid="position-size-toggle"
        className="fixed bottom-4 right-4 z-40 rounded-full bg-accent-cyan/90 px-4 py-2 text-xs font-bold text-bg-primary shadow-lg hover:bg-accent-cyan"
      >
        ⌗ Lot Calc
      </button>
      {open && (
        <div data-testid="position-size-calc"
             className="fixed bottom-16 right-4 z-40 w-72 rounded-lg border border-white/10 bg-bg-secondary p-3 shadow-xl">
          <h3 className="text-xs font-bold uppercase tracking-wider text-white/70">Position Size</h3>
          <div className="mt-2 grid grid-cols-2 gap-2 text-xs font-mono">
            <Field label="Equity"      value={equity}     onChange={() => {}} readOnly />
            <Field label="Risk %"      value={riskPct}    onChange={setRiskPct} step={0.1} />
            <Field label="Entry"       value={entry}      onChange={setEntry}    step={0.0001} />
            <Field label="SL"          value={sl}         onChange={setSl}       step={0.0001} />
            <Field label="Tick size"   value={tickSize}   onChange={setTickSize} step={0.00001} />
            <Field label="Tick value"  value={tickValue}  onChange={setTickValue} step={0.1} />
            <Field label="Vol step"    value={step}       onChange={setStep}     step={0.01} />
          </div>
          <div className="mt-2 rounded bg-bg-tertiary p-2 text-xs">
            <p>SL distance: <span className="font-mono">{sld.toFixed(5)}</span></p>
            <p>Risk USD: <span className="font-mono text-accent-gold">${riskUsd.toFixed(2)}</span></p>
            <p>Raw lot: <span className="font-mono">{raw.toFixed(4)}</span></p>
            <p>Lot (quantised): <span data-testid="calc-lot" className="font-mono text-accent-green">{lot.toFixed(2)}</span></p>
          </div>
        </div>
      )}
    </>
  );
}

function Field({
  label, value, onChange, step = 1, readOnly = false,
}: {
  label: string; value: number; onChange: (n: number) => void;
  step?: number; readOnly?: boolean;
}) {
  return (
    <label className="flex flex-col">
      <span className="text-[9px] uppercase text-white/40">{label}</span>
      <input
        type="number" step={step} value={value} readOnly={readOnly}
        onChange={(e) => onChange(Number(e.target.value))}
        className="rounded bg-bg-tertiary px-2 py-1 text-xs font-mono text-white outline-none focus:ring-1 focus:ring-accent-cyan"
      />
    </label>
  );
}
