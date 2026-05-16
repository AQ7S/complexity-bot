import { useEffect, useRef, useState } from 'react';
import {
  ColorType, createChart, LineStyle,
  type IChartApi, type ISeriesApi, type Time,
} from 'lightweight-charts';

export type Candle = { time: number; open: number; high: number; low: number; close: number };
export type Marker = { time: number; price: number; kind: 'BUY' | 'SELL' | 'EXIT'; text?: string };
export type EmaSeries = { period: number; data: { time: number; value: number }[] };
export type VwapSeries = { time: number; value: number }[];
export type Zone = { time_from: number; time_to: number; price_high: number; price_low: number; kind: 'OB_BULL' | 'OB_BEAR' | 'FVG_BULL' | 'FVG_BEAR' };
export type ShadedRange = { time_from: number; time_to: number; label?: string };
export type CrosshairInfo = { time: number; ohlc: Candle | null };

const ZONE_FILL: Record<Zone['kind'], string> = {
  OB_BULL:  'rgba(0, 212, 255, 0.18)',
  OB_BEAR:  'rgba(124, 92, 255, 0.18)',
  FVG_BULL: 'rgba(0, 255, 136, 0.10)',
  FVG_BEAR: 'rgba(255, 59, 107, 0.10)',
};
const ZONE_STROKE: Record<Zone['kind'], string> = {
  OB_BULL:  'rgba(0, 212, 255, 0.55)',
  OB_BEAR:  'rgba(124, 92, 255, 0.55)',
  FVG_BULL: 'rgba(0, 255, 136, 0.45)',
  FVG_BEAR: 'rgba(255, 59, 107, 0.45)',
};

export default function CandlestickChart({
  candles, emas = [], markers = [], vwap = [], zones = [], shadedRanges = [],
  height = 480, onCrosshair,
}: {
  candles: Candle[];
  emas?: EmaSeries[];
  markers?: Marker[];
  vwap?: VwapSeries;
  zones?: Zone[];
  shadedRanges?: ShadedRange[];
  height?: number;
  onCrosshair?: (info: CrosshairInfo) => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const overlayRef = useRef<SVGSVGElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const emaSeriesRef = useRef<ISeriesApi<'Line'>[]>([]);
  const vwapRef = useRef<ISeriesApi<'Line'> | null>(null);
  const [, force] = useState(0);

  useEffect(() => {
    if (!containerRef.current) return;
    const chart = createChart(containerRef.current, {
      autoSize: true,
      layout: {
        background: { type: ColorType.Solid, color: '#0a0e1a' },
        textColor: '#e2e8f0',
      },
      grid: { vertLines: { color: '#1a2238' }, horzLines: { color: '#1a2238' } },
      timeScale: { timeVisible: true, secondsVisible: false },
      crosshair: { mode: 0 },
    });
    chartRef.current = chart;
    const cs = chart.addCandlestickSeries({
      upColor: '#00ff88', downColor: '#ff3b6b',
      borderUpColor: '#00ff88', borderDownColor: '#ff3b6b',
      wickUpColor: '#00ff88', wickDownColor: '#ff3b6b',
    });
    candleRef.current = cs;
    chart.subscribeCrosshairMove((param) => {
      const t = param?.time as number | undefined;
      if (t == null) {
        onCrosshair?.({ time: 0, ohlc: null });
        return;
      }
      const data = param.seriesData.get(cs) as any;
      onCrosshair?.({
        time: t,
        ohlc: data ? { time: t, open: data.open, high: data.high, low: data.low, close: data.close } : null,
      });
    });
    const obs = new ResizeObserver(() => force((n) => n + 1));
    obs.observe(containerRef.current);
    const timeScaleSub = chart.timeScale().subscribeVisibleTimeRangeChange(() => force((n) => n + 1));
    void timeScaleSub;
    return () => { obs.disconnect(); chart.remove(); chartRef.current = null; };
  }, []);

  useEffect(() => {
    candleRef.current?.setData(candles.map((c) => ({
      time: c.time as Time, open: c.open, high: c.high, low: c.low, close: c.close,
    })));
    force((n) => n + 1);
  }, [candles]);

  useEffect(() => {
    const chart = chartRef.current; if (!chart) return;
    for (const s of emaSeriesRef.current) chart.removeSeries(s);
    const palette = ['#00d4ff', '#ffb800', '#7c5cff'];
    emaSeriesRef.current = emas.map((e, i) => {
      const s = chart.addLineSeries({
        color: palette[i % palette.length], lineWidth: 1,
        priceLineVisible: false, lastValueVisible: false,
      });
      s.setData(e.data.map((d) => ({ time: d.time as Time, value: d.value })));
      return s;
    });
  }, [emas]);

  useEffect(() => {
    const chart = chartRef.current; if (!chart) return;
    if (vwapRef.current) { chart.removeSeries(vwapRef.current); vwapRef.current = null; }
    if (vwap.length > 0) {
      const s = chart.addLineSeries({
        color: '#ffb800', lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        priceLineVisible: false, lastValueVisible: true,
      });
      s.setData(vwap.map((p) => ({ time: p.time as Time, value: p.value })));
      vwapRef.current = s;
    }
  }, [vwap]);

  useEffect(() => {
    if (!candleRef.current) return;
    candleRef.current.setMarkers(markers.map((m) => ({
      time: m.time as Time,
      position: m.kind === 'BUY' ? 'belowBar' : m.kind === 'SELL' ? 'aboveBar' : 'inBar',
      color: m.kind === 'BUY' ? '#00ff88' : m.kind === 'SELL' ? '#ff3b6b' : '#ffb800',
      shape: m.kind === 'BUY' ? 'arrowUp' : m.kind === 'SELL' ? 'arrowDown' : 'circle',
      text: m.text ?? m.kind,
    })));
  }, [markers]);

  // SVG overlay: zone rectangles + shaded backgrounds
  const overlayItems = (() => {
    const chart = chartRef.current; const cs = candleRef.current;
    if (!chart || !cs || !containerRef.current) return null;
    const w = containerRef.current.clientWidth;
    const h = containerRef.current.clientHeight;
    if (w === 0 || h === 0) return null;

    const t2x = (t: number) => chart.timeScale().timeToCoordinate(t as Time);
    const p2y = (p: number) => cs.priceToCoordinate(p);

    const rects: JSX.Element[] = [];
    for (const z of zones) {
      const x1 = t2x(z.time_from); const x2 = t2x(z.time_to);
      const yh = p2y(z.price_high); const yl = p2y(z.price_low);
      if (x1 == null || x2 == null || yh == null || yl == null) continue;
      const left = Math.min(x1, x2);
      const right = Math.max(x1, x2, w);
      const top = Math.min(yh, yl);
      const bottom = Math.max(yh, yl);
      rects.push(
        <rect
          key={`zone-${z.kind}-${z.time_from}-${z.price_high}`}
          x={left} y={top} width={right - left} height={bottom - top}
          fill={ZONE_FILL[z.kind]} stroke={ZONE_STROKE[z.kind]} strokeWidth={1} strokeDasharray="4,4"
        />,
      );
    }
    for (const r of shadedRanges) {
      const x1 = t2x(r.time_from); const x2 = t2x(r.time_to);
      if (x1 == null || x2 == null) continue;
      const left = Math.min(x1, x2);
      const right = Math.max(x1, x2);
      rects.push(
        <rect
          key={`kz-${r.time_from}`}
          x={left} y={0} width={right - left} height={h}
          fill="rgba(0, 212, 255, 0.04)"
        />,
      );
    }
    return rects;
  })();

  return (
    <div
      className="relative w-full rounded-lg border border-white/5 bg-bg-primary"
      style={{ height }}
    >
      <div
        data-testid="candlestick-chart"
        ref={containerRef}
        className="absolute inset-0"
      />
      <svg
        ref={overlayRef}
        className="pointer-events-none absolute inset-0"
        width="100%" height="100%"
      >
        {overlayItems}
      </svg>
    </div>
  );
}
