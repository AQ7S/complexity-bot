import { useEffect, useRef } from 'react';
import {
  ColorType, createChart,
  type IChartApi, type ISeriesApi, type Time,
} from 'lightweight-charts';

export type Candle = { time: number; open: number; high: number; low: number; close: number };
export type Marker = { time: number; price: number; kind: 'BUY' | 'SELL' | 'EXIT' };

export type EmaSeries = { period: number; data: { time: number; value: number }[] };

/**
 * Minimal Lightweight Charts v4 wrapper:
 *   - candlesticks
 *   - up to N EMA overlays
 *   - trade markers (entry/exit) via setMarkers
 */
export default function CandlestickChart({
  candles, emas = [], markers = [], height = 380,
}: {
  candles: Candle[];
  emas?: EmaSeries[];
  markers?: Marker[];
  height?: number;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const emaSeriesRef = useRef<ISeriesApi<'Line'>[]>([]);

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
    });
    chartRef.current = chart;
    const cs = chart.addCandlestickSeries({
      upColor: '#00ff88', downColor: '#ff3b6b',
      borderUpColor: '#00ff88', borderDownColor: '#ff3b6b',
      wickUpColor: '#00ff88', wickDownColor: '#ff3b6b',
    });
    candleRef.current = cs;
    return () => { chart.remove(); chartRef.current = null; };
  }, []);

  useEffect(() => {
    candleRef.current?.setData(candles.map((c) => ({
      time: c.time as Time, open: c.open, high: c.high, low: c.low, close: c.close,
    })));
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
    if (!candleRef.current) return;
    candleRef.current.setMarkers(markers.map((m) => ({
      time: m.time as Time,
      position: m.kind === 'BUY' ? 'belowBar' : 'aboveBar',
      color: m.kind === 'BUY' ? '#00ff88' : m.kind === 'SELL' ? '#ff3b6b' : '#ffb800',
      shape: m.kind === 'EXIT' ? 'square' : 'arrowUp',
      text: m.kind,
    })));
  }, [markers]);

  return (
    <div
      data-testid="candlestick-chart"
      ref={containerRef}
      style={{ height }}
      className="w-full rounded-lg border border-white/5 bg-bg-primary"
    />
  );
}
