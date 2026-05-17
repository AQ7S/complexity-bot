import '@testing-library/jest-dom/vitest';
import { vi } from 'vitest';

process.on('unhandledRejection', (reason: any) => {
  const msg = String(reason?.message ?? reason ?? '');
  if (msg.includes('AbortSignal') || msg.includes('Expected signal')) return;
  throw reason;
});

// jsdom shims that components reach for at import or mount time.
if (!('matchMedia' in window)) {
  (window as any).matchMedia = (query: string) => ({
    matches: false, media: query, onchange: null,
    addListener: () => {}, removeListener: () => {},
    addEventListener: () => {}, removeEventListener: () => {},
    dispatchEvent: () => false,
  });
}
if (!('ResizeObserver' in window)) {
  (window as any).ResizeObserver = class {
    observe(){} unobserve(){} disconnect(){}
  };
}

// Lightweight Charts uses canvas APIs jsdom doesn't implement; stub the whole
// module so any test that mounts a chart-bearing route can render.
vi.mock('lightweight-charts', () => {
  const makeSeries = () => ({
    setData: () => {},
    setMarkers: () => {},
    update: () => {},
    applyOptions: () => {},
    createPriceLine: () => ({}),
    removePriceLine: () => {},
    priceToCoordinate: () => 0,
    coordinateToPrice: () => 0,
  });
  const makeTimeScale = () => ({
    subscribeVisibleTimeRangeChange: () => {},
    unsubscribeVisibleTimeRangeChange: () => {},
    fitContent: () => {},
    setVisibleRange: () => {},
    timeToCoordinate: () => 0,
    coordinateToTime: () => 0,
    width: () => 800,
    options: () => ({}),
    applyOptions: () => {},
  });
  return {
    ColorType: { Solid: 'solid' },
    LineStyle: { Solid: 0, Dotted: 1, Dashed: 2, LargeDashed: 3, SparseDotted: 4 },
    CrosshairMode: { Normal: 0, Magnet: 1, Hidden: 2 },
    PriceScaleMode: { Normal: 0, Logarithmic: 1, Percentage: 2, IndexedTo100: 3 },
    createChart: () => ({
      addCandlestickSeries: makeSeries,
      addLineSeries: makeSeries,
      addAreaSeries: makeSeries,
      addHistogramSeries: makeSeries,
      addBarSeries: makeSeries,
      removeSeries: () => {},
      remove: () => {},
      subscribeCrosshairMove: () => {},
      unsubscribeCrosshairMove: () => {},
      subscribeClick: () => {},
      unsubscribeClick: () => {},
      timeScale: makeTimeScale,
      priceScale: () => ({ applyOptions: () => {} }),
      applyOptions: () => {},
      resize: () => {},
      takeScreenshot: () => null,
    }),
  };
});
