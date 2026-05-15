import '@testing-library/jest-dom/vitest';
import { vi } from 'vitest';

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
  const series = () => ({ setData: () => {}, setMarkers: () => {} });
  return {
    ColorType: { Solid: 'solid' },
    createChart: () => ({
      addCandlestickSeries: series,
      addLineSeries: series,
      removeSeries: () => {},
      remove: () => {},
    }),
  };
});
