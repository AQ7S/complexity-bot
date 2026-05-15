import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { TestRouter } from '@/router';
// lightweight-charts is mocked globally in tests/setup.ts.

describe('Charts page', () => {
  it('mounts the candlestick canvas', () => {
    render(<TestRouter initialPath="/charts" />);
    expect(screen.getByTestId('candlestick-chart')).toBeInTheDocument();
  });

  it('switches timeframes without crashing', () => {
    render(<TestRouter initialPath="/charts" />);
    fireEvent.click(screen.getByTestId('tf-H1'));
    fireEvent.click(screen.getByTestId('tf-D1'));
    expect(screen.getByTestId('page-charts')).toBeInTheDocument();
  });

  it('switches symbols via the dropdown', () => {
    render(<TestRouter initialPath="/charts" />);
    const sel = screen.getByTestId('chart-symbol') as HTMLSelectElement;
    fireEvent.change(sel, { target: { value: 'GOLD#' } });
    expect(sel.value).toBe('GOLD#');
  });
});
