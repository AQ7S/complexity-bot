import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { TestRouter } from '@/router';

describe('routing', () => {
  it.each([
    ['/',         'page-command-center'],
    ['/charts',   'page-charts'],
    ['/journal',  'page-trade-journal'],
    ['/ai',       'page-ai-engine'],
    ['/settings', 'page-settings'],
    ['/trace',    'page-decision-trace'],
  ])('mounts %s', (path, testid) => {
    render(<TestRouter initialPath={path} />);
    expect(screen.getByTestId(testid)).toBeInTheDocument();
    expect(screen.getByTestId('sidebar')).toBeInTheDocument();
  });
});
