import { describe, it, expect } from 'vitest';
import { toCsv } from '@/lib/csv';

describe('csv', () => {
  it('writes header + rows', () => {
    const rows = [
      { a: 1, b: 'x' },
      { a: 2, b: 'y' },
    ];
    const csv = toCsv(rows);
    expect(csv.split('\n')).toEqual(['a,b', '1,x', '2,y']);
  });

  it('quotes cells containing commas, quotes, or newlines', () => {
    const rows = [{ x: 'hello, world', y: 'she said "hi"', z: 'a\nb' }];
    const csv = toCsv(rows);
    expect(csv).toContain('"hello, world"');
    expect(csv).toContain('"she said ""hi"""');
    expect(csv).toContain('"a\nb"');
  });

  it('handles empty rows with explicit columns', () => {
    expect(toCsv([], ['x' as never, 'y' as never])).toBe('x,y');
  });

  it('treats null/undefined as empty', () => {
    const csv = toCsv([{ a: null as any, b: undefined as any, c: 0 }]);
    expect(csv).toBe('a,b,c\n,,0');
  });
});
