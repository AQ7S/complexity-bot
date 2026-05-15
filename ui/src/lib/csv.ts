/** Tiny RFC-4180-ish CSV serializer. Quotes only when needed; escapes "" inside. */

const NEEDS_QUOTING = /[",\n\r]/;

function cell(v: unknown): string {
  if (v === null || v === undefined) return '';
  const s = String(v);
  if (!NEEDS_QUOTING.test(s)) return s;
  return `"${s.replace(/"/g, '""')}"`;
}

export function toCsv<T extends Record<string, unknown>>(rows: T[], columns?: (keyof T)[]): string {
  if (rows.length === 0 && !columns) return '';
  const cols = (columns ?? Object.keys(rows[0] ?? {})) as (keyof T)[];
  const header = cols.map((c) => cell(String(c))).join(',');
  const body = rows.map((r) => cols.map((c) => cell(r[c])).join(',')).join('\n');
  return rows.length === 0 ? header : `${header}\n${body}`;
}

export function downloadCsv(filename: string, content: string): void {
  if (typeof document === 'undefined') return;
  const blob = new Blob([content], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click();
  document.body.removeChild(a); URL.revokeObjectURL(url);
}
