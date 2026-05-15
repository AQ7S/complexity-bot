export const fmtUsd = (n: number): string =>
  `$${n.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

export const fmtSignedUsd = (n: number): string =>
  `${n >= 0 ? '+' : '-'}$${Math.abs(n).toLocaleString('en-US', {
    minimumFractionDigits: 2, maximumFractionDigits: 2,
  })}`;

export const fmtPct = (frac: number, digits = 2): string =>
  `${(frac * 100).toFixed(digits)}%`;

export const fmtPrice = (n: number, digits = 5): string => n.toFixed(digits);
