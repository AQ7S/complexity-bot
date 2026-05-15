/**
 * Mirrors `engine/config/symbols.py` and `engine/utils/time_utils.py`.
 * Hand-maintained — keep in sync if the engine list changes.
 */
export const SYMBOLS_13 = [
  { name: 'EURUSD#',         kind: 'FX'     },
  { name: 'USDJPY#',         kind: 'FX'     },
  { name: 'GBPUSD#',         kind: 'FX'     },
  { name: 'USDCHF#',         kind: 'FX'     },
  { name: 'GOLD#',           kind: 'METAL'  },
  { name: 'BTCUSD#',         kind: 'CRYPTO' },
  { name: 'ETHUSD#',         kind: 'CRYPTO' },
  { name: 'AI_INDX#',        kind: 'INDEX'  },
  { name: 'Crypto_10#',      kind: 'INDEX'  },
  { name: 'TrumpWinners#',   kind: 'EVENT'  },
  { name: 'HarrisWinners#',  kind: 'EVENT'  },
  { name: 'EURJPY#',         kind: 'FX'     },
  { name: 'AUDUSD#',         kind: 'FX'     },
] as const;

export const ALWAYS_ON = new Set([
  'GOLD#', 'BTCUSD#', 'ETHUSD#', 'AI_INDX#', 'Crypto_10#',
]);

/**
 * ICT kill zones (EST). Matches `engine/utils/time_utils.py::KILL_ZONES`.
 * `start`/`end` are minutes since EST midnight.
 */
export const KILL_ZONES = [
  { label: 'Asian',         start: 19 * 60, end: 22 * 60 },
  { label: 'London Open',   start:  2 * 60, end:  5 * 60 },
  { label: 'NY Open',       start:  7 * 60, end: 10 * 60 },
  { label: 'London Close',  start: 10 * 60, end: 12 * 60 },
] as const;

export const TIMEFRAMES = ['M1', 'M5', 'M15', 'H1', 'H4', 'D1'] as const;
export type Timeframe = (typeof TIMEFRAMES)[number];
