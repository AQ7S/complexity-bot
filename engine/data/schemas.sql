-- =============================================================================
-- DuckDB schema for Complexity Engine market data store
-- Path: ./engine/data/store/market.duckdb
-- =============================================================================

CREATE SEQUENCE IF NOT EXISTS seq_smc;
CREATE SEQUENCE IF NOT EXISTS seq_perf;

CREATE TABLE IF NOT EXISTS ticks (
  symbol     VARCHAR NOT NULL,
  ts         TIMESTAMP_MS NOT NULL,
  bid        DOUBLE NOT NULL,
  ask        DOUBLE NOT NULL,
  volume     DOUBLE,
  flags      INTEGER,
  source     VARCHAR DEFAULT 'mt5'
);
CREATE INDEX IF NOT EXISTS idx_ticks_sym_ts ON ticks(symbol, ts);

CREATE TABLE IF NOT EXISTS bars (
  symbol     VARCHAR NOT NULL,
  timeframe  VARCHAR NOT NULL,
  ts         TIMESTAMP NOT NULL,
  open       DOUBLE NOT NULL,
  high       DOUBLE NOT NULL,
  low        DOUBLE NOT NULL,
  close      DOUBLE NOT NULL,
  volume     DOUBLE NOT NULL,
  spread     INTEGER
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_bars ON bars(symbol, timeframe, ts);
CREATE INDEX IF NOT EXISTS idx_bars_tf_ts ON bars(timeframe, ts);

CREATE TABLE IF NOT EXISTS features (
  symbol     VARCHAR NOT NULL,
  timeframe  VARCHAR NOT NULL,
  ts         TIMESTAMP NOT NULL,
  payload    JSON NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_feat ON features(symbol, timeframe, ts);

CREATE TABLE IF NOT EXISTS smc_zones (
  id         BIGINT PRIMARY KEY DEFAULT nextval('seq_smc'),
  symbol     VARCHAR NOT NULL,
  timeframe  VARCHAR NOT NULL,
  ts_created TIMESTAMP NOT NULL,
  ts_invalidated TIMESTAMP,
  zone_type  VARCHAR NOT NULL,
  direction  VARCHAR NOT NULL,
  price_high DOUBLE NOT NULL,
  price_low  DOUBLE NOT NULL,
  strength   DOUBLE,
  active     BOOLEAN DEFAULT TRUE
);
CREATE INDEX IF NOT EXISTS idx_smc_active ON smc_zones(symbol, timeframe, active);

CREATE TABLE IF NOT EXISTS spread_history (
  symbol     VARCHAR NOT NULL,
  ts         TIMESTAMP NOT NULL,
  spread     DOUBLE NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_spread ON spread_history(symbol, ts);

CREATE TABLE IF NOT EXISTS correlation_matrix (
  ts         TIMESTAMP NOT NULL,
  payload    JSON NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_corr_ts ON correlation_matrix(ts);

CREATE TABLE IF NOT EXISTS performance_sessions (
  id         BIGINT PRIMARY KEY DEFAULT nextval('seq_perf'),
  date       DATE NOT NULL,
  session    VARCHAR NOT NULL,
  symbol     VARCHAR NOT NULL,
  trades     INTEGER NOT NULL,
  wins       INTEGER NOT NULL,
  losses     INTEGER NOT NULL,
  pnl_usd    DOUBLE NOT NULL,
  avg_rr     DOUBLE,
  best_pnl   DOUBLE,
  worst_pnl  DOUBLE
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_perf ON performance_sessions(date, session, symbol);

CREATE TABLE IF NOT EXISTS price_alerts_cache (
  id         BIGINT NOT NULL,
  symbol     VARCHAR NOT NULL,
  direction  VARCHAR NOT NULL,
  threshold  DOUBLE NOT NULL,
  enabled    BOOLEAN NOT NULL DEFAULT TRUE
);
CREATE INDEX IF NOT EXISTS idx_alerts_sym ON price_alerts_cache(symbol, enabled);

CREATE TABLE IF NOT EXISTS regime_history (
  symbol     VARCHAR NOT NULL,
  ts         TIMESTAMP NOT NULL,
  regime     VARCHAR NOT NULL,
  adx        DOUBLE,
  atr_pct    DOUBLE
);
CREATE INDEX IF NOT EXISTS idx_regime ON regime_history(symbol, ts);
