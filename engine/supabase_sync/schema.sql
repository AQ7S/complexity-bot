-- =============================================================================
-- Supabase (Postgres) schema for remote sync + dashboards
-- Apply once via Supabase SQL editor or `supabase db push`.
-- =============================================================================

-- Restore Supabase's default grants in case prior migrations stripped them.
GRANT USAGE ON SCHEMA public TO service_role, anon, authenticated;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL    ON TABLES    TO service_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES    TO anon;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL    ON SEQUENCES TO service_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE  ON SEQUENCES TO anon;

CREATE TABLE IF NOT EXISTS public.trades (
  id BIGSERIAL PRIMARY KEY,
  mt5_ticket BIGINT UNIQUE,
  symbol TEXT NOT NULL,
  direction TEXT NOT NULL CHECK (direction IN ('BUY','SELL')),
  entry_price NUMERIC(18,8) NOT NULL,
  exit_price NUMERIC(18,8),
  lot_size NUMERIC(10,4) NOT NULL,
  sl NUMERIC(18,8) NOT NULL,
  tp NUMERIC(18,8) NOT NULL,
  pnl NUMERIC(18,4),
  r_r_achieved NUMERIC(8,4),
  open_time TIMESTAMPTZ NOT NULL,
  close_time TIMESTAMPTZ,
  close_reason TEXT CHECK (close_reason IN ('TP','SL','TRAIL','MANUAL','KILL','NEWS')),
  signal_confluence_score INT CHECK (signal_confluence_score BETWEEN 0 AND 5),
  claude_decision TEXT,
  claude_confidence INT CHECK (claude_confidence BETWEEN 0 AND 100),
  claude_reasoning TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_trades_symbol_open ON public.trades(symbol, open_time DESC);
CREATE INDEX IF NOT EXISTS idx_trades_close ON public.trades(close_time DESC);

CREATE TABLE IF NOT EXISTS public.model_performance (
  id BIGSERIAL PRIMARY KEY,
  model_name TEXT NOT NULL CHECK (model_name IN ('cnn_lstm','rl_dqn')),
  version TEXT NOT NULL,
  ts TIMESTAMPTZ NOT NULL DEFAULT now(),
  accuracy NUMERIC(6,4),
  loss NUMERIC(12,6),
  sharpe NUMERIC(8,4),
  total_trades_trained_on INT
);
CREATE INDEX IF NOT EXISTS idx_model_perf_ts ON public.model_performance(model_name, ts DESC);

CREATE TABLE IF NOT EXISTS public.claude_decisions (
  id BIGSERIAL PRIMARY KEY,
  trade_id BIGINT REFERENCES public.trades(id) ON DELETE SET NULL,
  symbol TEXT NOT NULL,
  context_json JSONB NOT NULL,
  decision TEXT NOT NULL CHECK (decision IN ('BUY','SELL','SKIP')),
  confidence INT NOT NULL CHECK (confidence BETWEEN 0 AND 100),
  reasoning TEXT NOT NULL,
  risk_adjustment NUMERIC(4,2) NOT NULL,
  ts TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_claude_ts ON public.claude_decisions(ts DESC);

CREATE TABLE IF NOT EXISTS public.account_snapshots (
  id BIGSERIAL PRIMARY KEY,
  ts TIMESTAMPTZ NOT NULL DEFAULT now(),
  equity NUMERIC(18,4) NOT NULL,
  balance NUMERIC(18,4) NOT NULL,
  drawdown_pct NUMERIC(6,4) NOT NULL,
  open_positions_count INT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_snapshots_ts ON public.account_snapshots(ts DESC);

CREATE TABLE IF NOT EXISTS public.signals (
  id BIGSERIAL PRIMARY KEY,
  symbol TEXT NOT NULL,
  timeframe TEXT NOT NULL,
  signal_type TEXT NOT NULL CHECK (signal_type IN ('BUY','SELL','HOLD')),
  smc_zone TEXT,
  confidence INT CHECK (confidence BETWEEN 0 AND 100),
  news_flag BOOLEAN NOT NULL DEFAULT FALSE,
  kill_zone_active BOOLEAN NOT NULL DEFAULT FALSE,
  ts TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_signals_sym_ts ON public.signals(symbol, ts DESC);

CREATE TABLE IF NOT EXISTS public.weekly_debriefs (
  id BIGSERIAL PRIMARY KEY,
  week_start DATE NOT NULL UNIQUE,
  markdown TEXT NOT NULL,
  param_recommendations JSONB,
  trades_count INT,
  net_pnl NUMERIC(18,4),
  win_rate NUMERIC(5,4),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE public.trades              ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.model_performance   ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.claude_decisions    ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.account_snapshots   ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.signals             ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.weekly_debriefs     ENABLE ROW LEVEL SECURITY;

DO $$
DECLARE r RECORD;
BEGIN
  FOR r IN
    SELECT * FROM (VALUES
      ('service_role_all_trades',    'public.trades',             'ALL',    'service_role'),
      ('service_role_all_model',     'public.model_performance',  'ALL',    'service_role'),
      ('service_role_all_claude',    'public.claude_decisions',   'ALL',    'service_role'),
      ('service_role_all_snapshots', 'public.account_snapshots',  'ALL',    'service_role'),
      ('service_role_all_signals',   'public.signals',            'ALL',    'service_role'),
      ('service_role_all_debriefs',  'public.weekly_debriefs',    'ALL',    'service_role'),
      ('anon_select_trades',         'public.trades',             'SELECT', 'anon'),
      ('anon_select_model',          'public.model_performance',  'SELECT', 'anon'),
      ('anon_select_claude',         'public.claude_decisions',   'SELECT', 'anon'),
      ('anon_select_snapshots',      'public.account_snapshots',  'SELECT', 'anon'),
      ('anon_select_signals',        'public.signals',            'SELECT', 'anon'),
      ('anon_select_debriefs',       'public.weekly_debriefs',    'SELECT', 'anon')
    ) AS t(pname, tname, op, role)
  LOOP
    EXECUTE format('DROP POLICY IF EXISTS %I ON %s', r.pname, r.tname);
    IF r.op = 'ALL' THEN
      EXECUTE format('CREATE POLICY %I ON %s FOR ALL TO %I USING (true) WITH CHECK (true)',
                     r.pname, r.tname, r.role);
    ELSE
      EXECUTE format('CREATE POLICY %I ON %s FOR SELECT TO %I USING (true)',
                     r.pname, r.tname, r.role);
    END IF;
  END LOOP;
END $$;
