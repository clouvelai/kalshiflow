-- Enable Row Level Security on all public tables
-- All access is via the service_role key from the backend, so we add
-- a permissive policy for service_role on each table.

-- lifecycle_events
ALTER TABLE public.lifecycle_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Allow all for service role" ON public.lifecycle_events
  FOR ALL USING (auth.role() = 'service_role');

-- markets
ALTER TABLE public.markets ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Allow all for service role" ON public.markets
  FOR ALL USING (auth.role() = 'service_role');

-- migrations
ALTER TABLE public.migrations ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Allow all for service role" ON public.migrations
  FOR ALL USING (auth.role() = 'service_role');

-- order_contexts
ALTER TABLE public.order_contexts ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Allow all for service role" ON public.order_contexts
  FOR ALL USING (auth.role() = 'service_role');

-- orderbook_signals
ALTER TABLE public.orderbook_signals ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Allow all for service role" ON public.orderbook_signals
  FOR ALL USING (auth.role() = 'service_role');

-- rl_models
ALTER TABLE public.rl_models ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Allow all for service role" ON public.rl_models
  FOR ALL USING (auth.role() = 'service_role');

-- rl_orderbook_deltas
ALTER TABLE public.rl_orderbook_deltas ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Allow all for service role" ON public.rl_orderbook_deltas
  FOR ALL USING (auth.role() = 'service_role');

-- rl_orderbook_sessions
ALTER TABLE public.rl_orderbook_sessions ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Allow all for service role" ON public.rl_orderbook_sessions
  FOR ALL USING (auth.role() = 'service_role');

-- rl_orderbook_snapshots
ALTER TABLE public.rl_orderbook_snapshots ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Allow all for service role" ON public.rl_orderbook_snapshots
  FOR ALL USING (auth.role() = 'service_role');

-- rl_trading_actions
ALTER TABLE public.rl_trading_actions ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Allow all for service role" ON public.rl_trading_actions
  FOR ALL USING (auth.role() = 'service_role');

-- rl_trading_episodes
ALTER TABLE public.rl_trading_episodes ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Allow all for service role" ON public.rl_trading_episodes
  FOR ALL USING (auth.role() = 'service_role');

-- trades
ALTER TABLE public.trades ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Allow all for service role" ON public.trades
  FOR ALL USING (auth.role() = 'service_role');
