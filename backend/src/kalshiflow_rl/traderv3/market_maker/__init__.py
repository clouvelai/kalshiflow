"""Market Maker - Deterministic quoting engine for Kalshi prediction markets.

QuoteEngine maintains continuous two-sided quotes on prediction market events.
Controlled by Captain via configure_quotes/pull_quotes/resume_quotes tools.
No separate LLM agent — Captain configures, QuoteEngine executes.
"""
