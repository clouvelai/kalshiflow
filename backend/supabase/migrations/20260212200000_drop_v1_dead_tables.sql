-- Drop V1 tables that have zero code references after V1→V2 migration.
-- Kept: markets, trades, agent_memories, captain_task_ledger, news_price_impacts,
--        order_contexts, schema_info, market_price_impacts

DROP TABLE IF EXISTS semantic_frames CASCADE;
DROP TABLE IF EXISTS research_decisions CASCADE;
DROP TABLE IF EXISTS reddit_entities CASCADE;
DROP TABLE IF EXISTS related_entities CASCADE;
DROP TABLE IF EXISTS entity_aliases CASCADE;
DROP TABLE IF EXISTS news_entities CASCADE;
DROP TABLE IF EXISTS entity_event_entities CASCADE;
DROP TABLE IF EXISTS kb_entities CASCADE;
DROP TABLE IF EXISTS objective_entities CASCADE;
DROP TABLE IF EXISTS entity_mentions CASCADE;
DROP TABLE IF EXISTS entity_relations CASCADE;
DROP TABLE IF EXISTS event_configs CASCADE;
DROP TABLE IF EXISTS extractions CASCADE;
DROP TABLE IF EXISTS extraction_examples CASCADE;
DROP TABLE IF EXISTS paired_markets CASCADE;
DROP TABLE IF EXISTS price_ticks CASCADE;
DROP TABLE IF EXISTS arb_trades CASCADE;
