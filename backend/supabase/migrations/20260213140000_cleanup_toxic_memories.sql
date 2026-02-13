-- One-time cleanup: deactivate toxic "mistake" memories that cause defensive behavior loops.
-- These memories contain self-imposed restrictions like "Avoid similar setups" that prevent
-- the Captain from trading when it should be finding edge.

UPDATE agent_memories
SET active = false
WHERE memory_type = 'mistake'
  AND (
    content ILIKE '%avoid similar%'
    OR content ILIKE '%crisis%'
    OR content ILIKE '%pause trading%'
    OR content ILIKE '%never trade%'
    OR content ILIKE '%no directional%'
    OR content ILIKE '%stop trading%'
    OR content ILIKE '%trading freeze%'
    OR content ILIKE '%recovery plan%'
  );
