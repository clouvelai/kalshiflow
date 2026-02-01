-- Deduplication index for extractions
-- Prevents duplicate extractions when the same source is reprocessed (e.g., after restart)
-- Uses (source_id, extraction_class, md5(extraction_text)) as the dedup key

CREATE UNIQUE INDEX IF NOT EXISTS idx_extractions_dedup
  ON extractions(source_id, extraction_class, md5(extraction_text));
