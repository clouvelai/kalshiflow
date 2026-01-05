# Tooling Backlog

Track automation ideas and improvements here. Prioritize based on time savings and frequency of use.

## High Priority

- [ ] Batch validation runner (run all strategies overnight)
- [ ] Automated data freshness alerts (warn if data >7 days old)
- [ ] Strategy comparison report generator (side-by-side metrics)

## Medium Priority

- [ ] Hypothesis queue management CLI
- [ ] Auto-generate YAML config from analysis script
- [ ] Validation result diff tool (compare before/after data updates)

## Ideas

- [ ] Slack/Discord notifications for completed validations
- [ ] Web UI for validation results browsing
- [ ] Strategy performance dashboard
- [ ] Automated weekly data update cron job
- [ ] ML-based hypothesis generator (find patterns in what works)

## Completed

- [x] **Validation Framework** - `research/scripts/validation/` (2026-01-04)
  - YAML config format, LSD/Full modes, 20x speedup with caching
- [x] **Data Update Script** - `research/scripts/update_research_data.py` (2026-01-04)
  - Idempotent updates from prod Supabase, incremental appends

---

## How to Add Items

When you notice repetitive work or friction:

1. Add it here with a brief description
2. Estimate time savings: `(time per occurrence) × (occurrences per week)`
3. Prioritize: High = >2 hours/week saved, Medium = 30min-2hr/week, Ideas = nice-to-have

## Review Schedule

Review this backlog monthly and promote items based on actual pain points encountered.
