## Backend

### Environments

The repo uses dotenv-style env files for different environments. For TRADER V3 paper mode, you can use any of:

- `backend/.env.paper` (existing convention)
- `backend/.paper.env` (also supported)
- `.paper.env` at repo root (also supported by `scripts/run-v3.sh`)

### Truth Social evidence (optional)

TRADER V3’s event-first research can optionally pull “Trump circle” narrative/intent evidence from Truth Social using [`truthbrush`](https://github.com/stanfordio/truthbrush).

**How it works:**
- The system maintains a **global in-memory cache** of posts from your authenticated Truth Social account’s **Following list** (auto-discovered “Trump circle”) plus trending hashtags/posts
- Cache refreshes periodically in the background
- Evidence queries search this cache (not fetched per-event)
- Engagement metrics (likes/reblogs/replies) are included in evidence metadata

**Required credentials** (add to your paper env file):

- `TRUTHSOCIAL_USERNAME` - Your Truth Social username
- `TRUTHSOCIAL_PASSWORD` - Your Truth Social password

**Configuration options:**

- `TRUTHSOCIAL_EVIDENCE_ENABLED=auto` (default `auto`; set `true`/`false` to force enable/disable)
- `TRUTHSOCIAL_CACHE_REFRESH_SECONDS=300` (how often to refresh posts from followed users, default 5 minutes)
- `TRUTHSOCIAL_TRENDING_REFRESH_SECONDS=600` (how often to refresh trending data, default 10 minutes)
- `TRUTHSOCIAL_HOURS_BACK=24` (how far back to fetch posts, default 24 hours)

**Important notes:**

- **Following discovery is required**: If the system cannot discover your following list, Truth Social evidence will be **hard-disabled** (no fallback). This ensures we only use the auto-curated “Trump circle” Truth Social created for you.
- Evidence is treated as **narrative/intent signal**, not independently verified fact
- If credentials are missing or `truthbrush` is not installed, the system continues normally without Truth Social evidence (graceful degradation)

