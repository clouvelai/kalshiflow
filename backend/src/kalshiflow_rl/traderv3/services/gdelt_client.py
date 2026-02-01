"""
GDELT BigQuery Client - Queries the GDELT Global Knowledge Graph via Google BigQuery.

GDELT indexes news from thousands of sources worldwide, updated every 15 minutes.
It complements Reddit signals as a more authoritative/timely news source.

Authentication:
- GDELT data is public and free to query
- BigQuery charges for bytes processed (first 1 TB/month free, then $6.25/TB)
- Requires a GCP project ID for billing (set via GDELT_GCP_PROJECT_ID)
- Local dev: `gcloud auth application-default login`
- Production: GOOGLE_APPLICATION_CREDENTIALS env var pointing to service account JSON

Cost mitigations:
- Partitioned table (_PARTITIONTIME filter) limits scan scope
- Column selection (only needed fields)
- LIMIT clause caps rows
- In-memory TTL cache (5 min default) deduplicates repeated queries
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger("kalshiflow_rl.traderv3.services.gdelt_client")

# CAMEO root codes — for enriching event results with human-readable labels
CAMEO_ROOT_CODES = {
    "01": "PUBLIC STATEMENT", "02": "APPEAL", "03": "INTENT TO COOPERATE",
    "04": "CONSULT", "05": "DIPLOMATIC COOPERATION", "06": "MATERIAL COOPERATION",
    "07": "PROVIDE AID", "08": "YIELD", "09": "INVESTIGATE", "10": "DEMAND",
    "11": "DISAPPROVE", "12": "REJECT", "13": "THREATEN", "14": "PROTEST",
    "15": "EXHIBIT FORCE", "16": "REDUCE RELATIONS", "17": "COERCE",
    "18": "ASSAULT", "19": "FIGHT", "20": "UNCONVENTIONAL MASS VIOLENCE",
}

# QuadClass labels — 4-way conflict/cooperation classification
QUAD_CLASS_LABELS = {
    1: "Verbal Cooperation", 2: "Material Cooperation",
    3: "Verbal Conflict", 4: "Material Conflict",
}


class GDELTClient:
    """
    BigQuery wrapper for querying the GDELT Global Knowledge Graph (GKG).

    Uses lazy initialization for the BigQuery client to avoid import-time
    dependencies. Queries are wrapped in asyncio.to_thread() to avoid
    blocking the event loop (same pattern as PRAW in RedditHistoricAgent).
    """

    def __init__(
        self,
        gcp_project_id: str,
        cache_ttl_seconds: float = 300.0,
        max_results: int = 100,
        default_window_hours: float = 4.0,
        max_bytes_per_session: int = 500 * 1024 * 1024,  # 500MB default (~$0.003)
    ):
        self._gcp_project_id = gcp_project_id
        self._cache_ttl_seconds = cache_ttl_seconds
        self._max_results = max_results
        self._default_window_hours = default_window_hours
        self._max_bytes_per_session = max_bytes_per_session

        # Lazy-initialized BigQuery client
        self._bq_client = None

        # In-memory TTL cache: {cache_key: (timestamp, result)}
        self._cache: Dict[str, tuple] = {}

        # Cost monitoring
        self._bytes_processed_total: int = 0
        self._query_count: int = 0
        self._budget_exhausted: bool = False

        logger.info(
            f"[gdelt_client] Initialized (project={gcp_project_id}, "
            f"cache_ttl={cache_ttl_seconds}s, max_results={max_results}, "
            f"budget={max_bytes_per_session/(1024*1024):.0f}MB)"
        )

    def _get_bq_client(self):
        """Lazy-initialize the BigQuery client."""
        if self._bq_client is None:
            from google.cloud import bigquery
            self._bq_client = bigquery.Client(project=self._gcp_project_id)
            logger.info(f"[gdelt_client] BigQuery client initialized for project {self._gcp_project_id}")
        return self._bq_client

    def _cache_key(self, search_terms: List[str], window_hours: float,
                   tone_filter: Optional[str], source_filter: Optional[str],
                   limit: int) -> str:
        """Generate a deterministic cache key from query parameters."""
        key_data = json.dumps({
            "terms": sorted(search_terms),
            "window": window_hours,
            "tone": tone_filter,
            "source": source_filter,
            "limit": limit,
        }, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_cached(self, key: str) -> Optional[Dict[str, Any]]:
        """Get a cached result if it exists and hasn't expired."""
        if key in self._cache:
            ts, result = self._cache[key]
            if time.time() - ts < self._cache_ttl_seconds:
                logger.debug(f"[gdelt_client] Cache hit for key {key[:8]}")
                return result
            else:
                del self._cache[key]
        return None

    def _set_cache(self, key: str, result: Dict[str, Any]) -> None:
        """Store a result in the cache."""
        self._cache[key] = (time.time(), result)
        # Prune expired entries periodically (keep cache small)
        if len(self._cache) > 50:
            now = time.time()
            self._cache = {
                k: (ts, v) for k, (ts, v) in self._cache.items()
                if now - ts < self._cache_ttl_seconds
            }

    def _build_query(
        self,
        search_terms: List[str],
        window_hours: float,
        tone_filter: Optional[str],
        source_filter: Optional[str],
        limit: int,
    ) -> str:
        """Build the BigQuery SQL for querying GDELT GKG."""
        # Build REGEXP pattern from search terms (case-insensitive via (?i))
        escaped_terms = [term.replace("'", "\\'") for term in search_terms]
        pattern = "|".join(escaped_terms)

        # Partition time filter for cost efficiency
        cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
        partition_date = cutoff.strftime("%Y-%m-%d")

        # Core query against partitioned GKG table
        query = f"""
SELECT
    DocumentIdentifier AS url,
    DATE AS date,
    V2Tone AS tone_raw,
    V2Persons AS persons_raw,
    V2Organizations AS orgs_raw,
    AllNames AS names_raw,
    V2Themes AS themes_raw,
    V2Counts AS counts_raw,
    Quotations AS quotations_raw,
    SourceCommonName AS source_name,
    SharingImage AS image_url
FROM `gdelt-bq.gdeltv2.gkg_partitioned`
WHERE
    _PARTITIONTIME >= TIMESTAMP('{partition_date}')
    AND DATE >= {int(cutoff.timestamp())}
    AND (
        REGEXP_CONTAINS(V2Persons, r'(?i){pattern}')
        OR REGEXP_CONTAINS(V2Organizations, r'(?i){pattern}')
        OR REGEXP_CONTAINS(AllNames, r'(?i){pattern}')
        OR REGEXP_CONTAINS(V2Themes, r'(?i){pattern}')
    )"""

        # Optional tone filter
        if tone_filter == "positive":
            query += "\n    AND CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64) > 1.5"
        elif tone_filter == "negative":
            query += "\n    AND CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64) < -1.5"

        # Optional source filter
        if source_filter:
            safe_source = source_filter.replace("'", "\\'")
            query += f"\n    AND LOWER(SourceCommonName) LIKE '%{safe_source.lower()}%'"

        query += f"\nORDER BY DATE DESC\nLIMIT {limit}"

        return query

    def _parse_tone(self, tone_raw: Optional[str]) -> Dict[str, float]:
        """Parse V2Tone semicolon-delimited field into structured dict."""
        if not tone_raw:
            return {"tone": 0.0, "positive_score": 0.0, "negative_score": 0.0,
                    "polarity": 0.0, "activity_density": 0.0, "word_count": 0}
        parts = tone_raw.split(",")
        try:
            return {
                "tone": float(parts[0]) if len(parts) > 0 else 0.0,
                "positive_score": float(parts[1]) if len(parts) > 1 else 0.0,
                "negative_score": float(parts[2]) if len(parts) > 2 else 0.0,
                "polarity": float(parts[3]) if len(parts) > 3 else 0.0,
                "activity_density": float(parts[4]) if len(parts) > 4 else 0.0,
                "word_count": int(float(parts[5])) if len(parts) > 5 else 0,
            }
        except (ValueError, IndexError):
            return {"tone": 0.0, "positive_score": 0.0, "negative_score": 0.0,
                    "polarity": 0.0, "activity_density": 0.0, "word_count": 0}

    def _parse_semicolon_field(self, raw: Optional[str], max_items: int = 20) -> List[str]:
        """Parse a semicolon-delimited GDELT field into a list of strings."""
        if not raw:
            return []
        items = [item.strip() for item in raw.split(";") if item.strip()]
        return items[:max_items]

    def _process_results(self, rows: List[Dict]) -> Dict[str, Any]:
        """Process raw BigQuery rows into structured output."""
        if not rows:
            return {
                "article_count": 0,
                "source_diversity": 0,
                "tone_summary": {"avg_tone": 0.0, "positive_count": 0, "negative_count": 0, "neutral_count": 0},
                "key_themes": [],
                "key_persons": [],
                "key_organizations": [],
                "top_articles": [],
                "timeline": [],
            }

        # Collect aggregates
        sources = set()
        tones = []
        theme_counts: Dict[str, int] = {}
        person_counts: Dict[str, int] = {}
        org_counts: Dict[str, int] = {}
        articles = []
        timeline_buckets: Dict[str, int] = {}

        for row in rows:
            # Source diversity
            source_name = row.get("source_name", "")
            if source_name:
                sources.add(source_name)

            # Tone analysis
            tone_data = self._parse_tone(row.get("tone_raw"))
            tones.append(tone_data["tone"])

            # Themes
            themes = self._parse_semicolon_field(row.get("themes_raw"))
            for theme in themes:
                # Clean theme names (GDELT themes are uppercase with underscores)
                clean_theme = theme.split(",")[0] if "," in theme else theme
                theme_counts[clean_theme] = theme_counts.get(clean_theme, 0) + 1

            # Persons
            persons = self._parse_semicolon_field(row.get("persons_raw"))
            for person in persons:
                clean_person = person.split(",")[0] if "," in person else person
                person_counts[clean_person] = person_counts.get(clean_person, 0) + 1

            # Organizations
            orgs = self._parse_semicolon_field(row.get("orgs_raw"))
            for org in orgs:
                clean_org = org.split(",")[0] if "," in org else org
                org_counts[clean_org] = org_counts.get(clean_org, 0) + 1

            # Article details
            url = row.get("url", "")
            quotations = self._parse_semicolon_field(row.get("quotations_raw"), max_items=3)
            if url:
                articles.append({
                    "url": url,
                    "source": source_name,
                    "tone": round(tone_data["tone"], 2),
                    "quotations": quotations[:2],
                    "key_persons": persons[:5],
                    "key_orgs": orgs[:5],
                })

            # Timeline (hourly buckets)
            date_val = row.get("date")
            if date_val:
                try:
                    if isinstance(date_val, (int, float)):
                        dt = datetime.fromtimestamp(date_val, tz=timezone.utc)
                    else:
                        dt = datetime.fromisoformat(str(date_val))
                    bucket = dt.strftime("%Y-%m-%d %H:00")
                    timeline_buckets[bucket] = timeline_buckets.get(bucket, 0) + 1
                except (ValueError, TypeError, OSError):
                    pass

        # Compute tone summary
        positive_count = sum(1 for t in tones if t > 1.5)
        negative_count = sum(1 for t in tones if t < -1.5)
        neutral_count = len(tones) - positive_count - negative_count
        avg_tone = sum(tones) / len(tones) if tones else 0.0

        # Sort and limit aggregates
        top_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        top_persons = sorted(person_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        top_orgs = sorted(org_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        # Timeline sorted chronologically
        timeline = sorted(
            [{"hour": k, "articles": v} for k, v in timeline_buckets.items()],
            key=lambda x: x["hour"]
        )

        return {
            "article_count": len(rows),
            "source_diversity": len(sources),
            "tone_summary": {
                "avg_tone": round(avg_tone, 2),
                "positive_count": positive_count,
                "negative_count": negative_count,
                "neutral_count": neutral_count,
            },
            "key_themes": [{"theme": t, "count": c} for t, c in top_themes],
            "key_persons": [{"person": p, "count": c} for p, c in top_persons],
            "key_organizations": [{"org": o, "count": c} for o, c in top_orgs],
            "top_articles": articles[:10],
            "timeline": timeline,
        }

    def _execute_query_sync(
        self,
        search_terms: List[str],
        window_hours: float,
        tone_filter: Optional[str],
        source_filter: Optional[str],
        limit: int,
    ) -> Dict[str, Any]:
        """Synchronous BigQuery execution (called via asyncio.to_thread)."""
        client = self._get_bq_client()
        query = self._build_query(search_terms, window_hours, tone_filter, source_filter, limit)

        logger.info(f"[gdelt_client] Executing query for terms={search_terms}, window={window_hours}h, limit={limit}")

        job = client.query(query)
        result = job.result()

        # Track cost
        bytes_processed = job.total_bytes_processed or 0
        self._bytes_processed_total += bytes_processed
        self._query_count += 1

        logger.info(
            f"[gdelt_client] Query complete: {result.total_rows} rows, "
            f"{bytes_processed / (1024*1024):.1f} MB processed "
            f"(total: {self._bytes_processed_total / (1024*1024*1024):.2f} GB)"
        )

        # Convert to list of dicts
        rows = []
        for row in result:
            rows.append(dict(row))

        return self._process_results(rows)

    async def query_news(
        self,
        search_terms: List[str],
        window_hours: Optional[float] = None,
        tone_filter: Optional[str] = None,
        source_filter: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Query GDELT news for the given search terms.

        Args:
            search_terms: List of terms to search for (ORed across persons, orgs, names, themes)
            window_hours: How far back to look (default: self._default_window_hours)
            tone_filter: Optional tone filter: "positive", "negative", or None
            source_filter: Optional source name filter (partial match)
            limit: Max articles to return (default: self._max_results)

        Returns:
            Dict with article_count, source_diversity, tone_summary, key_themes,
            key_persons, key_organizations, top_articles, timeline
        """
        window_hours = window_hours or self._default_window_hours
        limit = limit or self._max_results

        if not search_terms:
            return {"error": "No search terms provided", "article_count": 0}

        # Check budget before executing paid query
        if self._bytes_processed_total >= self._max_bytes_per_session:
            if not self._budget_exhausted:
                self._budget_exhausted = True
                logger.warning(
                    f"[gdelt_client] BigQuery budget exhausted: "
                    f"{self._bytes_processed_total/(1024*1024):.0f}MB / "
                    f"{self._max_bytes_per_session/(1024*1024):.0f}MB"
                )
            return {
                "error": (
                    "BigQuery budget exhausted for this session "
                    f"({self._bytes_processed_total/(1024*1024):.0f}MB used). "
                    "Use search_gdelt_articles() instead (free, no budget limit)."
                ),
                "article_count": 0,
                "search_terms": search_terms,
                "budget_exhausted": True,
            }

        # Check cache
        key = self._cache_key(search_terms, window_hours, tone_filter, source_filter, limit)
        cached = self._get_cached(key)
        if cached is not None:
            cached["_cached"] = True
            return cached

        # Execute query in thread pool (BigQuery client is synchronous)
        try:
            result = await asyncio.to_thread(
                self._execute_query_sync,
                search_terms, window_hours, tone_filter, source_filter, limit,
            )
            result["search_terms"] = search_terms
            result["window_hours"] = window_hours
            result["bytes_processed_total"] = self._bytes_processed_total
            result["query_count"] = self._query_count

            # Cache the result
            self._set_cache(key, result)

            return result

        except Exception as e:
            logger.error(f"[gdelt_client] Query failed: {e}")
            return {
                "error": str(e),
                "article_count": 0,
                "search_terms": search_terms,
            }

    # =========================================================================
    # GDELT Events Database (Actor-Event-Actor triples)
    # =========================================================================

    def _events_cache_key(self, actor_names: List[str], country_filter: Optional[str],
                          window_hours: float, limit: int) -> str:
        """Generate a deterministic cache key for events queries."""
        key_data = json.dumps({
            "type": "events",
            "actors": sorted(actor_names),
            "country": country_filter,
            "window": window_hours,
            "limit": limit,
        }, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()

    def _build_events_query(
        self,
        actor_names: List[str],
        country_filter: Optional[str],
        window_hours: float,
        limit: int,
    ) -> str:
        """Build BigQuery SQL for querying GDELT Events (Actor-Event-Actor triples)."""
        # Build REGEXP pattern for actor names (case-insensitive)
        escaped = [name.replace("'", "\\'") for name in actor_names]
        pattern = "|".join(escaped)

        # Partition time filter
        cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
        partition_date = cutoff.strftime("%Y-%m-%d")
        date_int = int(cutoff.strftime("%Y%m%d"))

        query = f"""
SELECT
    GLOBALEVENTID,
    SQLDATE,
    Actor1Name, Actor1CountryCode, Actor1Type1Code,
    Actor2Name, Actor2CountryCode, Actor2Type1Code,
    EventCode, EventBaseCode, EventRootCode,
    QuadClass,
    GoldsteinScale,
    NumMentions, NumSources, NumArticles,
    AvgTone,
    ActionGeo_FullName,
    SOURCEURL,
    DATEADDED
FROM `gdelt-bq.gdeltv2.events_partitioned`
WHERE
    _PARTITIONTIME >= TIMESTAMP('{partition_date}')
    AND SQLDATE >= {date_int}
    AND (
        REGEXP_CONTAINS(Actor1Name, r'(?i){pattern}')
        OR REGEXP_CONTAINS(Actor2Name, r'(?i){pattern}')
    )"""

        if country_filter:
            safe_country = country_filter.replace("'", "\\'").upper()[:3]
            query += f"\n    AND (Actor1CountryCode = '{safe_country}' OR Actor2CountryCode = '{safe_country}')"

        query += f"\nORDER BY NumMentions DESC\nLIMIT {limit}"
        return query

    def _process_event_results(self, rows: List[Dict]) -> Dict[str, Any]:
        """Process raw BigQuery event rows into structured output with CAMEO enrichment."""
        if not rows:
            return {
                "event_count": 0,
                "quad_class_summary": {},
                "goldstein_summary": {"avg": 0, "min": 0, "max": 0, "positive_count": 0, "negative_count": 0},
                "tone_summary": {"avg_tone": 0.0},
                "top_event_triples": [],
                "top_actors": [],
                "event_code_distribution": [],
                "geo_hotspots": [],
                "timeline": [],
            }

        # Aggregate collectors
        quad_counts: Dict[int, int] = {}
        goldstein_values: List[float] = []
        tone_values: List[float] = []
        actor_stats: Dict[str, Dict] = {}  # name -> {count, goldstein_sum}
        event_code_counts: Dict[str, int] = {}
        geo_counts: Dict[str, int] = {}
        timeline_buckets: Dict[str, int] = {}
        triples = []

        for row in rows:
            # QuadClass distribution
            qc = row.get("QuadClass")
            if qc is not None:
                qc_int = int(qc)
                quad_counts[qc_int] = quad_counts.get(qc_int, 0) + 1

            # Goldstein
            gs = row.get("GoldsteinScale")
            if gs is not None:
                goldstein_values.append(float(gs))

            # Tone
            tone = row.get("AvgTone")
            if tone is not None:
                tone_values.append(float(tone))

            # Actor tracking
            for actor_field in ("Actor1Name", "Actor2Name"):
                name = row.get(actor_field)
                if name:
                    if name not in actor_stats:
                        actor_stats[name] = {"count": 0, "goldstein_sum": 0.0}
                    actor_stats[name]["count"] += 1
                    if gs is not None:
                        actor_stats[name]["goldstein_sum"] += float(gs)

            # Event code distribution
            root_code = str(row.get("EventRootCode", ""))
            if root_code:
                event_code_counts[root_code] = event_code_counts.get(root_code, 0) + 1

            # Geo hotspots
            geo = row.get("ActionGeo_FullName")
            if geo:
                geo_counts[geo] = geo_counts.get(geo, 0) + 1

            # Timeline (daily buckets from SQLDATE)
            sql_date = row.get("SQLDATE")
            if sql_date:
                date_str = str(sql_date)
                if len(date_str) == 8:
                    bucket = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
                    timeline_buckets[bucket] = timeline_buckets.get(bucket, 0) + 1

            # Build triple
            root_code_str = str(row.get("EventRootCode", ""))
            triples.append({
                "actor1": {
                    "name": row.get("Actor1Name") or "Unknown",
                    "country": row.get("Actor1CountryCode") or "",
                    "type": row.get("Actor1Type1Code") or "",
                },
                "event": {
                    "code": str(row.get("EventCode", "")),
                    "root_code": root_code_str,
                    "description": CAMEO_ROOT_CODES.get(root_code_str, "UNKNOWN"),
                },
                "actor2": {
                    "name": row.get("Actor2Name") or "Unknown",
                    "country": row.get("Actor2CountryCode") or "",
                    "type": row.get("Actor2Type1Code") or "",
                },
                "goldstein": float(gs) if gs is not None else None,
                "quad_class": int(qc) if qc is not None else None,
                "quad_class_label": QUAD_CLASS_LABELS.get(int(qc), "Unknown") if qc is not None else None,
                "mentions": row.get("NumMentions", 0),
                "sources": row.get("NumSources", 0),
                "source_url": row.get("SOURCEURL", ""),
            })

        # Goldstein summary
        gs_positive = sum(1 for g in goldstein_values if g > 0)
        gs_negative = sum(1 for g in goldstein_values if g < 0)
        goldstein_summary = {
            "avg": round(sum(goldstein_values) / len(goldstein_values), 2) if goldstein_values else 0,
            "min": round(min(goldstein_values), 2) if goldstein_values else 0,
            "max": round(max(goldstein_values), 2) if goldstein_values else 0,
            "positive_count": gs_positive,
            "negative_count": gs_negative,
        }

        # Quad class summary with labels
        quad_class_summary = {
            str(qc): {"count": count, "label": QUAD_CLASS_LABELS.get(qc, "Unknown")}
            for qc, count in sorted(quad_counts.items())
        }

        # Tone summary
        avg_tone = sum(tone_values) / len(tone_values) if tone_values else 0.0

        # Top actors
        top_actors = sorted(actor_stats.items(), key=lambda x: x[1]["count"], reverse=True)[:15]
        top_actors_list = [
            {
                "name": name,
                "count": stats["count"],
                "avg_goldstein": round(stats["goldstein_sum"] / stats["count"], 2) if stats["count"] > 0 else 0,
            }
            for name, stats in top_actors
        ]

        # Event code distribution with labels
        sorted_codes = sorted(event_code_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        event_code_dist = [
            {"code": code, "description": CAMEO_ROOT_CODES.get(code, "UNKNOWN"), "count": count}
            for code, count in sorted_codes
        ]

        # Geo hotspots
        sorted_geos = sorted(geo_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        geo_hotspots = [{"location": loc, "count": count} for loc, count in sorted_geos]

        # Timeline
        timeline = sorted(
            [{"date": k, "events": v} for k, v in timeline_buckets.items()],
            key=lambda x: x["date"]
        )

        return {
            "event_count": len(rows),
            "quad_class_summary": quad_class_summary,
            "goldstein_summary": goldstein_summary,
            "tone_summary": {"avg_tone": round(avg_tone, 2)},
            "top_event_triples": triples[:15],
            "top_actors": top_actors_list,
            "event_code_distribution": event_code_dist,
            "geo_hotspots": geo_hotspots,
            "timeline": timeline,
        }

    def _execute_events_query_sync(
        self,
        actor_names: List[str],
        country_filter: Optional[str],
        window_hours: float,
        limit: int,
    ) -> Dict[str, Any]:
        """Synchronous BigQuery execution for events (called via asyncio.to_thread)."""
        client = self._get_bq_client()
        query = self._build_events_query(actor_names, country_filter, window_hours, limit)

        logger.info(f"[gdelt_client] Executing events query for actors={actor_names}, window={window_hours}h, limit={limit}")

        job = client.query(query)
        result = job.result()

        # Track cost
        bytes_processed = job.total_bytes_processed or 0
        self._bytes_processed_total += bytes_processed
        self._query_count += 1

        logger.info(
            f"[gdelt_client] Events query complete: {result.total_rows} rows, "
            f"{bytes_processed / (1024*1024):.1f} MB processed "
            f"(total: {self._bytes_processed_total / (1024*1024*1024):.2f} GB)"
        )

        rows = [dict(row) for row in result]
        return self._process_event_results(rows)

    async def query_events(
        self,
        actor_names: List[str],
        country_filter: Optional[str] = None,
        window_hours: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Query GDELT Events Database for Actor-Event-Actor triples.

        Searches the events_partitioned table for structured event data
        involving the specified actor names. Returns CAMEO-enriched triples
        with GoldsteinScale conflict/cooperation scoring.

        Args:
            actor_names: Entity names to search in both Actor1 and Actor2 positions
            country_filter: Optional 3-letter ISO country code filter
            window_hours: How far back to look (default: self._default_window_hours)
            limit: Max events to return (default: 50)

        Returns:
            Dict with event_count, quad_class_summary, goldstein_summary,
            top_event_triples, top_actors, event_code_distribution, geo_hotspots, timeline
        """
        window_hours = window_hours or self._default_window_hours
        limit = limit or min(self._max_results, 50)

        if not actor_names:
            return {"error": "No actor names provided", "event_count": 0}

        # Check budget
        if self._bytes_processed_total >= self._max_bytes_per_session:
            if not self._budget_exhausted:
                self._budget_exhausted = True
                logger.warning(
                    f"[gdelt_client] BigQuery budget exhausted: "
                    f"{self._bytes_processed_total/(1024*1024):.0f}MB / "
                    f"{self._max_bytes_per_session/(1024*1024):.0f}MB"
                )
            return {
                "error": (
                    "BigQuery budget exhausted for this session "
                    f"({self._bytes_processed_total/(1024*1024):.0f}MB used)."
                ),
                "event_count": 0,
                "actor_names": actor_names,
                "budget_exhausted": True,
            }

        # Check cache
        key = self._events_cache_key(actor_names, country_filter, window_hours, limit)
        cached = self._get_cached(key)
        if cached is not None:
            cached["_cached"] = True
            return cached

        # Execute query
        try:
            result = await asyncio.to_thread(
                self._execute_events_query_sync,
                actor_names, country_filter, window_hours, limit,
            )
            result["actor_names"] = actor_names
            result["window_hours"] = window_hours
            result["bytes_processed_total"] = self._bytes_processed_total
            result["query_count"] = self._query_count

            self._set_cache(key, result)
            return result

        except Exception as e:
            logger.error(f"[gdelt_client] Events query failed: {e}")
            return {
                "error": str(e),
                "event_count": 0,
                "actor_names": actor_names,
            }

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics for monitoring."""
        return {
            "bytes_processed_total": self._bytes_processed_total,
            "bytes_processed_gb": round(self._bytes_processed_total / (1024 * 1024 * 1024), 3),
            "max_bytes_per_session": self._max_bytes_per_session,
            "budget_used_pct": round(
                self._bytes_processed_total / self._max_bytes_per_session * 100, 1
            ) if self._max_bytes_per_session > 0 else 0,
            "budget_exhausted": self._budget_exhausted,
            "query_count": self._query_count,
            "cache_entries": len(self._cache),
            "cache_ttl_seconds": self._cache_ttl_seconds,
        }


class GDELTDocClient:
    """
    Free GDELT DOC 2.0 API client for article search and volume timelines.

    Unlike GDELTClient (BigQuery), this uses the free public REST API at
    api.gdeltproject.org. No GCP project or authentication needed.

    Limitations:
    - Last 3 months of data only
    - Max 250 articles per request
    - Less structured entity data than GKG

    Best for:
    - Quick article search when BigQuery isn't configured
    - Volume timeline analysis (coverage trends)
    - Free fallback for GDELT news intelligence
    """

    DOC_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

    def __init__(
        self,
        cache_ttl_seconds: float = 300.0,
        max_records: int = 75,
        default_timespan: str = "4h",
    ):
        self._cache_ttl_seconds = cache_ttl_seconds
        self._max_records = min(max_records, 250)
        self._default_timespan = default_timespan
        self._cache: Dict[str, tuple] = {}
        self._query_count: int = 0

        logger.info(
            f"[gdelt_doc] Initialized (cache_ttl={cache_ttl_seconds}s, "
            f"max_records={max_records}, timespan={default_timespan})"
        )

    def _cache_key(self, query: str, mode: str, timespan: str) -> str:
        """Generate deterministic cache key."""
        key_data = json.dumps({"q": query, "m": mode, "t": timespan}, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_cached(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached result if fresh."""
        if key in self._cache:
            ts, result = self._cache[key]
            if time.time() - ts < self._cache_ttl_seconds:
                return result
            del self._cache[key]
        return None

    def _set_cache(self, key: str, result: Dict[str, Any]) -> None:
        """Store result in cache."""
        self._cache[key] = (time.time(), result)
        if len(self._cache) > 50:
            now = time.time()
            self._cache = {
                k: (ts, v) for k, (ts, v) in self._cache.items()
                if now - ts < self._cache_ttl_seconds
            }

    def _build_query_string(self, search_terms: List[str], tone_filter: Optional[str] = None,
                            source_country: Optional[str] = None) -> str:
        """Build GDELT DOC API query string from parameters."""
        # Join terms with OR for broad matching
        query = " OR ".join(f'"{t}"' if " " in t else t for t in search_terms)

        # Optional tone filter
        if tone_filter == "positive":
            query += " tone>1.5"
        elif tone_filter == "negative":
            query += " tone<-1.5"

        # Optional country filter
        if source_country:
            query += f" sourcecountry:{source_country}"

        return query

    def _fetch_sync(self, params: Dict[str, str]) -> Dict[str, Any]:
        """Synchronous HTTP fetch (called via asyncio.to_thread)."""
        import urllib.request
        import urllib.parse
        import urllib.error

        url = f"{self.DOC_API_URL}?{urllib.parse.urlencode(params)}"
        logger.info(f"[gdelt_doc] Fetching: {url[:200]}")

        req = urllib.request.Request(url, headers={
            "User-Agent": "KalshiFlow/1.0 (GDELT Research)",
        })

        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                data = response.read().decode("utf-8")
                return json.loads(data)
        except urllib.error.HTTPError as e:
            logger.error(f"[gdelt_doc] HTTP {e.code}: {e.reason}")
            return {"error": f"HTTP {e.code}: {e.reason}"}
        except Exception as e:
            logger.error(f"[gdelt_doc] Fetch error: {e}")
            return {"error": str(e)}

    def _process_artlist(self, raw: Dict[str, Any], search_terms: List[str]) -> Dict[str, Any]:
        """Process article list response into structured output."""
        articles = raw.get("articles", [])

        if not articles:
            return {
                "article_count": 0,
                "source_diversity": 0,
                "articles": [],
                "search_terms": search_terms,
            }

        sources = set()
        processed_articles = []
        tones = []

        for article in articles[:self._max_records]:
            source = article.get("domain", "")
            if source:
                sources.add(source)

            tone = article.get("tone", 0)
            if isinstance(tone, (int, float)):
                tones.append(tone)

            processed_articles.append({
                "url": article.get("url", ""),
                "title": article.get("title", ""),
                "source": source,
                "seendate": article.get("seendate", ""),
                "language": article.get("language", ""),
                "source_country": article.get("sourcecountry", ""),
                "tone": round(tone, 2) if isinstance(tone, (int, float)) else 0,
                "socialimage": article.get("socialimage", ""),
            })

        # Tone summary
        positive_count = sum(1 for t in tones if t > 1.5)
        negative_count = sum(1 for t in tones if t < -1.5)
        neutral_count = len(tones) - positive_count - negative_count
        avg_tone = sum(tones) / len(tones) if tones else 0.0

        return {
            "article_count": len(processed_articles),
            "source_diversity": len(sources),
            "tone_summary": {
                "avg_tone": round(avg_tone, 2),
                "positive_count": positive_count,
                "negative_count": negative_count,
                "neutral_count": neutral_count,
            },
            "articles": processed_articles[:20],
            "search_terms": search_terms,
        }

    def _process_timeline(self, raw: Dict[str, Any], search_terms: List[str]) -> Dict[str, Any]:
        """Process timeline response into structured output."""
        timeline = raw.get("timeline", [])
        if not timeline:
            return {"timeline": [], "search_terms": search_terms}

        # timeline is a list of series, each with a "data" array of {date, value}
        all_points = []
        for series in timeline:
            series_data = series.get("data", [])
            for point in series_data:
                all_points.append({
                    "date": point.get("date", ""),
                    "value": point.get("value", 0),
                })

        return {
            "timeline": all_points,
            "search_terms": search_terms,
            "total_points": len(all_points),
        }

    async def search_articles(
        self,
        search_terms: List[str],
        timespan: Optional[str] = None,
        tone_filter: Optional[str] = None,
        source_country: Optional[str] = None,
        max_records: Optional[int] = None,
        sort: str = "datedesc",
    ) -> Dict[str, Any]:
        """
        Search GDELT for articles matching the given terms.

        Args:
            search_terms: List of terms to search for
            timespan: Time window (e.g., "4h", "1d", "2w"). Default: 4h
            tone_filter: Optional: "positive", "negative", or None
            source_country: Optional: ISO country code filter
            max_records: Max articles (default: 75, max: 250)
            sort: Sort order: "datedesc", "dateasc", "tonedesc", "toneasc"

        Returns:
            Dict with article_count, source_diversity, tone_summary, articles
        """
        if not search_terms:
            return {"error": "No search terms provided", "article_count": 0}

        timespan = timespan or self._default_timespan
        max_records = min(max_records or self._max_records, 250)

        cache_key = self._cache_key(
            self._build_query_string(search_terms, tone_filter, source_country),
            "artlist", timespan,
        )
        cached = self._get_cached(cache_key)
        if cached is not None:
            cached["_cached"] = True
            return cached

        query = self._build_query_string(search_terms, tone_filter, source_country)
        params = {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "timespan": timespan,
            "maxrecords": str(max_records),
            "sort": sort,
        }

        self._query_count += 1
        raw = await asyncio.to_thread(self._fetch_sync, params)

        if "error" in raw:
            return {"error": raw["error"], "article_count": 0, "search_terms": search_terms}

        result = self._process_artlist(raw, search_terms)
        result["timespan"] = timespan
        result["query_count"] = self._query_count
        self._set_cache(cache_key, result)
        return result

    async def get_volume_timeline(
        self,
        search_terms: List[str],
        timespan: Optional[str] = None,
        tone_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get coverage volume timeline for the given search terms.

        Useful for detecting trending topics and surge patterns.

        Args:
            search_terms: List of terms to search for
            timespan: Time window (e.g., "4h", "1d", "2w"). Default: 4h
            tone_filter: Optional: "positive", "negative", or None

        Returns:
            Dict with timeline data points (date, value)
        """
        if not search_terms:
            return {"error": "No search terms provided", "timeline": []}

        timespan = timespan or self._default_timespan

        cache_key = self._cache_key(
            self._build_query_string(search_terms, tone_filter),
            "timelinevol", timespan,
        )
        cached = self._get_cached(cache_key)
        if cached is not None:
            cached["_cached"] = True
            return cached

        query = self._build_query_string(search_terms, tone_filter)
        params = {
            "query": query,
            "mode": "timelinevol",
            "format": "json",
            "timespan": timespan,
        }

        self._query_count += 1
        raw = await asyncio.to_thread(self._fetch_sync, params)

        if "error" in raw:
            return {"error": raw["error"], "timeline": [], "search_terms": search_terms}

        result = self._process_timeline(raw, search_terms)
        result["timespan"] = timespan
        result["query_count"] = self._query_count
        self._set_cache(cache_key, result)
        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            "query_count": self._query_count,
            "cache_entries": len(self._cache),
            "cache_ttl_seconds": self._cache_ttl_seconds,
        }
