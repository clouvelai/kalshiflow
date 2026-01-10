"""
Truth Social API Client - Custom implementation with Cloudflare bypass.

This replaces the truthbrush library with a direct implementation using curl_cffi
with the correct browser impersonation profile (chrome136) that bypasses Cloudflare.

The truthbrush library uses chrome123 which is blocked by Cloudflare as of late 2025.
"""

import logging
import os
import re
import time
from typing import Any, Dict, Iterator, List, Optional
from datetime import datetime, timezone
from dateutil import parser as date_parse

try:
    from curl_cffi import requests
except ImportError:
    requests = None  # Will raise error on use

logger = logging.getLogger("kalshiflow_rl.traderv3.services.truth_social_api")

# API Constants
BASE_URL = "https://truthsocial.com"
API_BASE_URL = "https://truthsocial.com/api"

# OAuth client credentials (from truthbrush/official app)
CLIENT_ID = "9X1Fdd-pxNsAgEDNi_SfhJWi8T-vLuV2WVzKIbkTCw4"
CLIENT_SECRET = "ozF8jzI4968oTKFkEnsBC-UbLPCdrSv0MkXGQu2o_-M"

# Browser impersonation profile that works with Cloudflare
# Tested profiles that work: chrome136, chrome133a, safari184, safari180, firefox135, firefox133
# Profiles that don't work: chrome123 (truthbrush default), chrome131, edge101
BROWSER_PROFILE = "chrome136"

# Request delay to avoid rate limiting
REQUEST_DELAY_SECONDS = 0.5


class TruthSocialApiError(Exception):
    """Base exception for Truth Social API errors."""
    pass


class TruthSocialAuthError(TruthSocialApiError):
    """Authentication error."""
    pass


class TruthSocialRateLimitError(TruthSocialApiError):
    """Rate limit error."""
    pass


class TruthSocialApi:
    """
    Truth Social API client with Cloudflare bypass.

    Uses curl_cffi with chrome136 browser impersonation to bypass Cloudflare's
    bot detection. Maintains session cookies for authenticated requests.
    """

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        token: Optional[str] = None,
    ):
        """
        Initialize Truth Social API client.

        Args:
            username: Truth Social username (falls back to TRUTHSOCIAL_USERNAME env)
            password: Truth Social password (falls back to TRUTHSOCIAL_PASSWORD env)
            token: Optional pre-existing access token
        """
        if requests is None:
            raise ImportError("curl_cffi is required for Truth Social API. Install with: pip install curl_cffi")

        self._username = username or os.getenv("TRUTHSOCIAL_USERNAME")
        self._password = password or os.getenv("TRUTHSOCIAL_PASSWORD")
        self._token: Optional[str] = token or os.getenv("TRUTHSOCIAL_TOKEN")

        # Session with persistent cookies
        self._session: Optional[requests.Session] = None
        self._session_established = False

        # Rate limit tracking
        self._ratelimit_remaining: Optional[int] = None
        self._ratelimit_reset: Optional[datetime] = None

        # Request timing
        self._last_request_time: float = 0

    def _get_session(self) -> requests.Session:
        """Get or create session with cookies."""
        if self._session is None:
            self._session = requests.Session()
        return self._session

    def _establish_session(self) -> None:
        """Visit homepage to establish session cookies."""
        if self._session_established:
            return

        logger.debug("Establishing Truth Social session...")
        session = self._get_session()

        try:
            resp = session.get(
                BASE_URL,
                impersonate=BROWSER_PROFILE,
                timeout=15,
                headers={
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                    "Accept-Encoding": "gzip, deflate, br",
                    "Connection": "keep-alive",
                }
            )
            resp.raise_for_status()
            self._session_established = True
            logger.debug(f"Session established, {len(session.cookies)} cookies set")
        except Exception as e:
            logger.warning(f"Failed to establish session: {e}")

    def _ensure_authenticated(self) -> None:
        """Ensure we have a valid access token."""
        if self._token:
            return

        if not self._username or not self._password:
            raise TruthSocialAuthError(
                "No token or credentials provided. "
                "Set TRUTHSOCIAL_USERNAME and TRUTHSOCIAL_PASSWORD env vars."
            )

        self._token = self._authenticate(self._username, self._password)

    def _authenticate(self, username: str, password: str) -> str:
        """
        Authenticate and get access token.

        Args:
            username: Truth Social username
            password: Truth Social password

        Returns:
            Access token string
        """
        self._establish_session()
        session = self._get_session()

        payload = {
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "grant_type": "password",
            "username": username,
            "password": password,
            "redirect_uri": "urn:ietf:wg:oauth:2.0:oob",
            "scope": "read",
        }

        try:
            resp = session.post(
                f"{BASE_URL}/oauth/token",
                json=payload,
                impersonate=BROWSER_PROFILE,
                timeout=15,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "Origin": BASE_URL,
                    "Referer": f"{BASE_URL}/",
                }
            )
            resp.raise_for_status()
        except Exception as e:
            logger.error(f"Authentication request failed: {e}")
            raise TruthSocialAuthError(f"Authentication failed: {e}")

        try:
            data = resp.json()
            token = data.get("access_token")
            if not token:
                raise TruthSocialAuthError(f"No access_token in response: {data}")

            logger.info(f"Successfully authenticated as {username}")
            return token
        except Exception as e:
            logger.error(f"Failed to parse auth response: {e}")
            raise TruthSocialAuthError(f"Failed to parse auth response: {e}")

    def _wait_for_rate_limit(self) -> None:
        """Wait if approaching rate limit."""
        if self._ratelimit_remaining is not None and self._ratelimit_remaining <= 50:
            if self._ratelimit_reset:
                now = datetime.now(timezone.utc)
                wait_seconds = (self._ratelimit_reset - now).total_seconds()
                if wait_seconds > 0:
                    logger.warning(f"Rate limit approaching, waiting {wait_seconds:.1f}s...")
                    time.sleep(min(wait_seconds + 1, 60))  # Cap at 60s

    def _check_rate_limit(self, resp: requests.Response) -> None:
        """Update rate limit tracking from response headers."""
        if "x-ratelimit-remaining" in resp.headers:
            self._ratelimit_remaining = int(resp.headers["x-ratelimit-remaining"])
        if "x-ratelimit-reset" in resp.headers:
            try:
                self._ratelimit_reset = date_parse.parse(resp.headers["x-ratelimit-reset"])
            except Exception:
                pass

    def _throttle(self) -> None:
        """Enforce minimum delay between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_DELAY_SECONDS:
            time.sleep(REQUEST_DELAY_SECONDS - elapsed)
        self._last_request_time = time.time()

    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Any:
        """
        Make authenticated GET request to API.

        Args:
            endpoint: API endpoint (e.g., "/v1/accounts/lookup")
            params: Optional query parameters

        Returns:
            JSON response data
        """
        self._ensure_authenticated()
        self._establish_session()
        self._wait_for_rate_limit()
        self._throttle()

        session = self._get_session()
        url = f"{API_BASE_URL}{endpoint}"

        try:
            resp = session.get(
                url,
                params=params,
                impersonate=BROWSER_PROFILE,
                timeout=15,
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Accept": "application/json",
                    "Origin": BASE_URL,
                    "Referer": f"{BASE_URL}/",
                }
            )

            self._check_rate_limit(resp)

            if resp.status_code == 403:
                # Check for Cloudflare block
                if "cf-ray" in resp.headers and ("Just a moment" in resp.text or "cf-browser-verification" in resp.text):
                    raise TruthSocialApiError("Cloudflare blocked the request")
                raise TruthSocialApiError(f"API returned 403: {resp.text[:200]}")

            if resp.status_code == 429:
                raise TruthSocialRateLimitError("Rate limit exceeded")

            resp.raise_for_status()
            return resp.json()

        except requests.RequestsError as e:
            logger.error(f"API request failed: {e}")
            raise TruthSocialApiError(f"Request failed: {e}")

    def _get_paginated(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        max_pages: int = 10,
    ) -> Iterator[List[Dict]]:
        """
        Make paginated GET request following Link headers.

        Args:
            endpoint: API endpoint
            params: Optional query parameters
            max_pages: Maximum pages to fetch

        Yields:
            List of items from each page
        """
        self._ensure_authenticated()
        self._establish_session()

        next_url = f"{API_BASE_URL}{endpoint}"
        pages_fetched = 0

        while next_url and pages_fetched < max_pages:
            self._wait_for_rate_limit()
            self._throttle()

            session = self._get_session()

            try:
                resp = session.get(
                    next_url,
                    params=params if pages_fetched == 0 else None,
                    impersonate=BROWSER_PROFILE,
                    timeout=15,
                    headers={
                        "Authorization": f"Bearer {self._token}",
                        "Accept": "application/json",
                        "Origin": BASE_URL,
                        "Referer": f"{BASE_URL}/",
                    }
                )

                self._check_rate_limit(resp)

                if resp.status_code == 403:
                    raise TruthSocialApiError("API returned 403")

                resp.raise_for_status()

                # Parse Link header for next page
                link_header = resp.headers.get("Link", "")
                next_url = None
                for link in link_header.split(","):
                    parts = link.split(";")
                    if len(parts) == 2 and 'rel="next"' in parts[1]:
                        next_url = parts[0].strip().strip("<>")
                        break

                yield resp.json()
                pages_fetched += 1

            except requests.RequestsError as e:
                logger.error(f"Paginated request failed: {e}")
                raise TruthSocialApiError(f"Request failed: {e}")

    # ========== Public API Methods ==========

    def lookup(self, username: str) -> Optional[Dict]:
        """
        Look up a user by username.

        Args:
            username: User handle (without @)

        Returns:
            User dict with id, username, display_name, etc.
        """
        return self._get("/v1/accounts/lookup", params={"acct": username})

    def get_user_statuses(
        self,
        user_id: str,
        limit: int = 40,
        exclude_replies: bool = True,
        since_id: Optional[str] = None,
        max_id: Optional[str] = None,
    ) -> List[Dict]:
        """
        Get statuses (posts) for a user by ID.

        Args:
            user_id: User ID (from lookup)
            limit: Maximum posts to fetch
            exclude_replies: Whether to exclude replies
            since_id: Only fetch posts newer than this ID
            max_id: Only fetch posts older than this ID

        Returns:
            List of status dicts
        """
        params = {"limit": min(limit, 40), "exclude_replies": str(exclude_replies).lower()}
        if since_id:
            params["since_id"] = since_id
        if max_id:
            params["max_id"] = max_id

        return self._get(f"/v1/accounts/{user_id}/statuses", params=params)

    def pull_statuses(
        self,
        username: str,
        limit: int = 40,
        exclude_replies: bool = True,
        created_after: Optional[datetime] = None,
        since_id: Optional[str] = None,
    ) -> Iterator[Dict]:
        """
        Pull statuses for a user by username, with pagination.

        This is the primary method for fetching a user's posts.

        Args:
            username: User handle (without @)
            limit: Maximum total posts to fetch
            exclude_replies: Whether to exclude replies
            created_after: Only fetch posts created after this time
            since_id: Only fetch posts newer than this ID

        Yields:
            Status dicts in reverse chronological order
        """
        # Look up user ID
        user = self.lookup(username)
        if not user:
            logger.warning(f"User not found: {username}")
            return

        user_id = user.get("id")
        if not user_id:
            logger.warning(f"No user ID for: {username}")
            return

        posts_fetched = 0
        max_id = None

        while posts_fetched < limit:
            batch_limit = min(40, limit - posts_fetched)

            params = {
                "limit": batch_limit,
                "exclude_replies": str(exclude_replies).lower(),
            }
            if since_id:
                params["since_id"] = since_id
            if max_id:
                params["max_id"] = max_id

            try:
                statuses = self._get(f"/v1/accounts/{user_id}/statuses", params=params)
            except TruthSocialApiError as e:
                logger.error(f"Error fetching statuses for {username}: {e}")
                break

            if not statuses or not isinstance(statuses, list):
                break

            # Sort by ID (reverse chronological)
            statuses = sorted(statuses, key=lambda s: s.get("id", ""), reverse=True)

            for status in statuses:
                # Check time filter
                if created_after:
                    try:
                        post_time = date_parse.parse(status.get("created_at", ""))
                        if post_time.replace(tzinfo=timezone.utc) <= created_after.replace(tzinfo=timezone.utc):
                            return  # Stop iteration
                    except Exception:
                        pass

                yield status
                posts_fetched += 1

                if posts_fetched >= limit:
                    return

            # Set max_id for next page (oldest from this batch)
            max_id = statuses[-1].get("id") if statuses else None

            if not max_id or len(statuses) < batch_limit:
                break  # No more pages

    def user_following(
        self,
        username: str,
        limit: int = 1000,
    ) -> Iterator[Dict]:
        """
        Get list of users that a user follows.

        Args:
            username: User handle (without @)
            limit: Maximum users to fetch

        Yields:
            User dicts for followed accounts
        """
        user = self.lookup(username)
        if not user:
            return

        user_id = user.get("id")
        if not user_id:
            return

        count = 0
        for page in self._get_paginated(f"/v1/accounts/{user_id}/following"):
            if not isinstance(page, list):
                continue
            for u in page:
                yield u
                count += 1
                if count >= limit:
                    return

    def user_followers(
        self,
        username: str,
        limit: int = 1000,
    ) -> Iterator[Dict]:
        """
        Get list of users following a user.

        Args:
            username: User handle (without @)
            limit: Maximum users to fetch

        Yields:
            User dicts for followers
        """
        user = self.lookup(username)
        if not user:
            return

        user_id = user.get("id")
        if not user_id:
            return

        count = 0
        for page in self._get_paginated(f"/v1/accounts/{user_id}/followers"):
            if not isinstance(page, list):
                continue
            for u in page:
                yield u
                count += 1
                if count >= limit:
                    return

    def trending(self, limit: int = 20) -> List[Dict]:
        """
        Get trending posts.

        Args:
            limit: Maximum posts to fetch (max 20)

        Returns:
            List of trending status dicts
        """
        return self._get(f"/v1/truth/trending/truths", params={"limit": min(limit, 20)})

    def tags(self) -> List[Dict]:
        """
        Get trending hashtags.

        Returns:
            List of trending tag dicts
        """
        return self._get("/v1/trends")

    def search(
        self,
        query: str,
        search_type: str = "statuses",
        limit: int = 40,
    ) -> Dict:
        """
        Search for users, statuses, or hashtags.

        Args:
            query: Search query
            search_type: One of "accounts", "statuses", "hashtags"
            limit: Maximum results

        Returns:
            Search results dict with accounts/statuses/hashtags keys
        """
        return self._get("/v2/search", params={
            "q": query,
            "type": search_type,
            "limit": limit,
            "resolve": True,
        })

    def get_access_token(self) -> Optional[str]:
        """Get the current access token (authenticating if needed)."""
        self._ensure_authenticated()
        return self._token


# Convenience function for quick usage
def create_api(
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> TruthSocialApi:
    """
    Create a Truth Social API client.

    Args:
        username: Truth Social username (or set TRUTHSOCIAL_USERNAME env)
        password: Truth Social password (or set TRUTHSOCIAL_PASSWORD env)

    Returns:
        Configured TruthSocialApi instance
    """
    return TruthSocialApi(username=username, password=password)
