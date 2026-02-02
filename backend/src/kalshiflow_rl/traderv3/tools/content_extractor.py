"""
Content Extractor - Detects content type and extracts text from URLs.

Handles:
- Text posts: Returns selftext directly
- Video posts: Delegates to VideoTranscriber (Whisper API)
- Link posts: Fetches HTML and uses LLM to extract article text
- Image posts: Returns empty (not processable as text)
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING
from urllib.parse import urlparse

import aiohttp

if TYPE_CHECKING:
    from .video_transcriber import VideoTranscriber

logger = logging.getLogger("kalshiflow_rl.traderv3.tools.content_extractor")

# Image URL patterns (skip these)
IMAGE_PATTERNS = [
    r"i\.redd\.it",
    r"imgur\.com",
    r"i\.imgur\.com",
    r"\.(jpg|jpeg|png|gif|webp|bmp)(\?|$)",
    r"preview\.redd\.it",
    r"pbs\.twimg\.com",
]
IMAGE_REGEX = re.compile("|".join(IMAGE_PATTERNS), re.IGNORECASE)

# Social media patterns (skip article extraction)
SOCIAL_PATTERNS = [
    r"twitter\.com(?!/.*video)",  # Twitter without video
    r"x\.com(?!/.*video)",  # X without video
    r"facebook\.com",
    r"instagram\.com",
    r"tiktok\.com(?!/.*video)",  # TikTok without video
    r"reddit\.com/r/",  # Reddit links to other subreddits
]
SOCIAL_REGEX = re.compile("|".join(SOCIAL_PATTERNS), re.IGNORECASE)


def extract_source_domain(url: str) -> str:
    """
    Extract domain from URL (e.g., 'youtube.com', 'foxnews.com').

    Args:
        url: The URL to extract domain from

    Returns:
        Domain string or 'reddit.com' if no URL or 'unknown' on error
    """
    if not url:
        return "reddit.com"
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        # Remove www. prefix
        if domain.startswith("www."):
            domain = domain[4:]
        return domain or "reddit.com"
    except Exception:
        return "unknown"


@dataclass
class ContentExtractorConfig:
    """Configuration for content extraction."""

    # LLM settings for article extraction
    extraction_model: str = "gpt-4o-mini"
    openai_api_key: Optional[str] = None  # Falls back to OPENAI_API_KEY env var

    # Content limits
    max_html_chars: int = 50_000  # Max HTML to send to LLM
    max_output_chars: int = 10_000  # Max extracted text to return

    # HTTP settings
    request_timeout: float = 15.0
    user_agent: str = "Mozilla/5.0 (compatible; KalshiFlow/1.0; +https://kalshiflow.com)"

    # Feature flags
    article_extraction_enabled: bool = True
    video_transcription_enabled: bool = True


@dataclass
class ExtractedContent:
    """Result of content extraction."""

    url: str
    content_type: str  # "text", "video", "link", "image", "social", "unknown"
    text: str = ""
    source: str = ""  # "selftext", "whisper", "llm_extraction"
    success: bool = False
    error: Optional[str] = None
    source_domain: str = ""  # Extracted domain (e.g., "youtube.com", "foxnews.com")


class ContentExtractor:
    """
    Content extraction orchestrator.

    Detects content type from URL and delegates to appropriate extractor:
    - Video URLs -> VideoTranscriber (Whisper API)
    - News/Article URLs -> LLM-based HTML extraction
    - Image/Social URLs -> Skip (not text-processable)
    """

    def __init__(
        self,
        config: Optional[ContentExtractorConfig] = None,
        video_transcriber: Optional["VideoTranscriber"] = None,
    ):
        self._config = config or ContentExtractorConfig()
        self._video_transcriber = video_transcriber

        # HTTP session (lazy init)
        self._session: Optional[aiohttp.ClientSession] = None

        # Stats
        self._extractions_attempted = 0
        self._extractions_successful = 0
        self._by_type = {"text": 0, "video": 0, "link": 0, "image": 0, "social": 0}

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self._config.request_timeout)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={"User-Agent": self._config.user_agent},
            )
        return self._session

    async def close(self) -> None:
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    def detect_content_type(self, url: str, selftext: str = "") -> str:
        """
        Detect content type from URL and selftext.

        Returns: "text", "video", "link", "image", "social", or "unknown"
        """
        if not url:
            return "text" if selftext else "unknown"

        # Check for image URLs first
        if IMAGE_REGEX.search(url):
            return "image"

        # Check for social media (non-video)
        if SOCIAL_REGEX.search(url):
            return "social"

        # Check for video URLs
        if self._video_transcriber and self._video_transcriber.is_video_url(url):
            return "video"

        # If URL points back to reddit.com and has selftext, it's a text post
        if "reddit.com" in url and selftext:
            return "text"

        # Otherwise assume it's a link to external content
        return "link"

    @staticmethod
    def _normalize_url(url: str) -> str:
        """
        Normalize URL, converting relative Reddit paths to absolute URLs.

        PRAW sometimes returns relative paths like '/r/AskReddit/comments/...'
        which fail aiohttp requests. This prepends the Reddit base URL.
        """
        if not url:
            return url
        # Relative Reddit path (starts with /r/)
        if url.startswith("/r/"):
            return f"https://www.reddit.com{url}"
        # Other relative paths starting with /
        if url.startswith("/") and not url.startswith("//"):
            return f"https://www.reddit.com{url}"
        return url

    async def extract(self, url: str, selftext: str = "") -> ExtractedContent:
        """
        Extract text content from URL.

        Args:
            url: Post URL
            selftext: Post body text (for text posts)

        Returns:
            ExtractedContent with extracted text or error
        """
        self._extractions_attempted += 1

        # Normalize relative URLs (e.g., /r/AskReddit/...) to absolute
        url = self._normalize_url(url)

        content_type = self.detect_content_type(url, selftext)
        self._by_type[content_type] = self._by_type.get(content_type, 0) + 1

        # Extract domain for all content types
        source_domain = extract_source_domain(url)

        try:
            if content_type == "text":
                # Text post - just use selftext
                return ExtractedContent(
                    url=url,
                    content_type="text",
                    text=selftext[: self._config.max_output_chars] if selftext else "",
                    source="selftext",
                    success=bool(selftext),
                    source_domain=source_domain,
                )

            elif content_type == "video":
                # Video post - transcribe with Whisper
                result = await self._extract_video(url)
                result.source_domain = source_domain
                return result

            elif content_type == "link":
                # Link post - fetch and extract article
                result = await self._extract_article(url)
                result.source_domain = source_domain
                return result

            elif content_type == "image":
                # Image post - skip
                return ExtractedContent(
                    url=url,
                    content_type="image",
                    success=False,
                    error="Image content cannot be extracted as text",
                    source_domain=source_domain,
                )

            elif content_type == "social":
                # Social media - skip
                return ExtractedContent(
                    url=url,
                    content_type="social",
                    success=False,
                    error="Social media links not supported for extraction",
                    source_domain=source_domain,
                )

            else:
                return ExtractedContent(
                    url=url,
                    content_type="unknown",
                    success=False,
                    error="Unknown content type",
                    source_domain=source_domain,
                )

        except Exception as e:
            logger.error(f"[content_extractor] Error extracting {url}: {e}")
            return ExtractedContent(
                url=url,
                content_type=content_type,
                success=False,
                error=str(e),
                source_domain=source_domain,
            )

    async def _extract_video(self, url: str) -> ExtractedContent:
        """Extract text from video via transcription."""
        if not self._video_transcriber:
            return ExtractedContent(
                url=url,
                content_type="video",
                success=False,
                error="Video transcriber not available",
            )

        if not self._config.video_transcription_enabled:
            return ExtractedContent(
                url=url,
                content_type="video",
                success=False,
                error="Video transcription disabled",
            )

        result = await self._video_transcriber.transcribe(url)

        if result.success:
            self._extractions_successful += 1
            return ExtractedContent(
                url=url,
                content_type="video",
                text=result.transcript[: self._config.max_output_chars],
                source="whisper",
                success=True,
            )
        else:
            return ExtractedContent(
                url=url,
                content_type="video",
                success=False,
                error=result.error,
            )

    async def _extract_article(self, url: str) -> ExtractedContent:
        """Extract article text from URL using LLM."""
        if not self._config.article_extraction_enabled:
            return ExtractedContent(
                url=url,
                content_type="link",
                success=False,
                error="Article extraction disabled",
            )

        # Fetch HTML
        html = await self._fetch_html(url)
        if not html:
            return ExtractedContent(
                url=url,
                content_type="link",
                success=False,
                error="Failed to fetch HTML",
            )

        # Extract article text with LLM
        article_text = await self._llm_extract_article(html)

        if article_text:
            self._extractions_successful += 1
            return ExtractedContent(
                url=url,
                content_type="link",
                text=article_text[: self._config.max_output_chars],
                source="llm_extraction",
                success=True,
            )
        else:
            return ExtractedContent(
                url=url,
                content_type="link",
                success=False,
                error="LLM failed to extract article text",
            )

    async def _fetch_html(self, url: str) -> Optional[str]:
        """Fetch HTML content from URL."""
        try:
            session = await self._ensure_session()
            async with session.get(url, allow_redirects=True) as response:
                if response.status != 200:
                    logger.warning(
                        f"[content_extractor] HTTP {response.status} for {url}"
                    )
                    return None

                # Check content type
                content_type = response.headers.get("content-type", "")
                if "text/html" not in content_type and "text/plain" not in content_type:
                    logger.debug(
                        f"[content_extractor] Non-HTML content: {content_type}"
                    )
                    return None

                html = await response.text()
                return html[: self._config.max_html_chars]

        except asyncio.TimeoutError:
            logger.warning(f"[content_extractor] Timeout fetching {url}")
            return None
        except Exception as e:
            logger.warning(f"[content_extractor] Fetch error for {url}: {e}")
            return None

    async def _llm_extract_article(self, html: str) -> Optional[str]:
        """Use LLM to extract article text from HTML."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            logger.error("[content_extractor] openai not installed")
            return None

        api_key = self._config.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("[content_extractor] OPENAI_API_KEY not set")
            return None

        # Truncate HTML to fit in context
        html_truncated = html[: self._config.max_html_chars]

        prompt = f"""Extract the main article text from this HTML page.

Rules:
1. Return ONLY the article body text - the main content
2. Remove all navigation, headers, footers, ads, sidebars, comments
3. Remove HTML tags, keeping just the text
4. Keep the article structure (paragraphs separated by newlines)
5. If this is not a news article or blog post, return "NOT_AN_ARTICLE"
6. If you cannot extract meaningful content, return "NO_CONTENT"

HTML:
{html_truncated}

Article text:"""

        try:
            client = AsyncOpenAI(api_key=api_key)
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=self._config.extraction_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4000,
                    temperature=0,
                ),
                timeout=30.0,
            )

            result = response.choices[0].message.content.strip()

            # Check for failure indicators
            if result in ("NOT_AN_ARTICLE", "NO_CONTENT", ""):
                return None

            # Basic validation - should have some substance
            if len(result) < 100:
                logger.debug(f"[content_extractor] Extracted text too short: {len(result)} chars")
                return None

            return result

        except asyncio.TimeoutError:
            logger.warning("[content_extractor] LLM extraction timeout")
            return None
        except Exception as e:
            logger.error(f"[content_extractor] LLM extraction error: {e}")
            return None

    def get_stats(self) -> dict:
        """Get extraction statistics."""
        return {
            "extractions_attempted": self._extractions_attempted,
            "extractions_successful": self._extractions_successful,
            "success_rate": (
                round(self._extractions_successful / self._extractions_attempted, 2)
                if self._extractions_attempted > 0
                else 0.0
            ),
            "by_type": self._by_type.copy(),
        }
