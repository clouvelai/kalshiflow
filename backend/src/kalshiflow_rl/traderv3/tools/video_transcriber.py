"""
Video Transcriber - Downloads videos and transcribes using OpenAI Whisper API.

Pipeline: URL -> yt-dlp download -> ffmpeg audio extraction -> Whisper API -> text
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("kalshiflow_rl.traderv3.tools.video_transcriber")

# Video URL patterns
VIDEO_PATTERNS = [
    r"v\.redd\.it",  # Reddit hosted video
    r"reddit\.com/.*\.(mp4|webm)",
    r"youtube\.com/watch",
    r"youtu\.be/",
    r"vimeo\.com/\d+",
    r"dailymotion\.com/video",
    r"twitter\.com/.*/video",
    r"x\.com/.*/video",
    r"tiktok\.com/",
    r"streamable\.com/",
]

VIDEO_REGEX = re.compile("|".join(VIDEO_PATTERNS), re.IGNORECASE)


@dataclass
class VideoTranscriberConfig:
    """Configuration for video transcription."""

    # OpenAI Whisper settings
    whisper_model: str = "whisper-1"
    openai_api_key: Optional[str] = None  # Falls back to OPENAI_API_KEY env var

    # Video constraints
    max_duration_seconds: int = 300  # 5 minutes max
    max_file_size_mb: int = 25  # Whisper API limit is 25MB

    # Timeouts
    download_timeout: float = 60.0
    ffmpeg_timeout: float = 30.0
    transcription_timeout: float = 120.0

    # Budget tracking (~$0.006 per minute)
    daily_budget_minutes: float = 60.0  # ~$0.36/day max
    budget_reset_hour: int = 0  # Reset at midnight UTC

    # Temp directory (None = system temp)
    temp_dir: Optional[str] = None
    cleanup_temp_files: bool = True

    # yt-dlp settings
    yt_dlp_format: str = "bestaudio/best"  # Audio-only is faster/smaller
    yt_dlp_quiet: bool = True


@dataclass
class TranscriptionResult:
    """Result of video transcription."""

    url: str
    transcript: str = ""
    duration_seconds: float = 0.0
    language: Optional[str] = None
    word_count: int = 0
    success: bool = False
    error: Optional[str] = None
    processing_time_seconds: float = 0.0


class VideoTranscriber:
    """
    Async video transcription using OpenAI Whisper API.

    Pipeline:
    1. yt-dlp downloads video/audio from URL
    2. ffmpeg extracts/converts to audio (MP3, 16kHz, mono)
    3. OpenAI Whisper API transcribes audio to text
    4. Cleanup temp files
    """

    def __init__(self, config: Optional[VideoTranscriberConfig] = None):
        self._config = config or VideoTranscriberConfig()

        # Temp directory for downloads
        self._temp_dir = Path(self._config.temp_dir or tempfile.gettempdir())

        # Budget tracking
        self._minutes_used_today: float = 0.0
        self._last_budget_reset: float = 0.0

        # Concurrency limit
        self._semaphore = asyncio.Semaphore(2)  # Max 2 concurrent transcriptions

        # Stats
        self._total_transcriptions = 0
        self._total_minutes = 0.0
        self._total_errors = 0

        # Check ffmpeg availability at init time (once) to avoid wasting bandwidth
        # downloading videos that can never be processed
        self._ffmpeg_available = self._check_ffmpeg()
        if not self._ffmpeg_available:
            logger.warning(
                "[video_transcriber] ffmpeg not found at startup. "
                "Video transcription is disabled. Install: brew install ffmpeg"
            )

    @staticmethod
    def _check_ffmpeg() -> bool:
        """Check if ffmpeg is available on the system PATH."""
        import shutil
        return shutil.which("ffmpeg") is not None

    def is_video_url(self, url: str) -> bool:
        """Check if URL is a video that can be transcribed."""
        if not url:
            return False
        return VIDEO_REGEX.search(url) is not None

    async def transcribe(self, url: str) -> TranscriptionResult:
        """
        Download and transcribe video from URL.

        Args:
            url: Video URL (YouTube, Reddit, etc.)

        Returns:
            TranscriptionResult with transcript or error
        """
        start_time = time.time()

        # Check budget
        self._maybe_reset_budget()
        if self._minutes_used_today >= self._config.daily_budget_minutes:
            return TranscriptionResult(
                url=url,
                success=False,
                error=f"Daily budget exhausted ({self._config.daily_budget_minutes} min)",
            )

        # Acquire semaphore for concurrency limit
        async with self._semaphore:
            try:
                return await self._transcribe_impl(url, start_time)
            except Exception as e:
                self._total_errors += 1
                logger.error(f"[video_transcriber] Error transcribing {url}: {e}")
                return TranscriptionResult(
                    url=url,
                    success=False,
                    error=str(e),
                    processing_time_seconds=time.time() - start_time,
                )

    async def _transcribe_impl(
        self, url: str, start_time: float
    ) -> TranscriptionResult:
        """Implementation of transcription pipeline."""
        # Early exit: skip download entirely if ffmpeg is unavailable
        # (avoids wasting bandwidth on videos we cannot extract audio from)
        if not self._ffmpeg_available:
            return TranscriptionResult(
                url=url,
                success=False,
                error="ffmpeg not available - video transcription disabled",
                processing_time_seconds=time.time() - start_time,
            )

        temp_video: Optional[Path] = None
        temp_audio: Optional[Path] = None

        try:
            # Step 1: Download video with yt-dlp
            logger.info(f"[video_transcriber] Downloading: {url}")
            temp_video, duration = await self._download_video(url)

            if not temp_video or not temp_video.exists():
                return TranscriptionResult(
                    url=url,
                    success=False,
                    error="Download failed - no file created",
                    processing_time_seconds=time.time() - start_time,
                )

            # Check duration limit
            if duration and duration > self._config.max_duration_seconds:
                return TranscriptionResult(
                    url=url,
                    success=False,
                    error=f"Video too long ({duration:.0f}s > {self._config.max_duration_seconds}s)",
                    processing_time_seconds=time.time() - start_time,
                )

            # Step 2: Extract audio with ffmpeg
            logger.info(f"[video_transcriber] Extracting audio...")
            temp_audio = await self._extract_audio(temp_video)

            if not temp_audio or not temp_audio.exists():
                return TranscriptionResult(
                    url=url,
                    success=False,
                    error="Audio extraction failed",
                    processing_time_seconds=time.time() - start_time,
                )

            # Check file size limit
            file_size_mb = temp_audio.stat().st_size / (1024 * 1024)
            if file_size_mb > self._config.max_file_size_mb:
                return TranscriptionResult(
                    url=url,
                    success=False,
                    error=f"Audio file too large ({file_size_mb:.1f}MB > {self._config.max_file_size_mb}MB)",
                    processing_time_seconds=time.time() - start_time,
                )

            # Step 3: Transcribe with Whisper API
            logger.info(f"[video_transcriber] Transcribing with Whisper API...")
            transcript, language = await self._transcribe_audio(temp_audio)

            # Update budget tracking
            actual_duration = duration or 60.0  # Estimate if unknown
            self._minutes_used_today += actual_duration / 60.0
            self._total_minutes += actual_duration / 60.0
            self._total_transcriptions += 1

            processing_time = time.time() - start_time
            word_count = len(transcript.split()) if transcript else 0

            logger.info(
                f"[video_transcriber] Transcribed {word_count} words in {processing_time:.1f}s"
            )

            return TranscriptionResult(
                url=url,
                transcript=transcript,
                duration_seconds=actual_duration,
                language=language,
                word_count=word_count,
                success=True,
                processing_time_seconds=processing_time,
            )

        finally:
            # Cleanup temp files
            if self._config.cleanup_temp_files:
                if temp_video and temp_video.exists():
                    try:
                        temp_video.unlink()
                    except Exception:
                        pass
                if temp_audio and temp_audio.exists():
                    try:
                        temp_audio.unlink()
                    except Exception:
                        pass

    async def _download_video(self, url: str) -> tuple[Optional[Path], Optional[float]]:
        """
        Download video using yt-dlp.

        Returns:
            Tuple of (path to downloaded file, duration in seconds)
        """
        try:
            import yt_dlp
        except ImportError:
            logger.error("[video_transcriber] yt-dlp not installed. Run: uv add yt-dlp")
            return None, None

        # Generate unique temp file path
        file_id = uuid.uuid4().hex[:8]
        temp_path = self._temp_dir / f"video_{file_id}"

        ydl_opts = {
            "format": self._config.yt_dlp_format,
            "outtmpl": str(temp_path) + ".%(ext)s",
            "quiet": self._config.yt_dlp_quiet,
            "no_warnings": self._config.yt_dlp_quiet,
            "socket_timeout": self._config.download_timeout,
            "max_filesize": self._config.max_file_size_mb * 1024 * 1024,
            # Don't download if too long
            "match_filter": yt_dlp.utils.match_filter_func(
                f"duration <= {self._config.max_duration_seconds}"
            ),
        }

        def download():
            """Blocking download (runs in thread pool)."""
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                if info is None:
                    return None, None

                duration = info.get("duration")
                # Find the downloaded file
                ext = info.get("ext", "webm")
                actual_path = Path(str(temp_path) + f".{ext}")
                if actual_path.exists():
                    return actual_path, duration

                # Try common extensions
                for try_ext in ["webm", "mp4", "m4a", "mp3", "opus"]:
                    try_path = Path(str(temp_path) + f".{try_ext}")
                    if try_path.exists():
                        return try_path, duration

                return None, duration

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(download),
                timeout=self._config.download_timeout,
            )
            return result
        except asyncio.TimeoutError:
            logger.warning(f"[video_transcriber] Download timeout for {url}")
            return None, None
        except Exception as e:
            logger.warning(f"[video_transcriber] Download error for {url}: {e}")
            return None, None

    async def _extract_audio(self, video_path: Path) -> Optional[Path]:
        """
        Extract audio from video using ffmpeg.

        Outputs: MP3, 16kHz sample rate, mono, 64kbps
        (Whisper API prefers 16kHz sample rate)
        """
        audio_path = video_path.with_suffix(".mp3")

        cmd = [
            "ffmpeg",
            "-i",
            str(video_path),
            "-vn",  # No video
            "-acodec",
            "libmp3lame",
            "-ar",
            "16000",  # 16kHz sample rate
            "-ac",
            "1",  # Mono
            "-b:a",
            "64k",  # Low bitrate to reduce file size
            "-y",  # Overwrite output
            str(audio_path),
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(
                process.wait(),
                timeout=self._config.ffmpeg_timeout,
            )

            if process.returncode == 0 and audio_path.exists():
                return audio_path

            logger.warning(
                f"[video_transcriber] ffmpeg returned {process.returncode}"
            )
            return None

        except asyncio.TimeoutError:
            logger.warning("[video_transcriber] ffmpeg timeout")
            return None
        except FileNotFoundError:
            logger.error(
                "[video_transcriber] ffmpeg not found. Install: brew install ffmpeg"
            )
            return None
        except Exception as e:
            logger.warning(f"[video_transcriber] ffmpeg error: {e}")
            return None

    async def _transcribe_audio(self, audio_path: Path) -> tuple[str, Optional[str]]:
        """
        Transcribe audio file using OpenAI Whisper API.

        Returns:
            Tuple of (transcript text, detected language)
        """
        try:
            from openai import AsyncOpenAI
        except ImportError:
            logger.error("[video_transcriber] openai not installed")
            return "", None

        api_key = self._config.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("[video_transcriber] OPENAI_API_KEY not set")
            return "", None

        client = AsyncOpenAI(api_key=api_key)

        try:
            with open(audio_path, "rb") as audio_file:
                response = await asyncio.wait_for(
                    client.audio.transcriptions.create(
                        model=self._config.whisper_model,
                        file=audio_file,
                        response_format="verbose_json",  # Get language info
                    ),
                    timeout=self._config.transcription_timeout,
                )

            # Extract transcript and language
            transcript = response.text if hasattr(response, "text") else str(response)
            language = getattr(response, "language", None)

            return transcript.strip(), language

        except asyncio.TimeoutError:
            logger.warning("[video_transcriber] Whisper API timeout")
            return "", None
        except Exception as e:
            logger.error(f"[video_transcriber] Whisper API error: {e}")
            return "", None

    def _maybe_reset_budget(self) -> None:
        """Reset daily budget if past reset hour."""
        import datetime

        now = datetime.datetime.now(datetime.timezone.utc)
        reset_time = now.replace(
            hour=self._config.budget_reset_hour, minute=0, second=0, microsecond=0
        )

        # If current time is past reset time and we haven't reset today
        if now >= reset_time and self._last_budget_reset < reset_time.timestamp():
            self._minutes_used_today = 0.0
            self._last_budget_reset = now.timestamp()
            logger.info("[video_transcriber] Daily budget reset")

    def get_stats(self) -> dict:
        """Get transcription statistics."""
        return {
            "total_transcriptions": self._total_transcriptions,
            "total_minutes": round(self._total_minutes, 2),
            "total_errors": self._total_errors,
            "minutes_used_today": round(self._minutes_used_today, 2),
            "daily_budget_minutes": self._config.daily_budget_minutes,
            "budget_remaining": round(
                self._config.daily_budget_minutes - self._minutes_used_today, 2
            ),
        }
