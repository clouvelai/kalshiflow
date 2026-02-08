"""Shared Kalshi authentication utilities for WebSocket clients.

Provides reusable setup/cleanup for KalshiAuth instances that require
a temporary PEM file from KALSHI_PRIVATE_KEY_CONTENT.
"""

import logging
import os
import tempfile
from typing import Optional, Tuple

from kalshiflow.auth import KalshiAuth
from ...config import config

logger = logging.getLogger("kalshiflow_rl.traderv3.clients.auth_utils")


def format_private_key(raw_content: str) -> str:
    """Format a private key string into valid PEM format.

    Handles escaped newlines from dotenv and missing PEM headers.
    """
    if not raw_content.startswith('-----BEGIN'):
        return f"-----BEGIN PRIVATE KEY-----\n{raw_content}\n-----END PRIVATE KEY-----"

    formatted = raw_content.replace('\\n', '\n')

    # Handle case where newlines might be lost
    if '\n' not in formatted and '-----BEGIN' in formatted:
        begin_marker = '-----BEGIN'
        end_marker = '-----END'
        begin_idx = formatted.find(begin_marker)
        end_idx = formatted.find(end_marker)

        if begin_idx != -1 and end_idx != -1:
            begin_end = formatted.find('-----', begin_idx + len(begin_marker))
            if begin_end != -1:
                begin_end += 5
                content = formatted[begin_end:end_idx].strip()
                formatted = (
                    formatted[:begin_end] + '\n' +
                    content + '\n' +
                    formatted[end_idx:]
                )

    return formatted


def setup_kalshi_auth(prefix: str = "kalshi_key_") -> Tuple[KalshiAuth, str]:
    """Create a KalshiAuth instance with a temporary PEM key file.

    Args:
        prefix: Prefix for the temporary key file name.

    Returns:
        Tuple of (KalshiAuth instance, temp file path).

    Raises:
        ValueError: If required config values are missing.
        RuntimeError: If auth initialization fails.
    """
    if not config.KALSHI_API_KEY_ID:
        raise ValueError("KALSHI_API_KEY_ID not configured")

    if not config.KALSHI_PRIVATE_KEY_CONTENT:
        raise ValueError("KALSHI_PRIVATE_KEY_CONTENT not configured")

    temp_fd, temp_path = tempfile.mkstemp(suffix='.pem', prefix=prefix)
    try:
        with os.fdopen(temp_fd, 'w') as temp_file:
            formatted_key = format_private_key(config.KALSHI_PRIVATE_KEY_CONTENT)
            temp_file.write(formatted_key)

        auth = KalshiAuth(
            api_key_id=config.KALSHI_API_KEY_ID,
            private_key_path=temp_path,
        )
        return auth, temp_path

    except Exception:
        # Clean up on failure
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


def cleanup_kalshi_auth(temp_key_file: Optional[str]) -> None:
    """Remove a temporary PEM key file.

    Args:
        temp_key_file: Path to the temp file, or None (no-op).
    """
    if temp_key_file:
        try:
            os.unlink(temp_key_file)
            logger.debug("Cleaned up temporary key file")
        except Exception as e:
            logger.warning(f"Failed to clean up temp key file: {e}")
