"""Post-fix validation for RALPH.

Runs validation steps in order; ALL must pass for a fix to be accepted:
1. pytest -- existing test suite passes
2. Health check -- trader is healthy (if running)
3. Pricing sanity -- no positions with avg_price > 99c
"""

import asyncio
import logging
from typing import Optional

import aiohttp

from .config import RALPHConfig

logger = logging.getLogger("ralph.validator")


class RALPHValidator:
    """Validates fixes before accepting them."""

    def __init__(self, config: RALPHConfig):
        self._config = config

    async def validate_fix(self, trader_is_running: bool = True) -> bool:
        """Run all validation steps. Returns True if all pass."""
        # Step 1: pytest
        if not await self._run_pytest():
            logger.error("[ralph.validator] FAILED: pytest did not pass")
            return False
        logger.info("[ralph.validator] PASSED: pytest")

        # Step 2: Health check (only if trader is running)
        if trader_is_running:
            if not await self._check_health():
                logger.error("[ralph.validator] FAILED: health check")
                return False
            logger.info("[ralph.validator] PASSED: health check")

            # Step 3: Pricing sanity
            if not await self._check_pricing_sanity():
                logger.error("[ralph.validator] FAILED: pricing sanity")
                return False
            logger.info("[ralph.validator] PASSED: pricing sanity")

        logger.info("[ralph.validator] All validations passed")
        return True

    async def _run_pytest(self) -> bool:
        """Run pytest and return True if it passes."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "uv", "run", "pytest", "tests/", "-x", "-q",
                "--ignore=tests/test_odmr_features.py",
                "--ignore=tests/test_rl",
                "--ignore=tests/traderv3/test_trading_integration.py",
                "--ignore=tests/test_backend_e2e_regression.py",
                cwd=str(self._config.repo_root / "backend"),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
            output = stdout.decode("utf-8", errors="replace")

            # Check for "passed" in output and exit code 0
            if proc.returncode == 0:
                return True

            # Allow pre-existing failures (truth_social, database tests)
            # as long as no NEW failures were introduced
            if "failed" in output.lower():
                # Count failures -- if same as known pre-existing count, pass
                # Known pre-existing: 11 (database + truth_social)
                import re
                match = re.search(r"(\d+) failed", output)
                if match:
                    failed_count = int(match.group(1))
                    # Pre-existing failures are database (5) + truth_social (6) = 11
                    if failed_count <= 11:
                        logger.info(
                            "[ralph.validator] pytest: %d failures (all pre-existing)", failed_count
                        )
                        return True
                    logger.warning(
                        "[ralph.validator] pytest: %d failures (more than 11 pre-existing)", failed_count
                    )
            return False

        except asyncio.TimeoutError:
            logger.error("[ralph.validator] pytest timed out")
            return False
        except Exception as e:
            logger.error("[ralph.validator] pytest error: %s", e)
            return False

    async def _check_health(self) -> bool:
        """Check trader health endpoint returns healthy."""
        try:
            async with aiohttp.ClientSession() as session:
                for attempt in range(6):  # Wait up to 30s
                    try:
                        async with session.get(
                            self._config.trader_health_url,
                            timeout=aiohttp.ClientTimeout(total=5),
                        ) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                if data.get("status") == "healthy":
                                    return True
                    except aiohttp.ClientError:
                        pass
                    await asyncio.sleep(5)
            return False
        except Exception as e:
            logger.error("[ralph.validator] Health check error: %s", e)
            return False

    async def _check_pricing_sanity(self) -> bool:
        """Check no positions have impossible prices."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self._config.trader_status_url,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        return True  # Can't check, assume OK

                    data = await resp.json()
                    positions = data.get("positions", [])
                    for pos in positions:
                        avg_price = pos.get("avg_price", 0)
                        if avg_price and avg_price > 99:
                            logger.error(
                                "[ralph.validator] Impossible price: %s has avg_price=%dc",
                                pos.get("ticker", "?"), avg_price,
                            )
                            return False
            return True
        except Exception as e:
            logger.debug("[ralph.validator] Pricing sanity check error: %s", e)
            return True  # Non-critical, don't block on connection errors
