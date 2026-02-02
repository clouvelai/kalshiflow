"""Trader process control - start, stop, restart the V3 trader."""

import asyncio
import logging
import signal
import subprocess
from typing import Optional

import aiohttp

from .config import RALPHConfig

logger = logging.getLogger("ralph.trader_control")


class TraderControl:
    """Manages the V3 trader process lifecycle."""

    def __init__(self, config: RALPHConfig):
        self._config = config
        self._process: Optional[asyncio.subprocess.Process] = None

    async def is_running(self) -> bool:
        """Check if the trader process is reachable."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self._config.trader_health_url,
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    return resp.status == 200
        except Exception:
            return False

    async def stop(self) -> bool:
        """Stop the trader process.

        Finds PID by port, sends SIGTERM, waits, then SIGKILL if needed.
        """
        logger.info("[ralph.trader_control] Stopping trader on port %d...", self._config.trader_port)

        pid = await self._find_pid_by_port()
        if pid is None:
            logger.info("[ralph.trader_control] No process found on port %d", self._config.trader_port)
            return True

        try:
            # Send SIGTERM
            import os
            os.kill(pid, signal.SIGTERM)
            logger.info("[ralph.trader_control] Sent SIGTERM to PID %d", pid)

            # Wait up to 15s for graceful shutdown
            for _ in range(30):
                await asyncio.sleep(0.5)
                if not await self._pid_exists(pid):
                    logger.info("[ralph.trader_control] Process %d stopped gracefully", pid)
                    return True

            # Force kill
            logger.warning("[ralph.trader_control] Process %d didn't stop, sending SIGKILL", pid)
            os.kill(pid, signal.SIGKILL)
            await asyncio.sleep(1)
            return not await self._pid_exists(pid)

        except ProcessLookupError:
            return True  # Already dead
        except Exception as e:
            logger.error("[ralph.trader_control] Error stopping process: %s", e)
            return False

    async def start(self) -> bool:
        """Start the V3 trader process and wait for it to become healthy."""
        logger.info("[ralph.trader_control] Starting V3 trader...")

        if await self.is_running():
            logger.info("[ralph.trader_control] Trader already running")
            return True

        try:
            # Use the run-v3.sh script which handles env setup
            script_path = self._config.repo_root / "scripts" / "run-v3.sh"
            if script_path.exists():
                self._process = await asyncio.create_subprocess_exec(
                    str(script_path),
                    self._config.environment,  # paper/live
                    "discovery",
                    "10",
                    cwd=str(self._config.repo_root),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            else:
                # Fallback: run uvicorn directly
                self._process = await asyncio.create_subprocess_exec(
                    "uv", "run", "uvicorn",
                    "kalshiflow_rl.traderv3.app:app",
                    "--host", "0.0.0.0",
                    "--port", str(self._config.trader_port),
                    cwd=str(self._config.repo_root / "backend"),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env={"ENVIRONMENT": self._config.environment},
                )

            # Wait for healthy (up to 30s)
            for i in range(60):
                await asyncio.sleep(0.5)
                if await self.is_running():
                    logger.info("[ralph.trader_control] Trader started and healthy after %.1fs", (i + 1) * 0.5)
                    return True

            logger.error("[ralph.trader_control] Trader didn't become healthy within 30s")
            return False

        except Exception as e:
            logger.error("[ralph.trader_control] Failed to start trader: %s", e)
            return False

    async def restart(self) -> bool:
        """Stop, pause, then start the trader."""
        logger.info("[ralph.trader_control] Restarting trader...")
        await self.stop()
        await asyncio.sleep(2)
        return await self.start()

    async def _find_pid_by_port(self) -> Optional[int]:
        """Find the PID of the process listening on the trader port."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "lsof", "-ti", f":{self._config.trader_port}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            output = stdout.decode().strip()
            if output:
                # May return multiple PIDs; take the first
                return int(output.split("\n")[0])
        except Exception as e:
            logger.debug("[ralph.trader_control] lsof error: %s", e)
        return None

    @staticmethod
    async def _pid_exists(pid: int) -> bool:
        """Check if a process with the given PID exists."""
        import os
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True  # Exists but we can't signal it
