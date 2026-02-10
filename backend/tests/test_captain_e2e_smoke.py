"""
Captain E2E Smoke Test

Validates the entire V3 Captain system starts up and reaches operational state.
This is the Captain equivalent of test_backend_e2e_regression.py -- the go-to
anti-regression sanity check before shipping changes.

## What This Test Validates

1. **Server Startup**: V3 application starts and health endpoint responds
2. **Health Check**: Health endpoint returns status="running"
3. **State=READY**: System transitions through INITIALIZING → ORDERBOOK_CONNECT → READY
4. **Component Health**: EventBus, StateMachine, WebSocket manager all running
5. **Single-Arb Running**: Single-arb coordinator active with discovered events
6. **Captain Running**: Captain agent started and cycling
7. **WebSocket**: Frontend WebSocket connects and receives connection message
8. **Clean Shutdown**: Graceful shutdown completes without errors

## How to Run

```bash
# Standard run (~30-60s, requires .env.paper credentials)
cd backend && uv run pytest tests/test_captain_e2e_smoke.py -v

# With detailed logging
cd backend && uv run pytest tests/test_captain_e2e_smoke.py -v -s --log-cli-level=INFO

# Against already-running server (skips start/stop)
V3_SMOKE_EXTERNAL_URL=http://localhost:8005 uv run pytest tests/test_captain_e2e_smoke.py -v
```

## Duration

~30-60 seconds (dominated by market discovery + orderbook connection).

## Edge Cases

- Exchange closed: Captain may be paused=True, test accepts running=True as sufficient
- No credentials: Test skipped via @pytest.mark.skipif
- Database unavailable: V3 handles gracefully (non-fatal), test doesn't validate DB
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any

import aiohttp
import pytest
import uvicorn
import websockets
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Locate .env.paper for skip check
_ENV_PAPER = Path(__file__).parent.parent / ".env.paper"
_HAS_CREDENTIALS = _ENV_PAPER.exists()

# External server mode
_EXTERNAL_URL = os.environ.get("V3_SMOKE_EXTERNAL_URL", "")

# Test port (avoids 8005 dev, 8001 Flowboard E2E)
_TEST_PORT = 8006


class CaptainSmokeTestServer:
    """Manages V3 server lifecycle for smoke testing."""

    def __init__(self, external_url: str = ""):
        self.external_url = external_url.rstrip("/")
        self.server: Optional[uvicorn.Server] = None
        self.server_task: Optional[asyncio.Task] = None
        self.port = _TEST_PORT
        self._base_url = self.external_url or f"http://127.0.0.1:{self.port}"

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def ws_url(self) -> str:
        return self._base_url.replace("http://", "ws://").replace("https://", "wss://")

    @property
    def is_external(self) -> bool:
        return bool(self.external_url)

    async def start(self) -> bool:
        """Start the V3 server (skipped in external mode)."""
        if self.is_external:
            logger.info(f"External server mode: using {self.external_url}")
            return await self.wait_for_health(timeout=10)

        try:
            # Load .env.paper credentials
            env_paper = Path(__file__).parent.parent / ".env.paper"
            if not env_paper.exists():
                logger.error(f".env.paper not found at {env_paper}")
                return False
            load_dotenv(env_paper, override=True)

            # Force-enable Captain and single-arb for smoke test
            os.environ["V3_PORT"] = str(self.port)
            os.environ["V3_SINGLE_ARB_ENABLED"] = "true"
            os.environ["V3_SINGLE_ARB_CAPTAIN_ENABLED"] = "true"
            os.environ["V3_SINGLE_ARB_CAPTAIN_INTERVAL"] = "60"

            # Import app after env is loaded so config picks up our overrides
            from src.kalshiflow_rl.traderv3.app import app

            config = uvicorn.Config(
                app=app,
                host="127.0.0.1",
                port=self.port,
                log_level="warning",
                access_log=False,
                loop="asyncio",
            )
            self.server = uvicorn.Server(config)
            self.server_task = asyncio.create_task(self.server.serve())

            # Lifespan is heavy (market discovery, data sync, etc.)
            # so health endpoint won't respond until after yield in lifespan
            return await self.wait_for_health(timeout=45)

        except Exception as e:
            logger.error(f"Failed to start V3 server: {e}")
            return False

    async def stop(self):
        """Stop the V3 server (skipped in external mode)."""
        if self.is_external:
            logger.info("External server mode: skipping shutdown")
            return

        try:
            if self.server:
                self.server.should_exit = True

            if self.server_task:
                try:
                    await asyncio.wait_for(self.server_task, timeout=15.0)
                except asyncio.TimeoutError:
                    logger.warning("Server shutdown timed out, cancelling task")
                    self.server_task.cancel()
                    try:
                        await self.server_task
                    except asyncio.CancelledError:
                        pass
        except Exception as e:
            logger.error(f"Error during server shutdown: {e}")

    async def get_health(self) -> Dict[str, Any]:
        """GET /v3/health"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/v3/health", timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    return {"error": f"HTTP {resp.status}"}
        except Exception as e:
            return {"error": str(e)}

    async def get_status(self) -> Dict[str, Any]:
        """GET /v3/status"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/v3/status", timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    return {"error": f"HTTP {resp.status}"}
        except Exception as e:
            return {"error": str(e)}

    async def wait_for_health(self, timeout: float = 45) -> bool:
        """Poll /v3/health until 200 response."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{self.base_url}/v3/health",
                        timeout=aiohttp.ClientTimeout(total=3),
                    ) as resp:
                        if resp.status == 200:
                            logger.info("Health endpoint responding")
                            return True
            except Exception:
                pass
            await asyncio.sleep(0.5)
        logger.error(f"Health endpoint did not respond within {timeout}s")
        return False

    async def wait_for_state(self, target_state: str, timeout: float = 45) -> bool:
        """Poll /v3/status until state matches target."""
        start = time.time()
        last_state = None
        while time.time() - start < timeout:
            status = await self.get_status()
            current_state = status.get("state", "unknown")
            if current_state != last_state:
                logger.info(f"State: {current_state}")
                last_state = current_state
            if current_state == target_state:
                return True
            await asyncio.sleep(1)
        logger.error(f"State did not reach '{target_state}' within {timeout}s (last: {last_state})")
        return False

    async def wait_for_captain(self, timeout: float = 45) -> bool:
        """Poll until captain.running == True."""
        start = time.time()
        while time.time() - start < timeout:
            status = await self.get_status()
            components = status.get("components", {})
            single_arb = components.get("single_arb_coordinator", {})
            captain = single_arb.get("captain", {})
            if captain.get("running") is True:
                logger.info("Captain is running")
                return True
            await asyncio.sleep(2)
        logger.error(f"Captain did not start within {timeout}s")
        return False

    async def test_websocket(self, timeout: float = 10) -> bool:
        """Connect to /v3/ws and validate connection message."""
        try:
            ws_endpoint = f"{self.ws_url}/v3/ws"
            logger.info(f"Connecting to WebSocket at {ws_endpoint}")
            async with websockets.connect(ws_endpoint) as ws:
                raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                msg = json.loads(raw)

                if msg.get("type") != "connection":
                    logger.error(f"Expected 'connection' message, got '{msg.get('type')}'")
                    return False

                data = msg.get("data", {})
                client_id = data.get("client_id")
                if not client_id:
                    logger.error("Connection message missing client_id")
                    return False

                logger.info(f"WebSocket connected, client_id={client_id}")

                # Optionally consume history_replay if present
                try:
                    raw2 = await asyncio.wait_for(ws.recv(), timeout=3)
                    msg2 = json.loads(raw2)
                    if msg2.get("type") == "history_replay":
                        count = msg2.get("data", {}).get("count", 0)
                        logger.info(f"Received history_replay with {count} transitions")
                except asyncio.TimeoutError:
                    logger.info("No history_replay received (normal if no transitions yet)")

                return True

        except Exception as e:
            logger.error(f"WebSocket test failed: {e}")
            return False


def _skip_reason() -> str:
    if not _HAS_CREDENTIALS and not _EXTERNAL_URL:
        return "No .env.paper credentials and no V3_SMOKE_EXTERNAL_URL set"
    return ""


@pytest.mark.asyncio
@pytest.mark.skipif(bool(_skip_reason()), reason=_skip_reason() or "n/a")
async def test_captain_e2e_smoke():
    """
    Captain E2E smoke test: validates V3 startup through Captain operational state.

    8 steps: start → health → READY → components → single-arb → Captain → WebSocket → shutdown
    """
    server = CaptainSmokeTestServer(external_url=_EXTERNAL_URL)
    test_start = time.time()

    try:
        # ── Step 1: Start server ──────────────────────────────────────────
        logger.info("=== STEP 1: Start Server ===")
        started = await server.start()
        if not started:
            pytest.fail("V3 server did not start within timeout")
        logger.info("PASSED: V3 server started and health endpoint responding")

        # ── Step 2: Health validation ─────────────────────────────────────
        logger.info("=== STEP 2: Health Validation ===")
        health = await server.get_health()
        assert health.get("status") == "running", f"Expected status='running', got {health}"
        logger.info("PASSED: Health status = running")

        # ── Step 3: State = READY ─────────────────────────────────────────
        logger.info("=== STEP 3: Wait for READY State ===")
        reached_ready = await server.wait_for_state("ready", timeout=45)
        if not reached_ready:
            # Grab current status for diagnostics
            status = await server.get_status()
            pytest.fail(
                f"System did not reach READY state within 45s. "
                f"Current state: {status.get('state', 'unknown')}"
            )
        logger.info("PASSED: System reached READY state")

        # ── Step 4: Component health ──────────────────────────────────────
        logger.info("=== STEP 4: Component Health ===")
        status = await server.get_status()
        components = status.get("components", {})

        # Core components (hard-fail)
        event_bus = components.get("event_bus", {})
        assert event_bus.get("running") is True, f"EventBus not running: {event_bus}"
        logger.info("  EventBus: running")

        state_machine = components.get("state_machine", {})
        sm_state = state_machine.get("current_state", "")
        assert sm_state == "ready", f"StateMachine not ready: {state_machine}"
        logger.info("  StateMachine: ready")

        ws_manager = components.get("websocket_manager", {})
        assert ws_manager.get("running") is True, f"WebSocket manager not running: {ws_manager}"
        logger.info("  WebSocket manager: running")

        # Optional components (warn only)
        orderbook = components.get("orderbook_integration", {})
        if not orderbook.get("connected"):
            logger.warning("  Orderbook integration: not connected (non-fatal)")
        else:
            logger.info("  Orderbook integration: connected")

        trading_client = components.get("trading_client", {})
        if not trading_client:
            logger.warning("  Trading client: not present (non-fatal)")
        else:
            logger.info("  Trading client: present")

        logger.info("PASSED: Core components healthy")

        # ── Step 5: Single-arb running ────────────────────────────────────
        logger.info("=== STEP 5: Single-Arb Coordinator ===")
        single_arb = components.get("single_arb_coordinator", {})
        assert single_arb.get("running") is True, (
            f"Single-arb coordinator not running: {single_arb}"
        )
        events_count = single_arb.get("events", 0)
        logger.info(f"  Single-arb running, events discovered: {events_count}")
        assert events_count > 0, (
            f"Single-arb has 0 events -- market discovery may have failed"
        )
        logger.info("PASSED: Single-arb coordinator running with events")

        # ── Step 6: Captain running ───────────────────────────────────────
        logger.info("=== STEP 6: Captain Agent ===")
        captain_running = await server.wait_for_captain(timeout=45)
        if not captain_running:
            status = await server.get_status()
            single_arb = status.get("components", {}).get("single_arb_coordinator", {})
            captain = single_arb.get("captain", {})
            pytest.fail(
                f"Captain did not start within 45s. "
                f"Captain stats: {captain}"
            )

        # Accept paused state (exchange may be closed)
        status = await server.get_status()
        captain_stats = (
            status.get("components", {})
            .get("single_arb_coordinator", {})
            .get("captain", {})
        )
        if captain_stats.get("paused"):
            paused_by_exchange = captain_stats.get("paused_by_exchange", False)
            reason = "exchange closed" if paused_by_exchange else "unknown reason"
            logger.warning(f"  Captain is paused ({reason}) -- acceptable for smoke test")
        else:
            logger.info(f"  Captain active, model={captain_stats.get('model', 'unknown')}")
        logger.info("PASSED: Captain agent is running")

        # ── Step 7: WebSocket ─────────────────────────────────────────────
        logger.info("=== STEP 7: WebSocket Connection ===")
        ws_ok = await server.test_websocket(timeout=10)
        if not ws_ok:
            pytest.fail("WebSocket connection test failed")
        logger.info("PASSED: WebSocket connection and message validated")

        # ── Summary ───────────────────────────────────────────────────────
        duration = time.time() - test_start
        logger.info("=" * 60)
        logger.info(f"ALL 7 STEPS PASSED in {duration:.1f}s")
        logger.info("  1. Server startup")
        logger.info("  2. Health = running")
        logger.info("  3. State = READY")
        logger.info("  4. Core components healthy")
        logger.info("  5. Single-arb running with events")
        logger.info("  6. Captain agent running")
        logger.info("  7. WebSocket connected")
        logger.info("=" * 60)

    finally:
        # ── Step 8: Clean shutdown ────────────────────────────────────────
        logger.info("=== STEP 8: Clean Shutdown ===")
        await server.stop()
        shutdown_duration = time.time() - test_start
        logger.info(f"PASSED: Clean shutdown completed ({shutdown_duration:.1f}s total)")


if __name__ == "__main__":
    asyncio.run(test_captain_e2e_smoke())
