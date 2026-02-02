"""Configuration for the RALPH self-healing agent."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Set


@dataclass
class RALPHConfig:
    """Configuration with safety limits for RALPH."""

    # Paths
    repo_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[4])
    traderv3_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    log_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[4] / "ralph_logs")

    # Trader connection
    trader_host: str = "localhost"
    trader_port: int = 8005
    environment: str = "paper"

    # Claude Code session limits
    max_claude_turns: int = 15
    claude_session_timeout: int = 300  # 5 minutes
    max_lines_per_fix: int = 50

    # Safety guardrails
    max_fix_attempts: int = 3
    max_issues_per_hour: int = 5
    dedup_window: float = 300.0  # 5 minutes -- ignore duplicate issues within this window

    # Monitoring intervals (seconds)
    health_poll_interval: float = 30.0
    memory_scan_interval: float = 60.0
    process_check_interval: float = 15.0
    main_loop_interval: float = 60.0  # sleep between full RALPH cycles

    # File safety -- only allow edits within traderv3/
    allowed_edit_dirs: List[str] = field(default_factory=lambda: [
        "backend/src/kalshiflow_rl/traderv3/",
    ])
    forbidden_files: Set[str] = field(default_factory=lambda: {
        ".env", ".env.local", ".env.paper", ".env.production",
        "app.py",  # entry point -- too risky
    })

    # Git
    auto_commit_fixes: bool = True
    auto_rollback_on_failure: bool = True

    @property
    def trader_base_url(self) -> str:
        return f"http://{self.trader_host}:{self.trader_port}"

    @property
    def trader_health_url(self) -> str:
        return f"{self.trader_base_url}/v3/health"

    @property
    def trader_status_url(self) -> str:
        return f"{self.trader_base_url}/v3/status"

    @property
    def trader_ws_url(self) -> str:
        return f"ws://{self.trader_host}:{self.trader_port}/v3/ws"

    @property
    def memory_dir(self) -> Path:
        return self.traderv3_root / "deep_agent" / "memory"
