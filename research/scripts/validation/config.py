"""
Strategy Configuration Schema and YAML Parser.

This module defines the schema for strategy configurations and provides
utilities for loading and validating YAML config files.

Key Responsibilities:
    1. Define dataclasses for strategy configuration
    2. Parse and validate YAML config files
    3. Support the signal condition DSL (field, operator, value)
    4. Provide type-safe access to all configuration fields

Example YAML Config:
    strategy:
      name: "H123 - Reverse Line Movement NO"
      hypothesis_id: "H123"
      action: "bet_no"

    signal:
      conditions:
        - field: "yes_trade_ratio"
          operator: ">"
          value: 0.70
        - field: "price_dropped"
          operator: "=="
          value: true
      entry_price_field: "no_price"

    validation:
      mode: "full"
      min_markets: 50
      p_threshold: 0.001
      bucket_size: 5
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml


class Operator(str, Enum):
    """Supported comparison operators for signal conditions."""
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="
    EQ = "=="
    NE = "!="
    IN = "in"
    NOT_IN = "not_in"


class BetAction(str, Enum):
    """Supported bet actions."""
    BET_YES = "bet_yes"
    BET_NO = "bet_no"


class ValidationMode(str, Enum):
    """Validation modes."""
    LSD = "lsd"  # Quick screening (~2 seconds)
    FULL = "full"  # Rigorous validation (~30 seconds)


@dataclass
class SignalCondition:
    """
    A single signal condition in the strategy.

    Examples:
        - field: "yes_trade_ratio", operator: ">", value: 0.70
        - field: "price_dropped", operator: "==", value: True
        - field: "trade_count", operator: ">=", value: 5
        - field: "category", operator: "in", value: ["KXNBA", "KXNFL"]
    """
    field: str
    value: Any
    operator: str = "=="

    def __post_init__(self):
        """Validate the operator."""
        valid_ops = [op.value for op in Operator]
        if self.operator not in valid_ops:
            raise ValueError(f"Invalid operator '{self.operator}'. Must be one of: {valid_ops}")

    def evaluate(self, row_value: Any) -> bool:
        """
        Evaluate this condition against a row value.

        Args:
            row_value: The value from the data to compare against

        Returns:
            True if the condition is satisfied
        """
        op = self.operator

        if op == ">":
            return row_value > self.value
        elif op == ">=":
            return row_value >= self.value
        elif op == "<":
            return row_value < self.value
        elif op == "<=":
            return row_value <= self.value
        elif op == "==":
            return row_value == self.value
        elif op == "!=":
            return row_value != self.value
        elif op == "in":
            return row_value in self.value
        elif op == "not_in":
            return row_value not in self.value
        else:
            raise ValueError(f"Unknown operator: {op}")


@dataclass
class SignalConfig:
    """Configuration for the strategy signal."""
    conditions: List[SignalCondition]
    entry_price_field: str = "no_price"  # Field to use for breakeven calculation

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SignalConfig":
        """Create SignalConfig from a dictionary."""
        conditions = []
        for cond in data.get("conditions", []):
            conditions.append(SignalCondition(
                field=cond["field"],
                operator=cond.get("operator", "=="),
                value=cond["value"]
            ))

        return cls(
            conditions=conditions,
            entry_price_field=data.get("entry_price_field", "no_price")
        )


@dataclass
class ValidationConfig:
    """Configuration for validation parameters."""
    mode: ValidationMode = ValidationMode.FULL
    min_markets: int = 50  # Minimum markets for valid analysis
    p_threshold: float = 0.001  # P-value threshold for statistical significance
    bucket_size: int = 5  # Price bucket size in cents
    min_bucket_markets: int = 5  # Minimum markets per bucket
    bootstrap_iterations: int = 1000  # Bootstrap iterations for CI
    temporal_splits: int = 4  # Number of temporal splits (quarters)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationConfig":
        """Create ValidationConfig from a dictionary."""
        mode_str = data.get("mode", "full")
        mode = ValidationMode.LSD if mode_str.lower() == "lsd" else ValidationMode.FULL

        return cls(
            mode=mode,
            min_markets=data.get("min_markets", 50),
            p_threshold=data.get("p_threshold", 0.001),
            bucket_size=data.get("bucket_size", 5),
            min_bucket_markets=data.get("min_bucket_markets", 5),
            bootstrap_iterations=data.get("bootstrap_iterations", 1000),
            temporal_splits=data.get("temporal_splits", 4)
        )


@dataclass
class ParameterSensitivityConfig:
    """Configuration for parameter sensitivity analysis."""
    enabled: bool = False
    parameters: Dict[str, List[Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParameterSensitivityConfig":
        """Create ParameterSensitivityConfig from a dictionary."""
        return cls(
            enabled=data.get("enabled", False),
            parameters=data.get("parameters", {})
        )


@dataclass
class StrategyConfig:
    """
    Complete strategy configuration.

    This is the main configuration object that contains all settings
    needed to validate a trading strategy.
    """
    name: str
    hypothesis_id: str
    action: BetAction
    signal: SignalConfig
    validation: ValidationConfig
    parameter_sensitivity: Optional[ParameterSensitivityConfig] = None
    description: Optional[str] = None
    config_path: Optional[Path] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any], config_path: Optional[Path] = None) -> "StrategyConfig":
        """Create StrategyConfig from a dictionary."""
        strategy_data = data.get("strategy", {})

        # Parse action
        action_str = strategy_data.get("action", "bet_no")
        action = BetAction.BET_YES if action_str.lower() == "bet_yes" else BetAction.BET_NO

        # Parse signal config
        signal = SignalConfig.from_dict(data.get("signal", {}))

        # Parse validation config
        validation = ValidationConfig.from_dict(data.get("validation", {}))

        # Parse parameter sensitivity (optional)
        param_sens_data = data.get("parameter_sensitivity")
        param_sens = ParameterSensitivityConfig.from_dict(param_sens_data) if param_sens_data else None

        return cls(
            name=strategy_data.get("name", "Unnamed Strategy"),
            hypothesis_id=strategy_data.get("hypothesis_id", "UNKNOWN"),
            action=action,
            signal=signal,
            validation=validation,
            parameter_sensitivity=param_sens,
            description=strategy_data.get("description"),
            config_path=config_path
        )

    @property
    def bet_side(self) -> str:
        """Return the side to bet on (yes/no)."""
        return "yes" if self.action == BetAction.BET_YES else "no"

    def get_condition(self, field_name: str) -> Optional[SignalCondition]:
        """Get a signal condition by field name."""
        for cond in self.signal.conditions:
            if cond.field == field_name:
                return cond
        return None


def load_config(config_path: Union[str, Path]) -> StrategyConfig:
    """
    Load a strategy configuration from a YAML file.

    Args:
        config_path: Path to the YAML config file

    Returns:
        Parsed StrategyConfig object

    Raises:
        FileNotFoundError: If the config file doesn't exist
        ValueError: If the config is invalid
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if not data:
        raise ValueError(f"Empty config file: {path}")

    config = StrategyConfig.from_dict(data, config_path=path)

    # Validate required fields
    if not config.signal.conditions:
        raise ValueError("Config must have at least one signal condition")

    return config


def list_configs(configs_dir: Union[str, Path]) -> List[Path]:
    """
    List all strategy config files in a directory.

    Args:
        configs_dir: Directory containing YAML config files

    Returns:
        List of paths to config files
    """
    dir_path = Path(configs_dir)

    if not dir_path.exists():
        return []

    return sorted(dir_path.glob("*.yaml")) + sorted(dir_path.glob("*.yml"))
