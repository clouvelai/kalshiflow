"""Unit tests for V3Config dataclass and environment loading."""

import os
from unittest.mock import patch

import pytest

from kalshiflow_rl.traderv3.config.environment import V3Config
from .conftest import make_config


# ---------------------------------------------------------------------------
# Minimum required env vars for from_env()
# ---------------------------------------------------------------------------
REQUIRED_ENV = {
    "KALSHI_API_URL": "https://demo-api.kalshi.co/trade-api/v2",
    "KALSHI_WS_URL": "wss://demo-api.kalshi.co/trade-api/ws/v2",
    "KALSHI_API_KEY_ID": "test-key-id",
    "KALSHI_PRIVATE_KEY_CONTENT": "test-private-key",
    "V3_SUBACCOUNT": "0",
}


# ===========================================================================
# TestV3ConfigFromEnv
# ===========================================================================


class TestV3ConfigFromEnv:
    """Tests for V3Config.from_env() class method."""

    def test_from_env_happy_path(self):
        with patch.dict(os.environ, REQUIRED_ENV, clear=True):
            cfg = V3Config.from_env()
        assert cfg.api_url == REQUIRED_ENV["KALSHI_API_URL"]
        assert cfg.ws_url == REQUIRED_ENV["KALSHI_WS_URL"]
        assert cfg.api_key_id == REQUIRED_ENV["KALSHI_API_KEY_ID"]
        assert cfg.private_key_content == REQUIRED_ENV["KALSHI_PRIVATE_KEY_CONTENT"]
        assert cfg.port == 8005
        assert cfg.max_markets == 10

    def test_from_env_missing_required(self):
        env = {k: v for k, v in REQUIRED_ENV.items() if k != "KALSHI_API_URL"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="KALSHI_API_URL"):
                V3Config.from_env()

    def test_from_env_demo_url_detects_paper(self):
        env = {**REQUIRED_ENV, "KALSHI_WS_URL": "wss://demo-api.kalshi.co/ws/v2"}
        with patch.dict(os.environ, env, clear=True):
            cfg = V3Config.from_env()
        assert cfg.trading_mode == "paper"

    def test_from_env_comma_list_parsing(self):
        env = {**REQUIRED_ENV, "V3_MARKET_TICKERS": "A,B,C"}
        with patch.dict(os.environ, env, clear=True):
            cfg = V3Config.from_env()
        assert cfg.market_tickers == ["A", "B", "C"]

    def test_from_env_empty_tickers(self):
        with patch.dict(os.environ, REQUIRED_ENV, clear=True):
            cfg = V3Config.from_env()
        assert cfg.market_tickers == []

    def test_from_env_tavily_disabled_without_key(self):
        with patch.dict(os.environ, REQUIRED_ENV, clear=True):
            cfg = V3Config.from_env()
        assert cfg.tavily_enabled is False

    def test_from_env_missing_subaccount_fails_hard(self):
        env = {k: v for k, v in REQUIRED_ENV.items() if k != "V3_SUBACCOUNT"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="V3_SUBACCOUNT"):
                V3Config.from_env()

    def test_from_env_subaccount_parsed(self):
        env = {**REQUIRED_ENV, "V3_SUBACCOUNT": "3"}
        with patch.dict(os.environ, env, clear=True):
            cfg = V3Config.from_env()
        assert cfg.subaccount == 3


# ===========================================================================
# TestV3ConfigValidate
# ===========================================================================


class TestV3ConfigValidate:
    """Tests for V3Config.validate()."""

    def test_validate_happy(self):
        cfg = make_config()
        assert cfg.validate() is True

    def test_validate_max_markets_zero(self):
        cfg = make_config(max_markets=0)
        with pytest.raises(ValueError, match="max_markets"):
            cfg.validate()

    def test_validate_sync_duration_short(self):
        cfg = make_config(sync_duration=0.5)
        with pytest.raises(ValueError, match="Sync duration"):
            cfg.validate()

    def test_validate_port_too_low(self):
        cfg = make_config(port=80)
        with pytest.raises(ValueError, match="port"):
            cfg.validate()

    def test_validate_port_too_high(self):
        cfg = make_config(port=70000)
        with pytest.raises(ValueError, match="port"):
            cfg.validate()


# ===========================================================================
# TestV3ConfigHelpers
# ===========================================================================


class TestV3ConfigHelpers:
    """Tests for V3Config helper methods."""

    def test_is_demo_environment_true(self):
        cfg = make_config(api_url="https://demo-api.kalshi.co/trade-api/v2")
        assert cfg.is_demo_environment() is True

    def test_is_demo_environment_false(self):
        cfg = make_config(api_url="https://api.kalshi.com/trade-api/v2")
        assert cfg.is_demo_environment() is False

    def test_get_environment_name_demo(self):
        cfg = make_config(api_url="https://demo-api.kalshi.co/trade-api/v2")
        assert cfg.get_environment_name() == "DEMO (Paper Trading)"

    def test_get_environment_name_prod(self):
        cfg = make_config(api_url="https://api.kalshi.com/trade-api/v2")
        assert cfg.get_environment_name() == "PRODUCTION"


# ===========================================================================
# TestV3ConfigModelTiers
# ===========================================================================


class TestV3ConfigModelTiers:
    """Tests for centralized LLM model tier configuration."""

    def test_model_tier_defaults(self):
        cfg = make_config()
        assert cfg.model_captain == "claude-sonnet-4-20250514"
        assert cfg.model_subagent == "claude-haiku-4-5-20251001"
        assert cfg.model_utility == "gemini-2.0-flash"
        assert cfg.model_embedding == "text-embedding-3-small"

    def test_model_tier_env_overrides(self):
        env = {
            **REQUIRED_ENV,
            "V3_MODEL_CAPTAIN": "gpt-4o",
            "V3_MODEL_SUBAGENT": "claude-sonnet-4-20250514",
            "V3_MODEL_UTILITY": "gpt-4o-mini",
            "V3_MODEL_EMBEDDING": "text-embedding-3-large",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = V3Config.from_env()
        assert cfg.model_captain == "gpt-4o"
        assert cfg.model_subagent == "claude-sonnet-4-20250514"
        assert cfg.model_utility == "gpt-4o-mini"
        assert cfg.model_embedding == "text-embedding-3-large"

    def test_model_tier_partial_override(self):
        env = {**REQUIRED_ENV, "V3_MODEL_CAPTAIN": "gpt-4o"}
        with patch.dict(os.environ, env, clear=True):
            cfg = V3Config.from_env()
        assert cfg.model_captain == "gpt-4o"
        # Others keep defaults
        assert cfg.model_subagent == "claude-haiku-4-5-20251001"
        assert cfg.model_utility == "gemini-2.0-flash"

    def test_deprecated_cheval_model_overrides_subagent(self):
        env = {
            **REQUIRED_ENV,
            "V3_SINGLE_ARB_CHEVAL_MODEL": "claude-sonnet-4-20250514",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = V3Config.from_env()
        # Deprecated cheval model sets single_arb_cheval_model, not model_subagent
        assert cfg.single_arb_cheval_model == "claude-sonnet-4-20250514"
        assert cfg.model_subagent == "claude-haiku-4-5-20251001"

    def test_make_config_model_overrides(self):
        cfg = make_config(model_captain="custom-model", model_utility="custom-utility")
        assert cfg.model_captain == "custom-model"
        assert cfg.model_utility == "custom-utility"
        assert cfg.model_subagent == "claude-haiku-4-5-20251001"


# ===========================================================================
# TestMentionsModelsConfigure
# ===========================================================================


class TestMentionsModelsConfigure:
    """Tests for mentions_models.configure() and tier getters."""

    def setup_method(self):
        """Reset mentions_models module state before each test."""
        from kalshiflow_rl.traderv3.single_arb import mentions_models
        self._mm = mentions_models
        # Save originals
        self._orig = {
            "captain": mentions_models._captain_model,
            "subagent": mentions_models._subagent_model,
            "utility": mentions_models._utility_model,
            "embedding": mentions_models._embedding_model,
            "configured": mentions_models._configured,
        }

    def teardown_method(self):
        """Restore mentions_models module state after each test."""
        self._mm._captain_model = self._orig["captain"]
        self._mm._subagent_model = self._orig["subagent"]
        self._mm._utility_model = self._orig["utility"]
        self._mm._embedding_model = self._orig["embedding"]
        self._mm._configured = self._orig["configured"]

    def test_configure_sets_tiers(self):
        cfg = make_config(
            model_captain="test-captain",
            model_subagent="test-subagent",
            model_utility="test-utility",
            model_embedding="test-embedding",
        )
        self._mm.configure(cfg)
        assert self._mm.get_captain_model() == "test-captain"
        assert self._mm.get_subagent_model() == "test-subagent"
        assert self._mm.get_utility_model() == "test-utility"
        assert self._mm.get_embedding_model() == "test-embedding"

    def test_configure_defaults_without_attrs(self):
        """configure() gracefully handles config without model attrs."""
        class MinimalConfig:
            pass
        self._mm.configure(MinimalConfig())
        # Should keep module defaults
        assert self._mm.get_captain_model() == self._orig["captain"]
        assert self._mm.get_subagent_model() == self._orig["subagent"]

    def test_getters_before_configure(self):
        """Getters return module-level defaults before configure() is called."""
        assert self._mm.get_captain_model() == "claude-sonnet-4-20250514"
        assert self._mm.get_subagent_model() == "claude-haiku-4-5-20251001"
        assert self._mm.get_utility_model() == "gemini-2.0-flash"
        assert self._mm.get_embedding_model() == "text-embedding-3-small"


# ===========================================================================
# TestPortfolioCacheTTL
# ===========================================================================


class TestPortfolioCacheTTL:
    """Tests for portfolio_cache_ttl config field."""

    def test_default_value(self):
        cfg = make_config()
        assert cfg.portfolio_cache_ttl == 15.0

    def test_env_override(self):
        env = {**REQUIRED_ENV, "V3_PORTFOLIO_CACHE_TTL": "30.0"}
        with patch.dict(os.environ, env, clear=True):
            cfg = V3Config.from_env()
        assert cfg.portfolio_cache_ttl == 30.0

    def test_env_override_custom(self):
        env = {**REQUIRED_ENV, "V3_PORTFOLIO_CACHE_TTL": "5.0"}
        with patch.dict(os.environ, env, clear=True):
            cfg = V3Config.from_env()
        assert cfg.portfolio_cache_ttl == 5.0

    def test_make_config_override(self):
        cfg = make_config(portfolio_cache_ttl=42.0)
        assert cfg.portfolio_cache_ttl == 42.0


# ===========================================================================
# TestCaptainSizingConfig
# ===========================================================================


class TestCaptainSizingConfig:
    """Tests for captain_* sizing configuration fields."""

    def test_defaults(self):
        cfg = make_config()
        assert cfg.captain_eb_complement_size == "100-250"
        assert cfg.captain_eb_decide_size == "50-150"
        assert cfg.captain_news_size_small == "10-25"
        assert cfg.captain_news_size_medium == "25-50"
        assert cfg.captain_news_size_large == "50-100"
        assert cfg.captain_max_contracts_per_market == 200
        assert cfg.captain_max_capital_pct_per_event == 20

    def test_env_overrides(self):
        env = {
            **REQUIRED_ENV,
            "V3_CAPTAIN_EB_COMPLEMENT_SIZE": "200-500",
            "V3_CAPTAIN_EB_DECIDE_SIZE": "100-300",
            "V3_CAPTAIN_NEWS_SIZE_SMALL": "5-15",
            "V3_CAPTAIN_NEWS_SIZE_MEDIUM": "15-30",
            "V3_CAPTAIN_NEWS_SIZE_LARGE": "30-75",
            "V3_CAPTAIN_MAX_CONTRACTS": "500",
            "V3_CAPTAIN_MAX_CAPITAL_PCT": "30",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = V3Config.from_env()
        assert cfg.captain_eb_complement_size == "200-500"
        assert cfg.captain_eb_decide_size == "100-300"
        assert cfg.captain_news_size_small == "5-15"
        assert cfg.captain_news_size_medium == "15-30"
        assert cfg.captain_news_size_large == "30-75"
        assert cfg.captain_max_contracts_per_market == 500
        assert cfg.captain_max_capital_pct_per_event == 30

    def test_make_config_overrides(self):
        cfg = make_config(
            captain_eb_complement_size="50-100",
            captain_max_contracts_per_market=100,
        )
        assert cfg.captain_eb_complement_size == "50-100"
        assert cfg.captain_max_contracts_per_market == 100
        # Others keep defaults
        assert cfg.captain_eb_decide_size == "50-150"

    def test_partial_env_override(self):
        env = {**REQUIRED_ENV, "V3_CAPTAIN_MAX_CONTRACTS": "300"}
        with patch.dict(os.environ, env, clear=True):
            cfg = V3Config.from_env()
        assert cfg.captain_max_contracts_per_market == 300
        # Others keep defaults
        assert cfg.captain_eb_complement_size == "100-250"
        assert cfg.captain_max_capital_pct_per_event == 20
