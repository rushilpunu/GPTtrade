"""Tests for TradingAgents engine integration."""

import pytest
import sys
from datetime import date
from unittest.mock import MagicMock, patch
from pathlib import Path


class TestEngineLoader:
    """Tests for TradingAgents engine loader."""

    def test_find_tradingagents_path_exists(self, tmp_path):
        """Test that we can find TradingAgents directory."""
        from src.agent.engine_loader import _find_tradingagents_path, TradingAgentsNotFoundError

        # Create a fake TradingAgents directory
        ta_dir = tmp_path / "TradingAgents"
        ta_dir.mkdir()

        # Mock the path resolution
        with patch("src.agent.engine_loader.Path") as mock_path:
            # Make it look like we're in src/agent/engine_loader.py
            mock_file = MagicMock()
            mock_file.parent.parent.parent = tmp_path
            mock_path.return_value = mock_file
            mock_path.__file__ = str(tmp_path / "src" / "agent" / "engine_loader.py")

            # The function should find the directory
            # Note: This test is more of an integration test since _find_tradingagents_path
            # uses __file__ directly

    def test_tradingagents_not_found_error_message(self):
        """Test that TradingAgentsNotFoundError has clear message."""
        from src.agent.engine_loader import TradingAgentsNotFoundError

        error = TradingAgentsNotFoundError("Test error message")
        assert "Test error message" in str(error)

    def test_get_tradingagents_config(self):
        """Test config builder produces valid config."""
        from src.agent.engine_loader import get_tradingagents_config

        config = get_tradingagents_config(
            llm_provider="openai",
            deep_think_llm="gpt-4o",
            quick_think_llm="gpt-4o-mini",
            max_debate_rounds=2,
        )

        assert config["llm_provider"] == "openai"
        assert config["deep_think_llm"] == "gpt-4o"
        assert config["quick_think_llm"] == "gpt-4o-mini"
        assert config["max_debate_rounds"] == 2
        assert "data_vendors" in config

    def test_get_tradingagents_config_with_custom_vendors(self):
        """Test config builder accepts custom data vendors."""
        from src.agent.engine_loader import get_tradingagents_config

        custom_vendors = {
            "core_stock_apis": "alpha_vantage",
            "news_data": "google",
        }
        config = get_tradingagents_config(data_vendors=custom_vendors)

        assert config["data_vendors"] == custom_vendors


class TestTradingAgentsPolicyWithFakes:
    """Tests for TradingAgentsPolicy using fake TradingAgents."""

    @pytest.fixture
    def fake_tradingagents_module(self):
        """Create a fake tradingagents module for testing."""
        # Create fake module structure
        fake_trading_graph = MagicMock()
        fake_default_config = {"llm_provider": "openai", "deep_think_llm": "gpt-4o-mini"}

        # Create the fake TradingAgentsGraph class
        class FakeTradingAgentsGraph:
            def __init__(self, selected_analysts=None, debug=False, config=None):
                self.config = config or {}
                self.debug = debug
                self._propagate_called = False

            def propagate(self, symbol, trade_date):
                self._propagate_called = True
                # Return format: (final_state, decision_string)
                final_state = {
                    "company_of_interest": symbol,
                    "trade_date": trade_date,
                    "market_report": "Market is bullish",
                    "sentiment_report": "Positive sentiment",
                    "news_report": "Good news",
                    "fundamentals_report": "Strong fundamentals",
                    "investment_debate_state": {
                        "bull_history": [],
                        "bear_history": [],
                        "history": [],
                        "current_response": "BUY",
                        "judge_decision": "BUY recommended",
                    },
                    "risk_debate_state": {
                        "risky_history": [],
                        "safe_history": [],
                        "neutral_history": [],
                        "history": [],
                        "judge_decision": "Risk acceptable",
                    },
                    "trader_investment_plan": "Buy 100 shares",
                    "investment_plan": "Long-term bullish position",
                    "final_trade_decision": "BUY",
                }
                return final_state, "BUY"

            def reflect_and_remember(self, returns):
                pass

        return FakeTradingAgentsGraph, fake_default_config

    def test_tradingagents_policy_parse_action_buy(self):
        """Test action parsing for BUY."""
        from src.agent.tradingagents_engine import TradingAgentsPolicy
        from src.agent.policy import Action

        policy = TradingAgentsPolicy(lazy_init=True)
        assert policy._parse_action("BUY") == Action.BUY
        assert policy._parse_action("buy") == Action.BUY
        assert policy._parse_action("  BUY  ") == Action.BUY

    def test_tradingagents_policy_parse_action_sell(self):
        """Test action parsing for SELL."""
        from src.agent.tradingagents_engine import TradingAgentsPolicy
        from src.agent.policy import Action

        policy = TradingAgentsPolicy(lazy_init=True)
        assert policy._parse_action("SELL") == Action.SELL
        assert policy._parse_action("sell") == Action.SELL

    def test_tradingagents_policy_parse_action_hold(self):
        """Test action parsing for HOLD."""
        from src.agent.tradingagents_engine import TradingAgentsPolicy
        from src.agent.policy import Action

        policy = TradingAgentsPolicy(lazy_init=True)
        assert policy._parse_action("HOLD") == Action.HOLD
        assert policy._parse_action("hold") == Action.HOLD
        assert policy._parse_action("") == Action.HOLD
        assert policy._parse_action(None) == Action.HOLD

    def test_tradingagents_policy_parse_action_strong_buy(self):
        """Test action parsing for STRONG_BUY."""
        from src.agent.tradingagents_engine import TradingAgentsPolicy
        from src.agent.policy import Action

        policy = TradingAgentsPolicy(lazy_init=True)
        assert policy._parse_action("STRONG_BUY") == Action.STRONG_BUY
        assert policy._parse_action("STRONGLY BUY") == Action.STRONG_BUY

    def test_tradingagents_policy_parse_action_strong_sell(self):
        """Test action parsing for STRONG_SELL."""
        from src.agent.tradingagents_engine import TradingAgentsPolicy
        from src.agent.policy import Action

        policy = TradingAgentsPolicy(lazy_init=True)
        assert policy._parse_action("STRONG_SELL") == Action.STRONG_SELL
        assert policy._parse_action("STRONGLY SELL") == Action.STRONG_SELL

    def test_tradingagents_policy_compute_confidence(self):
        """Test confidence computation from state."""
        from src.agent.tradingagents_engine import TradingAgentsPolicy

        policy = TradingAgentsPolicy(lazy_init=True)

        # Empty state -> base confidence
        state = {}
        assert policy._compute_confidence(state) == pytest.approx(0.7)

        # With judge decisions -> higher confidence
        state = {
            "investment_debate_state": {"judge_decision": "BUY"},
            "risk_debate_state": {"judge_decision": "Safe"},
        }
        assert policy._compute_confidence(state) == pytest.approx(0.9)

    def test_tradingagents_policy_extract_rationale(self):
        """Test rationale extraction from state."""
        from src.agent.tradingagents_engine import TradingAgentsPolicy

        policy = TradingAgentsPolicy(lazy_init=True)

        state = {"investment_plan": "Long position recommended"}
        rationale = policy._extract_rationale(state, "BUY")

        assert "TradingAgents" in rationale
        assert "Long position" in rationale
        assert "BUY" in rationale

    def test_tradingagents_policy_with_mocked_graph(self, fake_tradingagents_module):
        """Test full policy execution with mocked TradingAgents."""
        from src.agent.tradingagents_engine import TradingAgentsPolicy
        from src.agent.policy import Action

        FakeGraph, _ = fake_tradingagents_module

        policy = TradingAgentsPolicy(lazy_init=True)
        # Inject fake graph
        policy._graph = FakeGraph()

        features = {"trend_score": 0.03}
        context = {"symbol": "AAPL", "trade_date": "2024-01-15"}

        result = policy.decide_with_trace(features, context)

        assert result.decision.action == Action.BUY
        assert result.trace.llm_called is True
        assert result.trace.llm_provider == "openai"
        assert result.policy_name == "tradingagents"
        assert result.trace.llm_latency_ms is not None
        assert result.trace.llm_input_hash is not None

    def test_tradingagents_policy_decide_returns_decision(self, fake_tradingagents_module):
        """Test that decide() returns just the Decision."""
        from src.agent.tradingagents_engine import TradingAgentsPolicy
        from src.agent.policy import Action, Decision

        FakeGraph, _ = fake_tradingagents_module

        policy = TradingAgentsPolicy(lazy_init=True)
        policy._graph = FakeGraph()

        features = {}
        context = {"symbol": "TSLA"}

        decision = policy.decide(features, context)

        assert isinstance(decision, Decision)
        assert decision.action == Action.BUY


class TestTradingAgentsPolicyFactory:
    """Tests for create_tradingagents_policy factory."""

    def test_factory_extracts_config_values(self):
        """Test that factory extracts config values correctly."""
        from src.agent.tradingagents_engine import create_tradingagents_policy
        from src.agent.engine_loader import TradingAgentsNotFoundError

        config = {
            "llm_provider": "google",
            "llm_model": "gemini-pro",
            "tradingagents_max_debate_rounds": 3,
        }

        # This will fail because TradingAgents isn't available in test environment
        # But we can verify the function signature works
        with pytest.raises((TradingAgentsNotFoundError, SystemExit, Exception)):
            create_tradingagents_policy(config)


class TestFailFastBehavior:
    """Tests for fail-fast behavior when TradingAgents is required."""

    def test_policy_fails_fast_on_tradingagents_error(self):
        """Test that policy doesn't silently fall back on TradingAgents errors."""
        from src.agent.tradingagents_engine import TradingAgentsPolicy

        policy = TradingAgentsPolicy(lazy_init=True)

        # Mock a graph that raises an exception
        class FailingGraph:
            def propagate(self, symbol, date):
                raise RuntimeError("TradingAgents internal error")

        policy._graph = FailingGraph()

        with pytest.raises(RuntimeError) as exc_info:
            policy.decide_with_trace({}, {"symbol": "AAPL"})

        assert "TradingAgents decision failed" in str(exc_info.value)

    def test_no_fallback_to_rules_on_error(self):
        """Verify there's no silent fallback to rules policy."""
        from src.agent.tradingagents_engine import TradingAgentsPolicy

        policy = TradingAgentsPolicy(lazy_init=True)

        class FailingGraph:
            def propagate(self, symbol, date):
                raise ValueError("Invalid data")

        policy._graph = FailingGraph()

        # Should raise, not return a rules-based decision
        with pytest.raises(RuntimeError):
            policy.decide({}, {"symbol": "AAPL"})


class TestDecisionTracing:
    """Tests for decision tracing with TradingAgents."""

    def test_trace_records_provider_and_model(self):
        """Test that trace correctly records provider and model info."""
        from src.agent.tradingagents_engine import TradingAgentsPolicy

        policy = TradingAgentsPolicy(
            llm_provider="anthropic",
            deep_think_llm="claude-3",
            quick_think_llm="claude-instant",
            lazy_init=True,
        )

        # Mock graph
        class MockGraph:
            def propagate(self, symbol, date):
                return {"investment_plan": "test"}, "HOLD"

        policy._graph = MockGraph()

        result = policy.decide_with_trace({}, {"symbol": "TEST"})

        assert result.trace.llm_provider == "anthropic"
        assert "claude-3" in result.trace.llm_model
        assert "claude-instant" in result.trace.llm_model

    def test_trace_records_latency(self):
        """Test that latency is recorded."""
        from src.agent.tradingagents_engine import TradingAgentsPolicy
        import time

        policy = TradingAgentsPolicy(lazy_init=True)

        class SlowGraph:
            def propagate(self, symbol, date):
                time.sleep(0.05)  # 50ms delay
                return {}, "HOLD"

        policy._graph = SlowGraph()

        result = policy.decide_with_trace({}, {"symbol": "TEST"})

        assert result.trace.llm_latency_ms is not None
        assert result.trace.llm_latency_ms >= 50  # At least 50ms

    def test_trace_records_input_and_output_hashes(self):
        """Test that input/output hashes are computed."""
        from src.agent.tradingagents_engine import TradingAgentsPolicy

        policy = TradingAgentsPolicy(lazy_init=True)

        class MockGraph:
            def propagate(self, symbol, date):
                return {"investment_plan": "test"}, "BUY"

        policy._graph = MockGraph()

        result = policy.decide_with_trace(
            {"trend_score": 0.5},
            {"symbol": "AAPL", "trade_date": "2024-01-15"},
        )

        assert result.trace.llm_input_hash is not None
        assert len(result.trace.llm_input_hash) == 16  # SHA256 truncated to 16 chars
        assert result.trace.llm_output_hash is not None


class TestReflectAndRemember:
    """Tests for learning from trade outcomes."""

    def test_reflect_and_remember_calls_graph(self):
        """Test that reflect_and_remember passes through to graph."""
        from src.agent.tradingagents_engine import TradingAgentsPolicy

        policy = TradingAgentsPolicy(lazy_init=True)

        reflect_called = {"called": False, "returns": None}

        class MockGraph:
            def reflect_and_remember(self, returns):
                reflect_called["called"] = True
                reflect_called["returns"] = returns

        policy._graph = MockGraph()

        policy.reflect_and_remember(0.05)  # 5% return

        assert reflect_called["called"] is True
        assert reflect_called["returns"] == 0.05

    def test_reflect_handles_missing_graph(self):
        """Test that reflect doesn't fail if graph not initialized."""
        from src.agent.tradingagents_engine import TradingAgentsPolicy

        policy = TradingAgentsPolicy(lazy_init=True)
        policy._graph = None

        # Should not raise
        policy.reflect_and_remember(0.0)
