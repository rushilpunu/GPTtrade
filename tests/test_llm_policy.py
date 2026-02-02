"""Tests verifying LLM policy is called and used correctly."""

import pytest
from unittest.mock import MagicMock, patch

from src.agent.policy import (
    LLMPolicy,
    RulesPolicy,
    Decision,
    DecisionWithTrace,
    LLMTrace,
    Action,
)
from src.agent.llm_clients import FakeLLMClient, LLMClientError


class TestLLMCalledAndUsedOpenAI:
    """Test that LLM is actually called and used when configured for OpenAI."""

    def test_llm_called_and_used_openai(self):
        """Inject FakeLLMClient returning BUY, verify decision uses it."""
        fake_response = {
            "action": "BUY",
            "confidence": 0.85,
            "rationale": "Strong uptrend detected by LLM analysis",
        }
        fake_client = FakeLLMClient(fixed_response=fake_response)

        policy = LLMPolicy(
            llm_client=fake_client,
            fallback_policy=RulesPolicy(),
            provider="openai",
            model="gpt-4o-mini",
        )

        features = {
            "trend_score": 0.01,  # Not strong enough for rules to say BUY
            "return_anomaly_zscore": 0.5,
            "sentiment_score": 0.0,
        }
        context = {"symbol": "AAPL"}

        result = policy.decide_with_trace(features, context)

        # Verify LLM was called
        assert result.trace.llm_called is True
        assert result.trace.llm_provider == "openai"
        assert result.trace.llm_model == "gpt-4o-mini"
        assert result.trace.used_fallback is False
        assert result.trace.llm_fallback_reason is None

        # Verify LLM action was used
        assert result.trace.llm_action == "BUY"
        assert result.decision.action == Action.BUY
        assert result.decision.confidence == 0.85
        assert "LLM" in result.decision.rationale or "uptrend" in result.decision.rationale

        # Verify policy name
        assert result.policy_name == "llm"

        # Verify latency was recorded
        assert result.trace.llm_latency_ms is not None
        assert result.trace.llm_latency_ms >= 0

        # Verify hashes were computed
        assert result.trace.llm_input_hash is not None
        assert result.trace.llm_output_hash is not None


class TestLLMCalledAndUsedGemini:
    """Test that LLM is actually called and used when configured for Gemini."""

    def test_llm_called_and_used_gemini(self):
        """Same test but with provider='gemini'."""
        fake_response = {
            "action": "STRONG_SELL",
            "confidence": 0.92,
            "rationale": "Gemini detected major bearish signals",
        }
        fake_client = FakeLLMClient(fixed_response=fake_response)

        policy = LLMPolicy(
            llm_client=fake_client,
            fallback_policy=RulesPolicy(),
            provider="gemini",
            model="gemini-2.0-flash",
        )

        features = {
            "trend_score": 0.0,  # Neutral
            "return_anomaly_zscore": 0.0,
            "sentiment_score": 0.0,
        }
        context = {"symbol": "TSLA"}

        result = policy.decide_with_trace(features, context)

        assert result.trace.llm_called is True
        assert result.trace.llm_provider == "gemini"
        assert result.trace.llm_model == "gemini-2.0-flash"
        assert result.trace.used_fallback is False

        assert result.trace.llm_action == "STRONG_SELL"
        assert result.decision.action == Action.STRONG_SELL
        assert result.decision.confidence == 0.92
        assert result.policy_name == "llm"


class TestLLMFailureFallsBack:
    """Test that LLM failures correctly fall back to rules."""

    def test_llm_timeout_falls_back(self):
        """FakeLLMClient raises timeout, verify fallback."""
        class TimeoutClient(FakeLLMClient):
            def generate_json(self, system_prompt, user_payload, schema):
                raise LLMClientError("Request timeout after 10s")

        policy = LLMPolicy(
            llm_client=TimeoutClient(fixed_response={}),
            fallback_policy=RulesPolicy(),
            provider="openai",
            model="gpt-4o-mini",
        )

        features = {
            "trend_score": 0.03,  # Strong enough for rules to say BUY
            "return_anomaly_zscore": 0.5,
            "sentiment_score": 0.0,
        }
        context = {"symbol": "AAPL"}

        result = policy.decide_with_trace(features, context)

        # Verify LLM was called but failed
        assert result.trace.llm_called is True
        assert result.trace.used_fallback is True
        assert result.trace.llm_fallback_reason == "timeout"

        # Verify rules policy was used
        assert result.decision.action == Action.BUY  # From rules
        assert "Rules-based" in result.decision.rationale
        assert result.policy_name == "llm"  # Still "llm" policy, but with fallback

    def test_llm_http_error_falls_back(self):
        """FakeLLMClient raises HTTP error, verify fallback."""
        class HTTPErrorClient(FakeLLMClient):
            def generate_json(self, system_prompt, user_payload, schema):
                raise LLMClientError("OpenAI API request failed: 500 Internal Server Error")

        policy = LLMPolicy(
            llm_client=HTTPErrorClient(fixed_response={}),
            fallback_policy=RulesPolicy(),
            provider="openai",
            model="gpt-4o-mini",
        )

        features = {"trend_score": 0.0}
        context = {}

        result = policy.decide_with_trace(features, context)

        assert result.trace.llm_called is True
        assert result.trace.used_fallback is True
        assert result.trace.llm_fallback_reason == "http_error"

    def test_llm_invalid_json_falls_back(self):
        """FakeLLMClient raises JSON parse error, verify fallback."""
        class JSONErrorClient(FakeLLMClient):
            def generate_json(self, system_prompt, user_payload, schema):
                raise LLMClientError("Failed to parse JSON response")

        policy = LLMPolicy(
            llm_client=JSONErrorClient(fixed_response={}),
            fallback_policy=RulesPolicy(),
            provider="gemini",
            model="gemini-2.0-flash",
        )

        features = {"trend_score": -0.03}  # Rules would say SELL
        context = {}

        result = policy.decide_with_trace(features, context)

        assert result.trace.llm_called is True
        assert result.trace.used_fallback is True
        assert result.trace.llm_fallback_reason == "invalid_json"
        assert result.decision.action == Action.SELL

    def test_llm_schema_validation_fails(self):
        """FakeLLMClient raises schema validation error, verify fallback."""
        class SchemaErrorClient(FakeLLMClient):
            def generate_json(self, system_prompt, user_payload, schema):
                raise LLMClientError("Schema validation failed: missing required field")

        policy = LLMPolicy(
            llm_client=SchemaErrorClient(fixed_response={}),
            fallback_policy=RulesPolicy(),
            provider="openai",
            model="gpt-4o-mini",
        )

        features = {"trend_score": 0.0}
        context = {}

        result = policy.decide_with_trace(features, context)

        assert result.trace.llm_called is True
        assert result.trace.used_fallback is True
        assert result.trace.llm_fallback_reason == "schema_fail"


class TestLLMOutputInfluencesSizing:
    """Test that LLM confidence affects position sizing."""

    def test_different_confidence_different_sizing(self):
        """Same market data, different LLM confidence -> different target size."""
        from src.risk.position_sizing import PositionSizer

        # Use 10 symbols so each gets 10% allocation, staying under max_position_pct
        config = {
            "max_position_pct": 0.20,  # 20% max per position
            "confidence_scale_min": 0.5,
            "confidence_scale_max": 1.5,
            "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
                        "META", "TSLA", "SPY", "QQQ", "AMD"],
        }
        sizer = PositionSizer(config)

        equity = 100000
        price = 100.0
        positions = []

        # Low confidence LLM decision (0.5 -> scale = 0.5 + 0.5*1.0 = 1.0)
        low_conf_shares = sizer.calculate_shares(
            symbol="AAPL",
            signal_strength="BUY",
            current_price=price,
            account_equity=equity,
            current_positions=positions,
            confidence=0.5,
        )

        # High confidence LLM decision (0.95 -> scale = 0.5 + 0.95*1.0 = 1.45)
        high_conf_shares = sizer.calculate_shares(
            symbol="AAPL",
            signal_strength="BUY",
            current_price=price,
            account_equity=equity,
            current_positions=positions,
            confidence=0.95,
        )

        # Verify both return non-zero
        assert low_conf_shares > 0, f"low_conf_shares={low_conf_shares}"
        assert high_conf_shares > 0, f"high_conf_shares={high_conf_shares}"

        # Higher confidence should result in more shares
        assert high_conf_shares > low_conf_shares, \
            f"high={high_conf_shares} should be > low={low_conf_shares}"


class TestOverridesAreVisible:
    """Test that risk gate overrides are properly tracked."""

    def test_llm_buy_blocked_by_risk_gate(self):
        """LLM says BUY but risk gate blocks (e.g., outside market hours)."""
        from src.risk.risk_gate import RiskGate

        class MockBroker:
            pass

        # Configure risk gate to block during "outside market hours"
        config = {}
        gate = RiskGate(config, MockBroker())

        # Simulate account info and positions
        account_info = {"equity": 100000, "prices": {"AAPL": 150.0}}
        positions = []

        # Features with blackout active
        features = {"blackout_active": True}

        # Check if order would be blocked
        approved, reasons = gate.check_order(
            symbol="AAPL",
            side="BUY",
            qty=10,
            current_positions=positions,
            account_info=account_info,
            features=features,
        )

        # Verify override
        assert approved is False
        assert "calendar_blackout" in reasons


class TestRulesPolicyTracesCorrectly:
    """Test that RulesPolicy also produces correct traces."""

    def test_rules_policy_trace(self):
        """Verify RulesPolicy.decide_with_trace returns proper trace."""
        policy = RulesPolicy()

        features = {
            "trend_score": 0.03,
            "return_anomaly_zscore": 0.5,
            "sentiment_score": 0.2,
        }
        context = {"symbol": "AAPL"}

        result = policy.decide_with_trace(features, context)

        assert result.policy_name == "rules"
        assert result.trace.llm_called is False
        assert result.trace.llm_provider is None
        assert result.trace.llm_model is None
        assert result.trace.used_fallback is False
        assert result.decision.action == Action.BUY


class TestDecisionRecordHasTraceFields:
    """Test that DecisionRecord model has all required trace fields."""

    def test_decision_record_trace_fields(self):
        """Verify DecisionRecord has all LLM trace fields."""
        from src.storage.models import DecisionRecord
        from datetime import datetime

        record = DecisionRecord(
            timestamp=datetime.utcnow(),
            symbol="AAPL",
            action="BUY",
            confidence=0.85,
            rationale="Test decision",
            feature_snapshot={"trend_score": 0.03},
            policy_type="llm",
            policy_name="llm",
            llm_provider="openai",
            llm_model="gpt-4o-mini",
            llm_called=True,
            llm_fallback_reason=None,
            llm_latency_ms=150.5,
            llm_input_hash="abc123",
            llm_output_hash="def456",
            llm_action="BUY",
            final_action="BUY",
            action_overridden_by_risk_gate=False,
            override_reason=None,
        )

        assert record.policy_name == "llm"
        assert record.llm_provider == "openai"
        assert record.llm_called is True
        assert record.llm_latency_ms == 150.5


class TestIntegrationWithTradingSystem:
    """Integration tests for LLM policy with TradingSystem."""

    def test_llm_decision_stored_in_database(self, tmp_path):
        """Verify LLM decisions are stored with full trace info."""
        from src.storage.db import TradingDatabase
        from src.storage.models import DecisionRecord, DecisionAction
        from datetime import datetime

        db_path = tmp_path / "test.db"
        db = TradingDatabase(str(db_path))

        # Create a decision record with LLM trace
        record = DecisionRecord(
            timestamp=datetime.utcnow(),
            symbol="AAPL",
            action=DecisionAction.BUY,
            confidence=0.85,
            rationale="LLM detected strong buy signal",
            feature_snapshot={"trend_score": 0.03},
            policy_type="llm",
            policy_name="llm",
            llm_provider="openai",
            llm_model="gpt-4o-mini",
            llm_called=True,
            llm_fallback_reason=None,
            llm_latency_ms=150.5,
            llm_input_hash="abc123",
            llm_output_hash="def456",
            llm_action="BUY",
        )

        db.save_decision(record)

        # Retrieve and verify
        decisions = db.get_decisions_by_symbol("AAPL", limit=1)
        assert len(decisions) == 1

        saved = decisions[0]
        assert saved.symbol == "AAPL"
        assert saved.action == DecisionAction.BUY
        assert saved.policy_name == "llm"
        assert saved.llm_provider == "openai"
        assert saved.llm_called is True
