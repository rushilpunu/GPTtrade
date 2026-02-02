from __future__ import annotations

import sqlite3
from typing import Any, Dict

import httpx
import pytest
import requests

from agent.policy import Action, Decision, LLMPolicy, RulesPolicy


class _AlwaysBuyPolicy(RulesPolicy):
    def decide(self, features: Dict[str, Any], context: Dict[str, Any]) -> Decision:
        return Decision(action=Action.BUY, confidence=0.9, rationale="Forced buy for offline test")


def test_simulator_runs_offline(
    mock_no_network: None,
    test_trading_system_factory,
    tmp_db_path,
) -> None:
    system = test_trading_system_factory(policy=_AlwaysBuyPolicy(), dry_run=False)

    system.run_cycle()

    with sqlite3.connect(tmp_db_path) as conn:
        decisions = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
        orders = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]

    assert decisions > 0
    assert orders > 0


def test_no_external_api_calls(
    monkeypatch: pytest.MonkeyPatch,
    test_trading_system_factory,
) -> None:
    def _blocked(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("External HTTP call attempted")

    monkeypatch.setattr(httpx.Client, "request", _blocked)
    monkeypatch.setattr(requests.Session, "request", _blocked)

    system = test_trading_system_factory(policy=_AlwaysBuyPolicy(), dry_run=False)
    system.run_cycle()


def test_llm_policy_graceful_offline(
    mock_no_network: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent.llm_clients import FakeLLMClient, LLMClientError

    # Create a FakeLLMClient that raises to simulate network failure
    class _FailingLLMClient(FakeLLMClient):
        def generate_json(self, system_prompt, user_payload, schema):
            raise LLMClientError("Simulated offline failure")

    policy = LLMPolicy(llm_client=_FailingLLMClient(fixed_response={}), fallback_policy=RulesPolicy())
    features = {"trend_score": 0.03, "return_anomaly_zscore": 0.0}
    decision = policy.decide(features, {"symbol": "AAPL"})

    assert decision.rationale == "Rules-based decision"
    assert decision.action == Action.BUY
