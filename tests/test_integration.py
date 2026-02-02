from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
import pytest

from agent.policy import Action, Decision, LLMPolicy, RulesPolicy
from data.market_data import MarketDataProvider, Quote
from features.behavioral_features import BehavioralFeatureCalculator
from observability.logging import setup_logging, with_correlation_id


class _FixedMarketData(MarketDataProvider):
    def __init__(self, ohlcv: pd.DataFrame) -> None:
        self._ohlcv = ohlcv.copy()

    def get_ohlcv(self, symbol: str, start: Optional[datetime], end: Optional[datetime]) -> pd.DataFrame:
        return self._ohlcv.copy()

    def get_quote(self, symbol: str) -> Quote:
        last_row = self._ohlcv.iloc[-1]
        last_price = float(last_row["close"])
        timestamp = self._ohlcv.index[-1].to_pydatetime()
        return Quote(
            bid=last_price * 0.999,
            ask=last_price * 1.001,
            last=last_price,
            volume=float(last_row.get("volume", 0.0)),
            timestamp=timestamp,
        )


def _make_trend_ohlcv(periods: int = 60) -> pd.DataFrame:
    index = pd.date_range(start=datetime(2024, 1, 1), periods=periods, freq="1min")
    base = np.linspace(100.0, 120.0, periods)
    oscillation = np.sin(np.linspace(0.0, 3.0 * np.pi, periods)) * 0.5
    close = base + oscillation
    open_prices = close + np.sin(np.linspace(0.0, 2.0 * np.pi, periods)) * 0.2
    high_prices = np.maximum(open_prices, close) + 0.3
    low_prices = np.minimum(open_prices, close) - 0.3
    volume = np.linspace(1000.0, 2000.0, periods) + np.cos(np.linspace(0.0, 2.0 * np.pi, periods)) * 10.0
    return pd.DataFrame(
        {
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


def _flush_logs() -> None:
    for handler in logging.getLogger().handlers:
        try:
            handler.flush()
        except Exception:
            continue


def _load_constraint_names(db_path: str) -> set[str]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT constraint_name FROM constraint_violations").fetchall()
    return {row[0] for row in rows}


def _read_structured_logs(log_path: str) -> list[dict]:
    if not log_path:
        return []
    records: list[dict] = []
    try:
        with open(log_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return []
    return records


def test_full_cycle_rules_policy(
    test_trading_system_factory,
    tmp_db_path,
    tmp_log_dir,
    simulator_broker_factory,
) -> None:
    log_path = str(tmp_log_dir / "integration_rules.jsonl")
    setup_logging(
        {
            "logging": {
                "to_console": False,
                "to_file": True,
                "file": log_path,
                "level": "INFO",
            }
        }
    )

    symbols = ["AAPL", "MSFT", "NVDA"]
    market_data = _FixedMarketData(_make_trend_ohlcv())
    broker = simulator_broker_factory(symbols=symbols)
    system = test_trading_system_factory(
        config={"symbols": symbols},
        broker=broker,
        market_data=market_data,
        policy=RulesPolicy(),
        dry_run=False,
    )

    correlation_id = "test-correlation-id-rules"
    with with_correlation_id(correlation_id):
        system.run_cycle()

    _flush_logs()

    with sqlite3.connect(tmp_db_path) as conn:
        decision_rows = conn.execute("SELECT symbol, action FROM decisions").fetchall()
        order_rows = conn.execute("SELECT symbol FROM orders").fetchall()

    decisions_by_symbol: Dict[str, str] = {}
    for symbol, action in decision_rows:
        decisions_by_symbol[str(symbol)] = str(action)

    assert set(decisions_by_symbol.keys()) == set(symbols)

    orders_by_symbol = {str(row[0]) for row in order_rows}
    for symbol, action in decisions_by_symbol.items():
        if action != "HOLD":
            assert symbol in orders_by_symbol

    positions = broker.get_positions()
    assert any(abs(pos.qty) > 0 for pos in positions)

    records = _read_structured_logs(log_path)
    assert any(rec.get("correlation_id") == correlation_id for rec in records)


def test_full_cycle_llm_fallback(
    test_trading_system_factory,
    tmp_db_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent.llm_clients import FakeLLMClient, LLMClientError

    # Create a FakeLLMClient that raises to force fallback
    class _FailingLLMClient(FakeLLMClient):
        def generate_json(self, system_prompt, user_payload, schema):
            raise LLMClientError("Simulated failure")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    policy = LLMPolicy(llm_client=_FailingLLMClient(fixed_response={}), fallback_policy=RulesPolicy())

    symbols = ["AAPL", "MSFT"]
    market_data = _FixedMarketData(_make_trend_ohlcv())
    system = test_trading_system_factory(
        config={"symbols": symbols, "policy_type": "llm"},
        market_data=market_data,
        policy=policy,
        dry_run=True,
    )

    system.run_cycle()

    with sqlite3.connect(tmp_db_path) as conn:
        rows = conn.execute("SELECT rationale FROM decisions").fetchall()

    assert rows
    assert all(str(row[0]) == "Rules-based decision" for row in rows)


def test_feature_computation() -> None:
    calculator = BehavioralFeatureCalculator()
    ohlcv = _make_trend_ohlcv(periods=80)
    features = calculator.compute_all_features("TEST", ohlcv)

    expected_keys = {
        "symbol",
        "return_anomaly_zscore",
        "volume_anomaly_zscore",
        "volatility_regime",
        "mean_reversion_score",
        "trend_score",
    }
    assert expected_keys.issubset(features.keys())

    for key, value in features.items():
        if key == "symbol":
            continue
        assert np.isfinite(float(value))


def test_risk_gate_integration(
    test_trading_system_factory,
    tmp_db_path,
    simulator_broker_factory,
) -> None:
    symbols = ["AAPL"]
    broker = simulator_broker_factory(symbols=symbols)
    system = test_trading_system_factory(
        config={
            "symbols": symbols,
            "max_trades_per_day": 0,
            "max_gross_exposure_pct": 0.01,
            "cost_buffer": 0.9,
        },
        broker=broker,
        policy=RulesPolicy(),
        dry_run=False,
    )

    system._decisions_cache = {
        "AAPL": Decision(action=Action.BUY, confidence=0.6, rationale="force_buy")
    }

    system.execute_orders()

    violations = _load_constraint_names(str(tmp_db_path))
    assert "max_trades_per_day" in violations
    assert "max_gross_exposure_pct" in violations
    assert "cost_buffer" in violations


def test_sqlite_persistence(
    test_trading_system_factory,
    tmp_db_path,
    simulator_broker_factory,
) -> None:
    symbols = ["AAPL", "MSFT"]
    market_data = _FixedMarketData(_make_trend_ohlcv())
    broker = simulator_broker_factory(symbols=symbols)
    system = test_trading_system_factory(
        config={"symbols": symbols},
        broker=broker,
        market_data=market_data,
        policy=RulesPolicy(),
        dry_run=False,
    )

    system.run_cycle()

    with sqlite3.connect(tmp_db_path) as conn:
        decisions = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
        orders = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        positions = conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]

    assert decisions > 0
    assert orders > 0
    assert positions > 0
