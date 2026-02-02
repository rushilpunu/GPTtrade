from __future__ import annotations

from datetime import datetime, timedelta

import pytz
import pytest

from risk.risk_gate import RiskGate


class MockBroker:
    def __init__(
        self,
        last_price: float | None = 100.0,
        last_trade_time: datetime | None = None,
        trades_today: int | list | None = 0,
        pdt_restricted: bool = False,
        expected_edge: float | None = 0.01,
    ) -> None:
        self._last_price = last_price
        self._last_trade_time = last_trade_time
        self._trades_today = trades_today
        self._pdt_restricted = pdt_restricted
        self._expected_edge = expected_edge

    def get_last_price(self, symbol: str) -> float | None:
        return self._last_price

    def get_last_trade_time(self, symbol: str) -> datetime | None:
        return self._last_trade_time

    def get_trades_today(self) -> int | list | None:
        return self._trades_today

    def is_pdt_restricted(self) -> bool:
        return self._pdt_restricted

    def get_expected_edge(self) -> float | None:
        return self._expected_edge


@pytest.fixture
def base_config() -> dict:
    return {}


@pytest.fixture
def mock_broker() -> MockBroker:
    return MockBroker()


@pytest.fixture
def base_account_info() -> dict:
    return {
        "equity": 10_000.0,
        "prices": {"AAPL": 100.0, "MSFT": 50.0},
        "daily_loss_pct": 0.0,
        "trades_today": 0,
        "pdt_restricted": False,
    }


def _make_gate(config: dict, broker: MockBroker, monkeypatch: pytest.MonkeyPatch) -> RiskGate:
    gate = RiskGate(config, broker)
    monkeypatch.setattr(gate, "_within_market_hours", lambda: True)
    return gate


def test_market_hours_check_inside_outside(monkeypatch: pytest.MonkeyPatch, mock_broker: MockBroker, base_account_info: dict) -> None:
    import risk.risk_gate as risk_gate

    class FixedInside(datetime):
        @classmethod
        def now(cls, tz=None) -> datetime:
            base = datetime(2025, 1, 2, 10, 0, 0)
            return tz.localize(base) if tz else base

        @classmethod
        def fromisoformat(cls, value: str) -> datetime:
            return datetime.fromisoformat(value)

    class FixedOutside(datetime):
        @classmethod
        def now(cls, tz=None) -> datetime:
            base = datetime(2025, 1, 2, 8, 0, 0)
            return tz.localize(base) if tz else base

        @classmethod
        def fromisoformat(cls, value: str) -> datetime:
            return datetime.fromisoformat(value)

    gate = RiskGate({}, mock_broker)

    monkeypatch.setattr(risk_gate, "datetime", FixedInside)
    assert gate._within_market_hours() is True
    ok, reasons = gate.check_order("AAPL", "BUY", 1, [], base_account_info)
    assert ok is True
    assert reasons == []

    monkeypatch.setattr(risk_gate, "datetime", FixedOutside)
    assert gate._within_market_hours() is False
    ok, reasons = gate.check_order("AAPL", "BUY", 1, [], base_account_info)
    assert ok is False
    assert "outside_market_hours" in reasons


def test_max_position_pct_constraint_violation(
    base_config: dict,
    mock_broker: MockBroker,
    base_account_info: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = dict(base_config)
    config["max_position_pct"] = 0.1
    gate = _make_gate(config, mock_broker, monkeypatch)

    ok, reasons = gate.check_order("AAPL", "BUY", 20, [], base_account_info)
    assert ok is False
    assert "max_position_pct" in reasons


def test_max_gross_exposure_pct_constraint_violation(
    base_config: dict,
    mock_broker: MockBroker,
    base_account_info: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = dict(base_config)
    config["max_gross_exposure_pct"] = 0.2
    gate = _make_gate(config, mock_broker, monkeypatch)

    positions = [
        {"symbol": "AAPL", "qty": 15, "market_value": 1500.0},
        {"symbol": "MSFT", "qty": 14, "market_value": 700.0},
    ]
    ok, reasons = gate.check_order("AAPL", "BUY", 10, positions, base_account_info)
    assert ok is False
    assert "max_gross_exposure_pct" in reasons


def test_daily_kill_switch_trigger(
    base_config: dict,
    mock_broker: MockBroker,
    base_account_info: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = dict(base_config)
    config["max_daily_loss_pct"] = 0.02
    gate = _make_gate(config, mock_broker, monkeypatch)

    account_info = dict(base_account_info)
    account_info["daily_loss_pct"] = 0.05
    ok, reasons = gate.check_order("AAPL", "BUY", 1, [], account_info)
    assert ok is False
    assert "daily_kill_switch" in reasons


def test_pdt_restriction_blocking(
    base_config: dict,
    mock_broker: MockBroker,
    base_account_info: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = dict(base_config)
    gate = _make_gate(config, mock_broker, monkeypatch)

    account_info = dict(base_account_info)
    account_info["pdt_restricted"] = True
    ok, reasons = gate.check_order("AAPL", "BUY", 1, [], account_info)
    assert ok is False
    assert "pdt_restricted" in reasons


def test_symbol_cooldown_blocks_trade(
    base_config: dict,
    mock_broker: MockBroker,
    base_account_info: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = dict(base_config)
    config["cooldown_minutes"] = 30
    gate = _make_gate(config, mock_broker, monkeypatch)

    account_info = dict(base_account_info)
    account_info["last_trade_times"] = {
        "AAPL": datetime.now(tz=pytz.UTC) - timedelta(minutes=5)
    }
    ok, reasons = gate.check_order("AAPL", "BUY", 1, [], account_info)
    assert ok is False
    assert "symbol_cooldown" in reasons


def test_max_trades_per_day_limit(
    base_config: dict,
    mock_broker: MockBroker,
    base_account_info: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = dict(base_config)
    config["max_trades_per_day"] = 3
    gate = _make_gate(config, mock_broker, monkeypatch)

    account_info = dict(base_account_info)
    account_info["trades_today"] = ["t1", "t2", "t3"]
    ok, reasons = gate.check_order("AAPL", "BUY", 1, [], account_info)
    assert ok is False
    assert "max_trades_per_day" in reasons


def test_missing_price_is_blocked(
    base_config: dict, base_account_info: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    broker = MockBroker(last_price=None)
    gate = _make_gate(base_config, broker, monkeypatch)

    account_info = dict(base_account_info)
    account_info.pop("prices", None)
    ok, reasons = gate.check_order("AAPL", "BUY", 1, [], account_info)
    assert ok is False
    assert "missing_price" in reasons


def test_invalid_equity_is_blocked(
    base_config: dict,
    mock_broker: MockBroker,
    base_account_info: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = _make_gate(base_config, mock_broker, monkeypatch)

    account_info = dict(base_account_info)
    account_info["equity"] = 0
    ok, reasons = gate.check_order("AAPL", "BUY", 1, [], account_info)
    assert ok is False
    assert "missing_or_invalid_equity" in reasons
