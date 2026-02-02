from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List

import pytest
from fastapi.testclient import TestClient

from execution.broker_interface import Position
from observability.web import APP_VERSION, _create_app


class _MockTable:
    def __init__(self, rows: Iterable[Dict[str, Any]]) -> None:
        self._rows = list(rows)

    def rows_where(self, *_args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        limit = kwargs.get("limit")
        if limit is None:
            return list(self._rows)
        return list(self._rows)[: int(limit)]


class _MockDatabase:
    def __init__(self, tables: Dict[str, _MockTable]) -> None:
        self._db = tables


class _MockBroker:
    def __init__(self, positions: Iterable[Position], market_open: bool = True) -> None:
        self._positions = list(positions)
        self._market_open = market_open

    def get_positions(self) -> List[Position]:
        return list(self._positions)

    def is_market_open(self) -> bool:
        return self._market_open


def test_health_endpoint() -> None:
    db = _MockDatabase({})
    broker = _MockBroker([])
    app = _create_app(db, broker, config={}, log_file=None)
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["version"] == APP_VERSION
    assert payload["timestamp"]


def test_status_endpoint() -> None:
    db = _MockDatabase({})
    positions = [
        Position(symbol="AAPL", qty=10, avg_price=100.0, market_value=1050.0, unrealized_pl=50.0),
        Position(symbol="TSLA", qty=-5, avg_price=200.0, market_value=950.0, unrealized_pl=-50.0),
    ]
    broker = _MockBroker(positions, market_open=True)
    last_run = datetime(2026, 1, 30, 10, 0, tzinfo=timezone.utc)
    config = {"trading_mode": "paper", "last_run": last_run}
    app = _create_app(db, broker, config=config, log_file=None)
    client = TestClient(app)

    response = client.get("/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "paper"
    # Handle both 'Z' and '+00:00' datetime formats
    last_run_str = payload["last_run"]
    if last_run_str.endswith("Z"):
        last_run_str = last_run_str[:-1] + "+00:00"
    assert last_run_str == last_run.isoformat()
    assert payload["market_open"] is True
    assert "positions_summary" in payload
    assert "pnl_summary" in payload

    positions_summary = payload["positions_summary"]
    assert positions_summary["count"] == 2
    assert positions_summary["long_count"] == 1
    assert positions_summary["short_count"] == 1


def test_decisions_endpoint() -> None:
    decisions = [
        {
            "timestamp": datetime(2026, 1, 30, 14, 5, tzinfo=timezone.utc),
            "symbol": "AAPL",
            "action": "BUY",
            "confidence": 0.82,
            "rationale": "Momentum improving",
            "feature_snapshot": {"rsi_14": 61.4},
            "policy_type": "rules_v1",
        }
    ]
    db = _MockDatabase({"decisions": _MockTable(decisions)})
    broker = _MockBroker([])
    app = _create_app(db, broker, config={}, log_file=None)
    client = TestClient(app)

    response = client.get("/decisions?limit=10")

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert len(payload) == 1
    assert payload[0]["symbol"] == "AAPL"
    assert payload[0]["action"] == "BUY"


def test_orders_endpoint() -> None:
    orders = [
        {
            "order_id": "ord_123",
            "symbol": "MSFT",
            "side": "BUY",
            "qty": 10.0,
            "order_type": "MARKET",
            "limit_price": None,
            "status": "FILLED",
            "submitted_at": datetime(2026, 1, 30, 14, 6, tzinfo=timezone.utc),
            "filled_at": datetime(2026, 1, 30, 14, 6, 5, tzinfo=timezone.utc),
            "filled_qty": 10.0,
            "filled_price": 318.5,
            "broker": "paper",
        }
    ]
    db = _MockDatabase({"orders": _MockTable(orders)})
    broker = _MockBroker([])
    app = _create_app(db, broker, config={}, log_file=None)
    client = TestClient(app)

    response = client.get("/orders?limit=10")

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert len(payload) == 1
    assert payload[0]["order_id"] == "ord_123"
    assert payload[0]["symbol"] == "MSFT"


def test_positions_endpoint() -> None:
    db = _MockDatabase({})
    positions = [
        Position(symbol="NVDA", qty=15, avg_price=400.0, market_value=6150.0, unrealized_pl=150.0)
    ]
    broker = _MockBroker(positions)
    app = _create_app(db, broker, config={}, log_file=None)
    client = TestClient(app)

    response = client.get("/positions")

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert len(payload) == 1
    assert payload[0]["symbol"] == "NVDA"
    assert payload[0]["qty"] == 15
