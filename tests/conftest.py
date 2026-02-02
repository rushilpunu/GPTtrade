from __future__ import annotations

import builtins
import hashlib
import socket
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import pytest

from agent.policy import RulesPolicy
from data.market_data import MarketDataProvider, Quote
from execution.simulator_broker import SimulatorBroker
from features.behavioral_features import BehavioralFeatureCalculator
from main import TradingSystem
from observability.notify import NullNotifier
from risk.position_sizing import PositionSizer
from risk.risk_gate import RiskGate
from storage.db import TradingDatabase


class _DeterministicMarketData(MarketDataProvider):
    def __init__(self, ohlcv_factory: Callable[[str, Optional[datetime], Optional[datetime]], pd.DataFrame]) -> None:
        self._ohlcv_factory = ohlcv_factory

    def get_ohlcv(self, symbol: str, start: Optional[datetime], end: Optional[datetime]) -> pd.DataFrame:
        return self._ohlcv_factory(symbol, start=start, end=end)

    def get_quote(self, symbol: str) -> Quote:
        df = self._ohlcv_factory(symbol, start=None, end=None)
        if df.empty:
            now = datetime.utcnow()
            return Quote(bid=None, ask=None, last=None, volume=None, timestamp=now)
        last_row = df.iloc[-1]
        last_price = float(last_row["close"])
        timestamp = df.index[-1].to_pydatetime()
        return Quote(
            bid=last_price * 0.999,
            ask=last_price * 1.001,
            last=last_price,
            volume=float(last_row.get("volume", 0.0)),
            timestamp=timestamp,
        )


class _TestRiskGate(RiskGate):
    def _within_market_hours(self) -> bool:  # pragma: no cover - deterministic override
        return True


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    return tmp_path / "trading.db"


@pytest.fixture
def tmp_log_dir(tmp_path: Path) -> Path:
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


@pytest.fixture
def deterministic_config() -> Dict[str, object]:
    symbols = [
        "AAPL",
        "MSFT",
        "GOOG",
        "AMZN",
        "TSLA",
        "NVDA",
        "META",
        "NFLX",
        "INTC",
        "AMD",
    ]
    return {
        "test_ignore_market_hours": True,
        "trading_mode": "paper",
        "symbols": symbols,
        "decision_interval_minutes": 1,
        "scheduler_intervals": {
            "data_refresh": 1,
            "compute_features": 1,
            "make_decisions": 1,
            "execute_orders": 1,
        },
    }


@pytest.fixture
def deterministic_ohlcv() -> Callable[[str, Optional[datetime], Optional[datetime]], pd.DataFrame]:
    def _symbol_seed(symbol: str) -> int:
        digest = hashlib.sha256(symbol.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "little")

    def _factory(
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        periods: int = 60,
        freq: str = "1min",
    ) -> pd.DataFrame:
        seed = _symbol_seed(symbol)
        rng = np.random.default_rng(seed)

        if start is None and end is None:
            end = datetime.utcnow()
            start = end - timedelta(minutes=periods - 1)
        elif start is None:
            start = end - timedelta(minutes=periods - 1)
        elif end is None:
            end = start + timedelta(minutes=periods - 1)

        index = pd.date_range(start=start, periods=periods, freq=freq)
        base = 100.0 + rng.normal(0.0, 1.0, size=periods).cumsum()
        open_prices = base + rng.normal(0.0, 0.2, size=periods)
        close_prices = base + rng.normal(0.0, 0.2, size=periods)
        high_prices = np.maximum(open_prices, close_prices) + rng.uniform(0.0, 0.4, size=periods)
        low_prices = np.minimum(open_prices, close_prices) - rng.uniform(0.0, 0.4, size=periods)
        volume = rng.integers(1000, 5000, size=periods)

        return pd.DataFrame(
            {
                "open": open_prices,
                "high": high_prices,
                "low": low_prices,
                "close": close_prices,
                "volume": volume,
            },
            index=index,
        )

    return _factory


@pytest.fixture
def deterministic_headlines() -> Callable[[Optional[str], int], List[str]]:
    def _factory(symbol: Optional[str] = None, limit: int = 5) -> List[str]:
        prefix = f"{symbol} " if symbol else ""
        headlines = [
            f"{prefix}Earnings outlook steady",
            f"{prefix}Analysts reiterate neutral stance",
            f"{prefix}Sector volatility cools",
            f"{prefix}Institutional flows stabilize",
            f"{prefix}Macro data remains mixed",
        ]
        return headlines[: max(0, int(limit))]

    return _factory


@pytest.fixture
def mock_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def _blocked(*_args: object, **_kwargs: object) -> None:
        raise ConnectionError("Network access disabled for tests")

    monkeypatch.setattr(socket, "create_connection", _blocked)


@pytest.fixture
def mock_no_input(monkeypatch: pytest.MonkeyPatch) -> None:
    def _blocked(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("Input disabled for tests")

    monkeypatch.setattr(builtins, "input", _blocked)


@pytest.fixture
def simulator_broker_factory(
    deterministic_config: Dict[str, object],
) -> Callable[..., SimulatorBroker]:
    def _factory(
        *,
        starting_equity: float = 100000.0,
        symbols: Optional[Iterable[str]] = None,
        slippage: float = 0.0,
        seed: int = 12345,
        price_volatility: float = 0.0,
        price_floor: float = 0.01,
    ) -> SimulatorBroker:
        return SimulatorBroker(
            starting_equity=starting_equity,
            symbols=list(symbols or deterministic_config["symbols"]),
            slippage=slippage,
            seed=seed,
            price_volatility=price_volatility,
            price_floor=price_floor,
        )

    return _factory


@pytest.fixture
def test_trading_system_factory(
    deterministic_config: Dict[str, object],
    deterministic_ohlcv: Callable[[str, Optional[datetime], Optional[datetime]], pd.DataFrame],
    simulator_broker_factory: Callable[..., SimulatorBroker],
    tmp_db_path: Path,
) -> Callable[..., TradingSystem]:
    def _factory(
        *,
        config: Optional[Dict[str, object]] = None,
        broker: Optional[SimulatorBroker] = None,
        market_data: Optional[MarketDataProvider] = None,
        feature_calculator: Optional[BehavioralFeatureCalculator] = None,
        policy: Optional[RulesPolicy] = None,
        risk_gate: Optional[RiskGate] = None,
        position_sizer: Optional[PositionSizer] = None,
        database: Optional[TradingDatabase] = None,
        dry_run: bool = True,
    ) -> TradingSystem:
        merged_config = dict(deterministic_config)
        if config:
            merged_config.update(config)

        broker = broker or simulator_broker_factory(symbols=merged_config["symbols"])
        market_data = market_data or _DeterministicMarketData(deterministic_ohlcv)
        feature_calculator = feature_calculator or BehavioralFeatureCalculator()
        policy = policy or RulesPolicy()
        risk_gate = risk_gate or _TestRiskGate(merged_config, broker)
        position_sizer = position_sizer or PositionSizer(merged_config)
        database = database or TradingDatabase(str(tmp_db_path))

        notifier = NullNotifier(merged_config)

        return TradingSystem(
            merged_config,
            broker,
            market_data,
            feature_calculator,
            policy,
            risk_gate,
            position_sizer,
            database,
            notifier,
            dry_run=dry_run,
        )

    return _factory
