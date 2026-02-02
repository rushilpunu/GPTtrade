"""Smoke test runner for GPTtrade."""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from agent.policy import RulesPolicy
from data.market_data import MarketDataProvider, Quote
from execution.simulator_broker import SimulatorBroker
from features.behavioral_features import BehavioralFeatureCalculator
from main import TradingSystem
from observability.notify import NullNotifier
from risk.position_sizing import PositionSizer
from risk.risk_gate import RiskGate
from storage.db import TradingDatabase


@dataclass(frozen=True)
class _DeterministicMarketDataConfig:
    base_time: datetime
    periods: int = 60
    freq: str = "1min"


class DeterministicMarketData(MarketDataProvider):
    def __init__(self, config: _DeterministicMarketDataConfig) -> None:
        self._config = config

    def get_ohlcv(self, symbol: str, start: Any, end: Any) -> pd.DataFrame:
        return self._build_frame(symbol)

    def get_quote(self, symbol: str) -> Quote:
        frame = self._build_frame(symbol)
        if frame.empty:
            now = self._config.base_time
            return Quote(bid=None, ask=None, last=None, volume=None, timestamp=now)
        last_row = frame.iloc[-1]
        last_price = float(last_row["close"])
        timestamp = frame.index[-1].to_pydatetime()
        return Quote(
            bid=last_price * 0.999,
            ask=last_price * 1.001,
            last=last_price,
            volume=float(last_row.get("volume", 0.0)),
            timestamp=timestamp,
        )

    def _build_frame(self, symbol: str) -> pd.DataFrame:
        seed = self._symbol_seed(symbol)
        rng = np.random.default_rng(seed)
        end = self._config.base_time
        start = end - timedelta(minutes=self._config.periods - 1)
        index = pd.date_range(start=start, periods=self._config.periods, freq=self._config.freq)
        base = 100.0 + rng.normal(0.0, 1.0, size=self._config.periods).cumsum()
        open_prices = base + rng.normal(0.0, 0.2, size=self._config.periods)
        close_prices = base + rng.normal(0.0, 0.2, size=self._config.periods)
        high_prices = np.maximum(open_prices, close_prices) + rng.uniform(0.0, 0.4, size=self._config.periods)
        low_prices = np.minimum(open_prices, close_prices) - rng.uniform(0.0, 0.4, size=self._config.periods)
        volume = rng.integers(1000, 5000, size=self._config.periods)
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

    @staticmethod
    def _symbol_seed(symbol: str) -> int:
        digest = hashlib.sha256(str(symbol).encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "little")


class SmokeRiskGate(RiskGate):
    def _within_market_hours(self) -> bool:
        if self._config.get("test_ignore_market_hours"):
            return True
        return super()._within_market_hours()


def _build_config(db_path: str, symbols: Iterable[str]) -> Dict[str, object]:
    return {
        "test_ignore_market_hours": True,
        "trading_mode": "paper",
        "symbols": list(symbols),
        "decision_interval_minutes": 1,
        "scheduler_intervals": {
            "data_refresh": 1,
            "compute_features": 1,
            "make_decisions": 1,
            "execute_orders": 1,
        },
        "starting_equity": 100000.0,
        "simulator_slippage": 0.0,
        "simulator_seed": 42,
        "simulator_price_volatility": 0.0,
        "simulator_price_floor": 0.01,
        "db_path": db_path,
    }


def _build_system(config: Dict[str, object], base_time: datetime) -> TradingSystem:
    broker = SimulatorBroker(
        starting_equity=float(config.get("starting_equity", 100000.0)),
        symbols=config.get("symbols"),
        slippage=float(config.get("simulator_slippage", 0.0)),
        seed=int(config.get("simulator_seed", 42)),
        price_volatility=float(config.get("simulator_price_volatility", 0.0)),
        price_floor=float(config.get("simulator_price_floor", 0.01)),
    )
    market_data = DeterministicMarketData(
        _DeterministicMarketDataConfig(base_time=base_time, periods=60, freq="1min")
    )
    feature_calculator = BehavioralFeatureCalculator()
    policy = RulesPolicy()
    risk_gate = SmokeRiskGate(config, broker)
    position_sizer = PositionSizer(config)
    database = TradingDatabase(str(config.get("db_path")))
    notifier = NullNotifier(config)
    return TradingSystem(
        config,
        broker,
        market_data,
        feature_calculator,
        policy,
        risk_gate,
        position_sizer,
        database,
        notifier,
        dry_run=False,
    )


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPTtrade smoke test")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
    base_time = datetime(2024, 1, 2, 15, 30, 0)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/smoke.db"
            config = _build_config(db_path, symbols)
            system = _build_system(config, base_time)
            system.run_cycle()
    except Exception as exc:
        message = f"SMOKE TEST FAILED: {exc}"
        print(message)
        if args.verbose:
            logging.exception("Smoke test failure")
        return 1

    print("SMOKE TEST PASSED: one simulator cycle completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
