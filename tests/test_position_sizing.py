from __future__ import annotations

import pytest

from risk.position_sizing import PositionSizer


@pytest.fixture
def base_config() -> dict:
    return {"symbols": ["AAPL"]}


@pytest.fixture
def base_positions() -> list[dict]:
    return [{"symbol": "AAPL", "qty": 0.0, "market_value": 0.0}]


def test_strong_buy_gives_2x_position(base_config: dict, base_positions: list[dict]) -> None:
    sizer = PositionSizer(base_config)
    delta = sizer.calculate_shares("AAPL", "STRONG_BUY", 100.0, 10_000.0, base_positions)
    assert delta == 200


def test_buy_gives_1x_position(base_config: dict, base_positions: list[dict]) -> None:
    sizer = PositionSizer(base_config)
    delta = sizer.calculate_shares("AAPL", "BUY", 100.0, 10_000.0, base_positions)
    assert delta == 100


def test_hold_gives_zero_shares(base_config: dict, base_positions: list[dict]) -> None:
    sizer = PositionSizer(base_config)
    delta = sizer.calculate_shares("AAPL", "HOLD", 100.0, 10_000.0, base_positions)
    assert delta == 0


def test_sell_gives_negative_delta(base_config: dict, base_positions: list[dict]) -> None:
    config = dict(base_config)
    config["enable_shorts"] = True
    sizer = PositionSizer(config)
    delta = sizer.calculate_shares("AAPL", "SELL", 100.0, 10_000.0, base_positions)
    assert delta < 0


def test_strong_sell_gives_2x_negative_delta(base_config: dict, base_positions: list[dict]) -> None:
    config = dict(base_config)
    config["enable_shorts"] = True
    sizer = PositionSizer(config)
    delta = sizer.calculate_shares("AAPL", "STRONG_SELL", 100.0, 10_000.0, base_positions)
    assert delta == -200


def test_max_position_pct_constraint_blocks_oversized_orders(
    base_config: dict, base_positions: list[dict]
) -> None:
    config = dict(base_config)
    config["max_position_pct"] = 0.1
    sizer = PositionSizer(config)
    delta = sizer.calculate_shares("AAPL", "BUY", 100.0, 10_000.0, base_positions)
    assert delta == 0


def test_enable_shorts_false_blocks_short_signals(base_config: dict, base_positions: list[dict]) -> None:
    sizer = PositionSizer(base_config)
    delta = sizer.calculate_shares("AAPL", "SELL", 100.0, 10_000.0, base_positions)
    assert delta == 0


def test_equal_weight_allocation_across_symbols(base_config: dict, base_positions: list[dict]) -> None:
    config = dict(base_config)
    config["symbols"] = ["AAPL", "MSFT"]
    positions = [
        {"symbol": "AAPL", "qty": 0.0, "market_value": 0.0},
        {"symbol": "MSFT", "qty": 0.0, "market_value": 0.0},
    ]
    sizer = PositionSizer(config)
    delta_aapl = sizer.calculate_shares("AAPL", "BUY", 50.0, 10_000.0, positions)
    delta_msft = sizer.calculate_shares("MSFT", "BUY", 50.0, 10_000.0, positions)
    assert delta_aapl == delta_msft
