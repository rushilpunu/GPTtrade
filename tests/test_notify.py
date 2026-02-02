from __future__ import annotations

from datetime import datetime

import pytest
import requests

from observability.notify import (
    DiscordNotifier,
    NullNotifier,
    TradingNotifier,
    create_notifier,
)


class _Response:
    def __init__(self, status_code: int, text: str = "") -> None:
        self.status_code = status_code
        self.text = text


def test_discord_notifier_nonblocking(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(requests, "post", _boom)

    notifier = DiscordNotifier("https://example.com/webhook")

    assert notifier.send("Title", "Body") is False


def test_discord_notifier_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def _ok(*_args: object, **_kwargs: object) -> _Response:
        return _Response(204)

    monkeypatch.setattr(requests, "post", _ok)

    notifier = DiscordNotifier("https://example.com/webhook")

    assert notifier.send("Title", "Body") is True


def test_trading_notifier_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    def _blocked(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Network call attempted")

    monkeypatch.setattr(requests, "post", _blocked)

    notifier = TradingNotifier({"enable_discord": False})

    assert (
        notifier.notify_trade(
            "AAPL",
            "buy",
            10.0,
            150.0,
            0.8,
            "Test rationale",
            "corr-1",
        )
        is False
    )
    assert (
        notifier.notify_fill(
            "AAPL",
            "buy",
            10.0,
            149.5,
            150.0,
            "corr-2",
        )
        is False
    )
    assert notifier.notify_risk_block("AAPL", "buy", ["rule"], "corr-3") is False
    assert notifier.notify_kill_switch("limit", -250.0) is False
    assert notifier.notify_large_move("AAPL", 10.0, 155.0) is False


def test_null_notifier() -> None:
    notifier = NullNotifier()

    assert (
        notifier.notify_trade(
            "AAPL",
            "buy",
            10.0,
            150.0,
            0.8,
            "Test rationale",
            "corr-1",
        )
        is False
    )
    assert (
        notifier.notify_fill(
            "AAPL",
            "buy",
            10.0,
            149.5,
            150.0,
            "corr-2",
        )
        is False
    )
    assert notifier.notify_risk_block("AAPL", "buy", ["rule"], "corr-3") is False
    assert notifier.notify_kill_switch("limit", -250.0) is False
    assert notifier.notify_large_move("AAPL", 10.0, 155.0) is False


def test_create_notifier_factory() -> None:
    enabled = create_notifier(
        {"enable_discord": True, "discord_webhook_url": "https://example.com/webhook"}
    )
    assert isinstance(enabled, TradingNotifier)

    disabled = create_notifier({"enable_discord": False})
    assert isinstance(disabled, NullNotifier)


def test_notify_trade_format(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _capture(url: str, json: object, timeout: float) -> _Response:
        captured["url"] = url
        captured["payload"] = json
        captured["timeout"] = timeout
        return _Response(204)

    monkeypatch.setattr(requests, "post", _capture)

    notifier = TradingNotifier(
        {"enable_discord": True, "discord_webhook_url": "https://example.com/webhook"}
    )

    assert (
        notifier.notify_trade(
            "AAPL",
            "buy",
            10.0,
            150.0,
            0.82,
            "Strong momentum",
            "corr-123",
        )
        is True
    )

    payload = captured["payload"]
    assert isinstance(payload, dict)
    embeds = payload["embeds"]
    assert isinstance(embeds, list)
    assert len(embeds) == 1

    embed = embeds[0]
    assert embed["title"] == "Trade Signal: AAPL"
    assert embed["description"] == "Strong momentum"
    assert embed["color"] == 0x00FF00
    assert "timestamp" in embed
    datetime.fromisoformat(embed["timestamp"])

    fields = embed["fields"]
    assert fields == [
        {"name": "Side", "value": "buy", "inline": True},
        {"name": "Qty", "value": "10.0", "inline": True},
        {"name": "Price", "value": "150.0", "inline": True},
        {"name": "Confidence", "value": "0.82", "inline": True},
        {"name": "Correlation ID", "value": "corr-123", "inline": False},
    ]
