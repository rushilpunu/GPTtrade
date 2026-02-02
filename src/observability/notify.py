"""Discord-based trading notifications."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Mapping

import requests

logger = logging.getLogger(__name__)


class DiscordNotifier:
    """Send Discord webhook notifications."""

    def __init__(self, webhook_url: str, timeout: float = 5.0) -> None:
        self._webhook_url = webhook_url
        self._timeout = timeout

    def send(
        self,
        title: str,
        message: str,
        color: int = 0x00FF00,
        fields: list[dict] | None = None,
    ) -> bool:
        """Send an embed message to Discord.

        Returns True on success, False otherwise. Never raises.
        """

        payload = {
            "embeds": [
                {
                    "title": title,
                    "description": message,
                    "color": color,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    **({"fields": fields} if fields else {}),
                }
            ]
        }

        try:
            response = requests.post(
                self._webhook_url,
                json=payload,
                timeout=self._timeout,
            )
            if 200 <= response.status_code < 300:
                return True
            logger.warning(
                "Discord webhook returned non-success status",
                extra={"status_code": response.status_code, "body": response.text},
            )
        except Exception:
            logger.exception("Failed to send Discord webhook")
        return False


class TradingNotifier:
    """Trading-specific notifications backed by Discord webhooks."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        cfg = dict(config or {})
        self._enabled = bool(cfg.get("enable_discord", False))
        self._webhook_url = cfg.get("discord_webhook_url") or os.getenv(
            "DISCORD_WEBHOOK_URL"
        )
        if self._enabled and not self._webhook_url:
            logger.warning(
                "Discord notifications enabled but webhook URL missing; disabling"
            )
            self._enabled = False

        self._notify_on_trade = bool(
            cfg.get("notify_on_trade", self._enabled)
        )
        self._notify_on_risk_blocks = bool(
            cfg.get("notify_on_risk_blocks", True)
        )
        self._notify_on_large_move = bool(
            cfg.get("notify_on_large_move", True)
        )
        self._large_move_threshold_pct = float(
            cfg.get("large_move_threshold_pct", 3.0)
        )
        self._notify_on_kill_switch = bool(
            cfg.get("notify_on_kill_switch", True)
        )

        self._discord = (
            DiscordNotifier(self._webhook_url)
            if self._enabled and self._webhook_url
            else None
        )

    def notify_trade(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        confidence: float,
        rationale: str,
        correlation_id: str,
    ) -> bool:
        if not (self._enabled and self._notify_on_trade and self._discord):
            return False
        try:
            fields = [
                {"name": "Side", "value": side, "inline": True},
                {"name": "Qty", "value": str(qty), "inline": True},
                {"name": "Price", "value": str(price), "inline": True},
                {"name": "Confidence", "value": str(confidence), "inline": True},
                {"name": "Correlation ID", "value": correlation_id, "inline": False},
            ]
            return self._discord.send(
                title=f"Trade Signal: {symbol}",
                message=rationale,
                color=0x00FF00 if side.lower() == "buy" else 0xFF0000,
                fields=fields,
            )
        except Exception:
            logger.exception("Failed to notify trade")
            return False

    def notify_fill(
        self,
        symbol: str,
        side: str,
        qty: float,
        fill_price: float,
        expected_price: float,
        correlation_id: str,
    ) -> bool:
        if not (self._enabled and self._notify_on_trade and self._discord):
            return False
        try:
            fields = [
                {"name": "Side", "value": side, "inline": True},
                {"name": "Qty", "value": str(qty), "inline": True},
                {"name": "Fill Price", "value": str(fill_price), "inline": True},
                {
                    "name": "Expected Price",
                    "value": str(expected_price),
                    "inline": True,
                },
                {"name": "Correlation ID", "value": correlation_id, "inline": False},
            ]
            return self._discord.send(
                title=f"Trade Fill: {symbol}",
                message="Order filled.",
                color=0x3498DB,
                fields=fields,
            )
        except Exception:
            logger.exception("Failed to notify fill")
            return False

    def notify_risk_block(
        self,
        symbol: str,
        action: str,
        reasons: list[str],
        correlation_id: str,
    ) -> bool:
        if not (self._enabled and self._notify_on_risk_blocks and self._discord):
            return False
        try:
            fields = [
                {"name": "Action", "value": action, "inline": True},
                {"name": "Reasons", "value": "\n".join(reasons), "inline": False},
                {"name": "Correlation ID", "value": correlation_id, "inline": False},
            ]
            return self._discord.send(
                title=f"Risk Block: {symbol}",
                message="Trade blocked by risk controls.",
                color=0xE67E22,
                fields=fields,
            )
        except Exception:
            logger.exception("Failed to notify risk block")
            return False

    def notify_kill_switch(self, reason: str, daily_pnl: float) -> bool:
        if not (self._enabled and self._notify_on_kill_switch and self._discord):
            return False
        try:
            fields = [
                {"name": "Reason", "value": reason, "inline": False},
                {"name": "Daily PnL", "value": str(daily_pnl), "inline": True},
            ]
            return self._discord.send(
                title="Kill Switch Activated",
                message="Trading halted.",
                color=0x8E44AD,
                fields=fields,
            )
        except Exception:
            logger.exception("Failed to notify kill switch")
            return False

    def notify_large_move(
        self, symbol: str, price_change_pct: float, current_price: float
    ) -> bool:
        if not (self._enabled and self._notify_on_large_move and self._discord):
            return False
        if abs(price_change_pct) < self._large_move_threshold_pct:
            return False
        try:
            fields = [
                {
                    "name": "Change %",
                    "value": f"{price_change_pct:.2f}%",
                    "inline": True,
                },
                {
                    "name": "Current Price",
                    "value": str(current_price),
                    "inline": True,
                },
            ]
            return self._discord.send(
                title=f"Large Move: {symbol}",
                message="Price moved beyond threshold.",
                color=0xF1C40F,
                fields=fields,
            )
        except Exception:
            logger.exception("Failed to notify large move")
            return False


class NullNotifier:
    """No-op notifier used when notifications are disabled."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self._config = dict(config or {})

    def notify_trade(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        confidence: float,
        rationale: str,
        correlation_id: str,
    ) -> bool:
        return False

    def notify_fill(
        self,
        symbol: str,
        side: str,
        qty: float,
        fill_price: float,
        expected_price: float,
        correlation_id: str,
    ) -> bool:
        return False

    def notify_risk_block(
        self,
        symbol: str,
        action: str,
        reasons: list[str],
        correlation_id: str,
    ) -> bool:
        return False

    def notify_kill_switch(self, reason: str, daily_pnl: float) -> bool:
        return False

    def notify_large_move(
        self, symbol: str, price_change_pct: float, current_price: float
    ) -> bool:
        return False


def create_notifier(config: Mapping[str, Any]) -> TradingNotifier | NullNotifier:
    """Factory for a configured trading notifier."""

    cfg = dict(config or {})
    enabled = bool(cfg.get("enable_discord", False))
    webhook_url = cfg.get("discord_webhook_url") or os.getenv("DISCORD_WEBHOOK_URL")
    if enabled and webhook_url:
        return TradingNotifier(cfg)
    return NullNotifier(cfg)
