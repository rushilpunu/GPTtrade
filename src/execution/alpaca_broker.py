"""Alpaca broker implementation."""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
import uuid
from datetime import datetime
from typing import Any, Iterable, List, Mapping, Optional

from alpaca_trade_api import REST
from alpaca_trade_api.rest import APIError

from .broker_interface import AccountInfo, BrokerInterface, OrderResult, OrderStatus, Position


class AlpacaBroker(BrokerInterface):
    """BrokerInterface implementation for Alpaca."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self._config = config or {}
        self._logger = logging.getLogger(__name__)
        self._enable_live = bool(self._config.get("ENABLE_LIVE_TRADING", False))
        self._is_live = self._resolve_live_mode()

        key_id = os.getenv("ALPACA_API_KEY_ID") or os.getenv("APCA_API_KEY_ID")
        secret_key = os.getenv("ALPACA_API_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
        if not key_id or not secret_key:
            raise ValueError("Missing Alpaca API credentials in environment variables.")

        base_url = self._resolve_base_url()
        data_url = self._resolve_data_url()

        self._max_retries = int(self._config.get("alpaca_max_retries", 5))
        self._backoff_base = float(self._config.get("alpaca_backoff_base_seconds", 1.0))
        self._backoff_max = float(self._config.get("alpaca_backoff_max_seconds", 30.0))

        self._api = REST(key_id, secret_key, base_url=base_url, data_url=data_url)

        self._logger.info(
            "Initialized AlpacaBroker mode=%s base_url=%s data_url=%s",
            "live" if self._is_live else "paper",
            base_url,
            data_url or "default",
        )

    def get_account(self) -> AccountInfo:
        account = self._call_api(self._api.get_account)
        equity = float(getattr(account, "equity", 0.0) or 0.0)
        buying_power = float(getattr(account, "buying_power", 0.0) or 0.0)
        pdt_flag = bool(getattr(account, "pattern_day_trader", False))
        daytrade_count = getattr(account, "daytrade_count", None)
        daytrade_count_val = int(daytrade_count) if daytrade_count is not None else 0
        if pdt_flag:
            day_trades_remaining = 0
        else:
            day_trades_remaining = max(0, 3 - daytrade_count_val)
        return AccountInfo(
            equity=equity,
            buying_power=buying_power,
            pdt_flag=pdt_flag,
            day_trades_remaining=day_trades_remaining,
        )

    def get_positions(self) -> List[Position]:
        positions = self._call_api(self._api.list_positions)
        results: List[Position] = []
        for pos in positions or []:
            symbol = str(getattr(pos, "symbol", "") or "")
            if not symbol:
                continue
            qty_val = float(getattr(pos, "qty", 0.0) or 0.0)
            side = str(getattr(pos, "side", "") or "").lower()
            if side == "short":
                qty_val = -abs(qty_val)
            avg_price = self._coerce_float(getattr(pos, "avg_entry_price", None))
            market_value = self._coerce_float(getattr(pos, "market_value", None))
            unrealized_pl = self._coerce_float(getattr(pos, "unrealized_pl", None))
            results.append(
                Position(
                    symbol=symbol,
                    qty=qty_val,
                    avg_price=avg_price,
                    market_value=market_value,
                    unrealized_pl=unrealized_pl,
                )
            )
        return results

    def get_last_price(self, symbol: str) -> float:
        trade = self._call_api(self._api.get_latest_trade, symbol)
        price = getattr(trade, "price", None)
        if price is None:
            price = getattr(trade, "p", None)
        if price is None:
            raise ValueError(f"Missing price for {symbol}.")
        return float(price)

    def submit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str,
        limit_price: Optional[float] = None,
    ) -> OrderResult:
        normalized_side = (side or "").lower()
        normalized_type = (order_type or "").lower()
        if normalized_side not in {"buy", "sell"}:
            raise ValueError(f"Invalid side: {side}")
        if normalized_type not in {"market", "limit"}:
            raise ValueError(f"Invalid order type: {order_type}")
        if normalized_type == "limit" and limit_price is None:
            raise ValueError("Limit price required for limit orders.")

        if self._is_live:
            self._assert_live_trading_confirmation()

        client_order_id = self._generate_idempotency_key(symbol)
        order_kwargs = {
            "symbol": symbol,
            "side": normalized_side,
            "type": normalized_type,
            "qty": qty,
            "client_order_id": client_order_id,
        }
        if normalized_type == "limit":
            order_kwargs["limit_price"] = limit_price

        order = self._call_api(self._api.submit_order, **order_kwargs)

        submitted_at = self._coerce_datetime(getattr(order, "submitted_at", None))
        filled_qty = self._coerce_float(getattr(order, "filled_qty", None))
        filled_avg_price = self._coerce_float(getattr(order, "filled_avg_price", None))
        return OrderResult(
            order_id=str(getattr(order, "id", "") or ""),
            status=str(getattr(order, "status", "") or ""),
            submitted_at=submitted_at,
            filled_qty=filled_qty,
            filled_avg_price=filled_avg_price,
        )

    def cancel_order(self, order_id: str) -> bool:
        try:
            self._call_api(self._api.cancel_order, order_id)
            return True
        except APIError as exc:
            status_code = getattr(exc, "status_code", None)
            if status_code == 404:
                return False
            raise

    def get_order_status(self, order_id: str) -> OrderStatus:
        order = self._call_api(self._api.get_order, order_id)
        filled_qty = self._coerce_float(getattr(order, "filled_qty", None))
        filled_avg_price = self._coerce_float(getattr(order, "filled_avg_price", None))
        qty_val = self._coerce_float(getattr(order, "qty", None))
        remaining_qty = None
        if qty_val is not None:
            remaining_qty = max(0.0, qty_val - (filled_qty or 0.0))
        updated_at = self._coerce_datetime(getattr(order, "updated_at", None))
        if updated_at is None:
            updated_at = self._coerce_datetime(getattr(order, "submitted_at", None))
        return OrderStatus(
            order_id=str(getattr(order, "id", "") or ""),
            status=str(getattr(order, "status", "") or ""),
            filled_qty=filled_qty,
            filled_avg_price=filled_avg_price,
            remaining_qty=remaining_qty,
            updated_at=updated_at,
        )

    def is_market_open(self) -> bool:
        clock = self._call_api(self._api.get_clock)
        return bool(getattr(clock, "is_open", False))

    def _resolve_live_mode(self) -> bool:
        trading_mode = str(self._config.get("trading_mode", "paper")).lower()
        if trading_mode == "live" and not self._enable_live:
            self._logger.warning(
                "trading_mode is live but ENABLE_LIVE_TRADING is false; forcing paper mode."
            )
        return trading_mode == "live" and self._enable_live

    def _resolve_base_url(self) -> str:
        if self._is_live:
            return (
                self._config.get("ALPACA_LIVE_URL")
                or self._config.get("alpaca_live_url")
                or "https://api.alpaca.markets"
            )
        return (
            self._config.get("ALPACA_PAPER_URL")
            or self._config.get("alpaca_paper_url")
            or "https://paper-api.alpaca.markets"
        )

    def _resolve_data_url(self) -> Optional[str]:
        return self._config.get("ALPACA_DATA_URL") or self._config.get("alpaca_data_url")

    def _assert_live_trading_confirmation(self) -> None:
        token = os.getenv("LIVE_TRADING_CONFIRMATION_TOKEN")
        expected_hash = self._get_confirmation_hash()
        if not token or not expected_hash:
            raise PermissionError("Live trading confirmation token missing.")
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        if not hmac.compare_digest(token_hash, expected_hash):
            raise PermissionError("Live trading confirmation token mismatch.")

    def _get_confirmation_hash(self) -> Optional[str]:
        for key in (
            "LIVE_TRADING_CONFIRMATION_HASH",
            "live_trading_confirmation_hash",
            "LIVE_TRADING_CONFIRMATION_TOKEN_HASH",
            "live_trading_confirmation_token_hash",
        ):
            value = self._config.get(key)
            if value:
                return str(value)
        return None

    def _generate_idempotency_key(self, symbol: str) -> str:
        return f"gpttrade-{symbol.lower()}-{uuid.uuid4().hex}"

    def _call_api(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        name = getattr(fn, "__name__", "api_call")
        self._logger.info(
            "Alpaca API call %s args=%s kwargs=%s", name, self._safe_log(args), self._safe_log(kwargs)
        )
        attempt = 0
        backoff = self._backoff_base
        while True:
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                if self._is_rate_limit_error(exc) and attempt < self._max_retries:
                    self._logger.warning(
                        "Alpaca rate limit on %s; retrying in %.2fs (attempt %s/%s)",
                        name,
                        backoff,
                        attempt + 1,
                        self._max_retries,
                    )
                    time.sleep(backoff)
                    attempt += 1
                    backoff = min(backoff * 2.0, self._backoff_max)
                    continue
                self._logger.exception("Alpaca API call failed %s", name)
                raise

    def _is_rate_limit_error(self, exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if status_code == 429:
            return True
        response = getattr(exc, "response", None)
        if response is not None and getattr(response, "status_code", None) == 429:
            return True
        message = str(exc).lower()
        return "rate limit" in message or "too many requests" in message

    @staticmethod
    def _coerce_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_datetime(value: Any) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        try:
            return datetime.fromisoformat(str(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_log(payload: Any) -> Any:
        if isinstance(payload, Mapping):
            return {k: ("***" if "secret" in str(k).lower() else v) for k, v in payload.items()}
        if isinstance(payload, Iterable) and not isinstance(payload, (str, bytes)):
            return list(payload)
        return payload
