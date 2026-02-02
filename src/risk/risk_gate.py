"""Deterministic risk checks for order gating."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import logging

import pytz


@dataclass(frozen=True)
class _PositionInfo:
    symbol: str
    qty: float
    market_value: Optional[float]


class RiskGate:
    """Run deterministic risk checks before approving orders."""

    def __init__(self, config: Mapping[str, Any], broker_interface: Any) -> None:
        self._config = config or {}
        self._broker = broker_interface
        self._logger = logging.getLogger(__name__)
        self._et_tz = pytz.timezone("US/Eastern")

    def check_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        current_positions: Iterable[Any],
        account_info: Mapping[str, Any],
    ) -> Tuple[bool, List[str]]:
        reasons: List[str] = []
        normalized_side = (side or "").upper()

        equity = self._get_account_value(
            account_info,
            "equity",
            "account_equity",
            "portfolio_value",
            "net_liquidation",
        )
        if equity is None or equity <= 0:
            reasons.append("missing_or_invalid_equity")

        if self._is_kill_switch_triggered(account_info, equity):
            reasons.append("daily_kill_switch")

        if not self._within_market_hours():
            reasons.append("outside_market_hours")

        positions = self._normalize_positions(current_positions)
        price = self._get_price(symbol, account_info)
        if price is None or price <= 0:
            reasons.append("missing_price")

        max_position_pct = self._get_config_float("max_position_pct", "MAX_POSITION_PCT")
        if max_position_pct is not None and equity and price:
            if self._violates_max_position_pct(
                symbol,
                normalized_side,
                qty,
                price,
                positions,
                equity,
                max_position_pct,
            ):
                reasons.append("max_position_pct")

        max_gross_exposure_pct = self._get_config_float(
            "max_gross_exposure_pct", "MAX_GROSS_EXPOSURE_PCT"
        )
        if max_gross_exposure_pct is not None and equity and price:
            if self._violates_gross_exposure_pct(
                symbol,
                normalized_side,
                qty,
                price,
                positions,
                equity,
                max_gross_exposure_pct,
            ):
                reasons.append("max_gross_exposure_pct")

        cooldown_minutes = self._get_config_float("cooldown_minutes", "COOLDOWN_MINUTES")
        if cooldown_minutes:
            last_trade_time = self._get_last_trade_time(symbol, account_info)
            if last_trade_time and self._in_cooldown(last_trade_time, cooldown_minutes):
                reasons.append("symbol_cooldown")

        max_trades_per_day = self._get_config_int(
            "max_trades_per_day", "MAX_TRADES_PER_DAY"
        )
        if max_trades_per_day is not None:
            trades_today = self._get_trades_today(account_info)
            if trades_today is None:
                reasons.append("missing_trades_today")
            elif trades_today >= max_trades_per_day:
                reasons.append("max_trades_per_day")

        if self._pdt_restricted(account_info):
            reasons.append("pdt_restricted")

        if not self._passes_cost_buffer(account_info):
            reasons.append("cost_buffer")

        if reasons:
            self._logger.warning(
                "RiskGate HOLD for %s %s qty=%s: %s",
                symbol,
                normalized_side,
                qty,
                ",".join(reasons),
            )
            return False, reasons

        return True, []

    def _within_market_hours(self) -> bool:
        now_et = datetime.now(tz=self._et_tz)
        if now_et.weekday() >= 5:
            return False
        open_time = time(9, 30)
        close_time = time(16, 0)
        now_time = now_et.time()
        return open_time <= now_time <= close_time

    def _get_config_float(self, *keys: str) -> Optional[float]:
        for key in keys:
            if key in self._config:
                try:
                    return float(self._config[key])
                except (TypeError, ValueError):
                    return None
        return None

    def _get_config_int(self, *keys: str) -> Optional[int]:
        for key in keys:
            if key in self._config:
                try:
                    return int(self._config[key])
                except (TypeError, ValueError):
                    return None
        return None

    def _get_account_value(
        self, account_info: Mapping[str, Any], *keys: str
    ) -> Optional[float]:
        for key in keys:
            if isinstance(account_info, Mapping) and key in account_info:
                try:
                    return float(account_info[key])
                except (TypeError, ValueError):
                    return None
        for key in keys:
            if hasattr(account_info, key):
                try:
                    return float(getattr(account_info, key))
                except (TypeError, ValueError):
                    return None
        return None

    def _normalize_positions(self, current_positions: Iterable[Any]) -> Dict[str, _PositionInfo]:
        positions: Dict[str, _PositionInfo] = {}
        for item in current_positions or []:
            symbol = None
            qty = None
            market_value = None
            if isinstance(item, Mapping):
                symbol = item.get("symbol")
                qty = item.get("qty")
                market_value = item.get("market_value")
            else:
                symbol = getattr(item, "symbol", None)
                qty = getattr(item, "qty", None)
                market_value = getattr(item, "market_value", None)
            if not symbol:
                continue
            try:
                qty_val = float(qty) if qty is not None else 0.0
            except (TypeError, ValueError):
                qty_val = 0.0
            mv_val: Optional[float]
            try:
                mv_val = float(market_value) if market_value is not None else None
            except (TypeError, ValueError):
                mv_val = None
            positions[str(symbol)] = _PositionInfo(
                symbol=str(symbol),
                qty=qty_val,
                market_value=mv_val,
            )
        return positions

    def _get_price(self, symbol: str, account_info: Mapping[str, Any]) -> Optional[float]:
        if isinstance(account_info, Mapping):
            prices = account_info.get("prices")
            if isinstance(prices, Mapping) and symbol in prices:
                try:
                    return float(prices[symbol])
                except (TypeError, ValueError):
                    return None
        if hasattr(self._broker, "get_last_price"):
            try:
                return float(self._broker.get_last_price(symbol))
            except Exception:
                return None
        if hasattr(self._broker, "get_price"):
            try:
                return float(self._broker.get_price(symbol))
            except Exception:
                return None
        return None

    def _violates_max_position_pct(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        positions: Mapping[str, _PositionInfo],
        equity: float,
        max_position_pct: float,
    ) -> bool:
        order_value = qty * price
        current = positions.get(symbol)
        current_value = current.market_value if current and current.market_value is not None else 0.0
        if current is None and current_value == 0.0:
            current_value = 0.0
        if side == "SELL":
            projected_value = current_value - order_value
        else:
            projected_value = current_value + order_value
        pct = abs(projected_value) / equity if equity else 0.0
        return pct > max_position_pct

    def _violates_gross_exposure_pct(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        positions: Mapping[str, _PositionInfo],
        equity: float,
        max_gross_exposure_pct: float,
    ) -> bool:
        order_value = qty * price
        gross = 0.0
        current_value = 0.0
        for pos in positions.values():
            value = pos.market_value
            if value is None:
                value = pos.qty * price if pos.symbol == symbol else 0.0
            gross += abs(value)
            if pos.symbol == symbol:
                current_value = value
        if side == "SELL":
            projected_value = current_value - order_value
        else:
            projected_value = current_value + order_value
        gross = gross - abs(current_value) + abs(projected_value)
        pct = gross / equity if equity else 0.0
        return pct > max_gross_exposure_pct

    def _get_last_trade_time(
        self, symbol: str, account_info: Mapping[str, Any]
    ) -> Optional[datetime]:
        if isinstance(account_info, Mapping):
            last_trades = account_info.get("last_trade_times")
            if isinstance(last_trades, Mapping) and symbol in last_trades:
                return self._coerce_datetime(last_trades[symbol])
            if "last_trade_time" in account_info:
                return self._coerce_datetime(account_info.get("last_trade_time"))
        if hasattr(self._broker, "get_last_trade_time"):
            try:
                return self._coerce_datetime(self._broker.get_last_trade_time(symbol))
            except Exception:
                return None
        return None

    def _coerce_datetime(self, value: Any) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return self._ensure_tz(value)
        if isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
            return self._ensure_tz(parsed)
        return None

    def _ensure_tz(self, dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return pytz.UTC.localize(dt)
        return dt

    def _in_cooldown(self, last_trade_time: datetime, cooldown_minutes: float) -> bool:
        now = datetime.now(tz=pytz.UTC)
        return now - last_trade_time < timedelta(minutes=cooldown_minutes)

    def _get_trades_today(self, account_info: Mapping[str, Any]) -> Optional[int]:
        if isinstance(account_info, Mapping):
            for key in (
                "trades_today",
                "trades_count_today",
                "trade_count_today",
                "orders_today",
            ):
                if key in account_info:
                    value = account_info[key]
                    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
                        try:
                            return len(list(value))
                        except TypeError:
                            return None
                    try:
                        return int(value)
                    except (TypeError, ValueError):
                        return None
        if hasattr(self._broker, "get_trades_today"):
            try:
                return int(self._broker.get_trades_today())
            except Exception:
                return None
        return None

    def _pdt_restricted(self, account_info: Mapping[str, Any]) -> bool:
        if isinstance(account_info, Mapping):
            for key in (
                "pdt_restricted",
                "pdt_flag",
                "pattern_day_trader",
                "is_pattern_day_trader",
            ):
                if key in account_info:
                    try:
                        return bool(account_info[key])
                    except Exception:
                        return True
            if "day_trades_remaining" in account_info:
                try:
                    return int(account_info["day_trades_remaining"]) <= 0
                except (TypeError, ValueError):
                    return True
        if hasattr(self._broker, "is_pdt_restricted"):
            try:
                return bool(self._broker.is_pdt_restricted())
            except Exception:
                return True
        return False

    def _passes_cost_buffer(self, account_info: Mapping[str, Any]) -> bool:
        cost_buffer = self._get_config_float("cost_buffer", "COST_BUFFER")
        if cost_buffer is None:
            return True
        expected_edge = None
        if isinstance(account_info, Mapping):
            expected_edge = account_info.get("expected_edge")
        if expected_edge is None and hasattr(self._broker, "get_expected_edge"):
            try:
                expected_edge = self._broker.get_expected_edge()
            except Exception:
                expected_edge = None
        try:
            if expected_edge is None:
                return False
            return float(expected_edge) > cost_buffer
        except (TypeError, ValueError):
            return False

    def _is_kill_switch_triggered(
        self, account_info: Mapping[str, Any], equity: Optional[float]
    ) -> bool:
        max_daily_loss_pct = self._get_config_float(
            "max_daily_loss_pct", "MAX_DAILY_LOSS_PCT"
        )
        if max_daily_loss_pct is None:
            return False
        daily_loss_pct = self._get_daily_loss_pct(account_info, equity)
        if daily_loss_pct is None:
            return False
        return daily_loss_pct > max_daily_loss_pct

    def _get_daily_loss_pct(
        self, account_info: Mapping[str, Any], equity: Optional[float]
    ) -> Optional[float]:
        if isinstance(account_info, Mapping):
            if "daily_loss_pct" in account_info:
                try:
                    return float(account_info["daily_loss_pct"])
                except (TypeError, ValueError):
                    return None
            if "daily_loss" in account_info:
                try:
                    loss = float(account_info["daily_loss"])
                except (TypeError, ValueError):
                    return None
                return abs(loss) / equity if equity else None
            for key in ("daily_pnl", "pnl_today", "realized_pnl_today"):
                if key in account_info:
                    try:
                        pnl = float(account_info[key])
                    except (TypeError, ValueError):
                        return None
                    loss = max(0.0, -pnl)
                    return loss / equity if equity else None
        if hasattr(self._broker, "get_daily_loss_pct"):
            try:
                return float(self._broker.get_daily_loss_pct())
            except Exception:
                return None
        return None
