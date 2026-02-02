"""Position sizing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional

import logging
import math


@dataclass(frozen=True)
class _PositionInfo:
    symbol: str
    qty: float
    market_value: Optional[float]


class PositionSizer:
    """Compute position sizes based on signal strength and risk constraints."""

    _SIGNAL_MULTIPLIERS = {
        "STRONG_BUY": 2.0,
        "BUY": 1.0,
        "HOLD": 0.0,
        "SELL": -1.0,
        "STRONG_SELL": -2.0,
    }

    def __init__(self, config: Mapping[str, Any]) -> None:
        self._config = config or {}
        self._logger = logging.getLogger(__name__)

    def calculate_shares(
        self,
        symbol: str,
        signal_strength: Any,
        current_price: float,
        account_equity: float,
        current_positions: Iterable[Any],
    ) -> int:
        """Return share delta for the given signal, or 0 if constrained."""
        if not symbol:
            return 0
        if current_price is None or current_price <= 0:
            return 0
        if account_equity is None or account_equity <= 0:
            return 0

        multiplier = self._normalize_signal(signal_strength)
        if multiplier == 0.0:
            return 0

        enable_shorts = self._get_config_bool("enable_shorts", "ENABLE_SHORTS")
        if multiplier < 0 and not enable_shorts:
            return 0

        positions = self._normalize_positions(current_positions)
        current_qty = positions.get(symbol, _PositionInfo(symbol, 0.0, None)).qty

        num_symbols = self._get_num_symbols(current_positions)
        if num_symbols <= 0:
            return 0

        base_allocation = account_equity / float(num_symbols)
        desired_position_value = base_allocation * multiplier
        desired_position_shares = desired_position_value / current_price

        desired_position_shares = self._apply_atr_sizing(
            symbol,
            desired_position_shares,
            current_price,
            account_equity,
        )

        target_shares = self._round_shares_toward_zero(desired_position_shares)
        if target_shares == current_qty:
            return 0

        if not enable_shorts and target_shares < 0:
            return 0

        max_position_pct = self._get_config_float("max_position_pct", "MAX_POSITION_PCT")
        if max_position_pct is not None:
            max_value = max_position_pct * account_equity
            resulting_value = abs(target_shares) * current_price
            if resulting_value > max_value:
                return 0

        delta = target_shares - current_qty
        if delta == 0:
            return 0

        return int(delta)

    def _normalize_signal(self, signal_strength: Any) -> float:
        if signal_strength is None:
            return 0.0
        if isinstance(signal_strength, str):
            key = signal_strength.strip().upper()
            if key in self._SIGNAL_MULTIPLIERS:
                return self._SIGNAL_MULTIPLIERS[key]
            try:
                return float(signal_strength)
            except ValueError:
                return 0.0
        try:
            value = float(signal_strength)
        except (TypeError, ValueError):
            return 0.0
        if math.isfinite(value):
            return value
        return 0.0

    def _get_num_symbols(self, current_positions: Iterable[Any]) -> int:
        symbols = self._config.get("symbols")
        if isinstance(symbols, (list, tuple)):
            return len(symbols) if symbols else 0
        if isinstance(symbols, int):
            return symbols
        positions = self._normalize_positions(current_positions)
        if positions:
            return len(positions)
        return 1

    def _apply_atr_sizing(
        self,
        symbol: str,
        desired_shares: float,
        current_price: float,
        account_equity: float,
    ) -> float:
        if not self._get_config_bool(
            "use_atr_sizing",
            "volatility_adjusted_sizing",
            "use_volatility_sizing",
        ):
            return desired_shares

        atr = self._get_atr(symbol)
        if atr is None or atr <= 0:
            return desired_shares

        adjusted = desired_shares
        atr_target = self._get_config_float("atr_target", "atr_target_abs")
        atr_target_pct = self._get_config_float("atr_target_pct", "atr_target_price_pct")
        if atr_target is None and atr_target_pct:
            atr_target = current_price * atr_target_pct
        if atr_target and atr_target > 0:
            multiplier = atr_target / atr
            max_mult = self._get_config_float("atr_max_multiplier")
            min_mult = self._get_config_float("atr_min_multiplier")
            if max_mult is None:
                max_mult = 1.0
            if min_mult is None:
                min_mult = 0.0
            multiplier = max(min(multiplier, max_mult), min_mult)
            adjusted = adjusted * multiplier

        atr_risk_pct = self._get_config_float("atr_risk_pct")
        if atr_risk_pct and atr_risk_pct > 0:
            risk_budget = account_equity * atr_risk_pct
            atr_cap = risk_budget / atr
            adjusted = self._clamp_shares_by_abs(adjusted, atr_cap)

        return adjusted

    def _get_atr(self, symbol: str) -> Optional[float]:
        for key in ("atr_by_symbol", "atr_values", "atr_map"):
            source = self._config.get(key)
            if isinstance(source, Mapping) and symbol in source:
                return self._to_float(source.get(symbol))
        atr_value = self._config.get("atr")
        if isinstance(atr_value, Mapping) and symbol in atr_value:
            return self._to_float(atr_value.get(symbol))
        if atr_value is not None and not isinstance(atr_value, Mapping):
            return self._to_float(atr_value)
        return None

    def _clamp_shares_by_abs(self, shares: float, cap: float) -> float:
        if cap <= 0:
            return 0.0
        sign = 1.0 if shares >= 0 else -1.0
        return sign * min(abs(shares), cap)

    def _round_shares_toward_zero(self, shares: float) -> int:
        if shares >= 0:
            return int(math.floor(shares))
        return int(math.ceil(shares))

    def _get_config_float(self, *keys: str) -> Optional[float]:
        for key in keys:
            if key in self._config:
                return self._to_float(self._config.get(key))
        return None

    def _get_config_bool(self, *keys: str) -> bool:
        for key in keys:
            if key in self._config:
                value = self._config.get(key)
                if isinstance(value, bool):
                    return value
                if isinstance(value, str):
                    return value.strip().lower() in {"1", "true", "yes", "on"}
                return bool(value)
        return False

    def _to_float(self, value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _normalize_positions(self, current_positions: Iterable[Any]) -> Dict[str, _PositionInfo]:
        positions: Dict[str, _PositionInfo] = {}
        for item in current_positions or []:
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
            qty_val = self._to_float(qty) or 0.0
            mv_val = self._to_float(market_value)
            positions[str(symbol)] = _PositionInfo(
                symbol=str(symbol),
                qty=qty_val,
                market_value=mv_val,
            )
        return positions
