"""In-memory broker simulation for paper trading and backtests."""

from __future__ import annotations

import logging
import random
import uuid
from datetime import datetime
from typing import Dict, Iterable, List, Optional

from .broker_interface import AccountInfo, BrokerInterface, OrderResult, OrderStatus, Position


class SimulatorBroker(BrokerInterface):
    """BrokerInterface implementation for offline simulation."""

    def __init__(
        self,
        starting_equity: float = 100000.0,
        symbols: Optional[Iterable[str]] = None,
        slippage: float = 0.001,
        seed: Optional[int] = None,
        price_volatility: float = 0.002,
        price_floor: float = 0.01,
    ) -> None:
        self._logger = logging.getLogger(__name__)
        self._symbols = [str(sym) for sym in (symbols or [])]
        self._cash = float(starting_equity)
        self._slippage = float(slippage)
        self._price_volatility = float(price_volatility)
        self._price_floor = float(price_floor)
        self._rng = random.Random(seed)
        self._positions: Dict[str, Dict[str, object]] = {}
        self._orders: Dict[str, Dict[str, object]] = {}
        self._last_prices: Dict[str, float] = {}
        self._trade_history: List[Dict[str, object]] = []

        for symbol in self._symbols:
            self._last_prices[symbol] = self._initial_price(symbol)

        self._logger.info(
            "Initialized SimulatorBroker equity=%.2f symbols=%s slippage=%.4f",
            self._cash,
            self._symbols,
            self._slippage,
        )

    @property
    def trade_history(self) -> List[Dict[str, object]]:
        return list(self._trade_history)

    def get_account(self) -> AccountInfo:
        equity = self._cash + sum(self._position_market_value(symbol) for symbol in self._positions)
        self._logger.info("Simulator get_account equity=%.2f cash=%.2f", equity, self._cash)
        return AccountInfo(
            equity=max(0.01, equity),
            buying_power=max(0.0, self._cash),
            pdt_flag=False,
            day_trades_remaining=3,
        )

    def get_positions(self) -> List[Position]:
        positions: List[Position] = []
        for symbol, pos in self._positions.items():
            qty = float(pos.get("qty") or 0.0)
            if qty == 0.0:
                continue
            last_price = self.get_last_price(symbol)
            avg_price = pos.get("avg_price")
            market_value = qty * last_price
            unrealized = None
            if avg_price is not None:
                unrealized = (last_price - avg_price) * qty
            positions.append(
                Position(
                    symbol=symbol,
                    qty=qty,
                    avg_price=avg_price,
                    market_value=market_value,
                    unrealized_pl=unrealized,
                )
            )
        self._logger.info("Simulator get_positions count=%d", len(positions))
        return positions

    def get_last_price(self, symbol: str) -> float:
        symbol = str(symbol)
        if symbol not in self._last_prices:
            self._last_prices[symbol] = self._initial_price(symbol)
        last_price = self._last_prices[symbol]
        change_pct = self._rng.gauss(0.0, self._price_volatility)
        new_price = max(self._price_floor, last_price * (1.0 + change_pct))
        self._last_prices[symbol] = new_price
        self._logger.debug("Simulator price %s %.4f->%.4f", symbol, last_price, new_price)
        return new_price

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
        if qty <= 0:
            raise ValueError("Quantity must be positive.")

        symbol = str(symbol)
        order_id = f"sim-{uuid.uuid4().hex}"
        submitted_at = datetime.utcnow()
        order = {
            "order_id": order_id,
            "symbol": symbol,
            "side": normalized_side,
            "qty": float(qty),
            "order_type": normalized_type,
            "limit_price": float(limit_price) if limit_price is not None else None,
            "status": "new",
            "submitted_at": submitted_at,
            "filled_qty": None,
            "filled_avg_price": None,
            "updated_at": submitted_at,
        }
        self._orders[order_id] = order

        self._logger.info(
            "Simulator submit_order id=%s symbol=%s side=%s qty=%.4f type=%s limit=%s",
            order_id,
            symbol,
            normalized_side,
            qty,
            normalized_type,
            limit_price,
        )

        self._maybe_fill_order(order)

        return OrderResult(
            order_id=order_id,
            status=str(order["status"]),
            submitted_at=submitted_at,
            filled_qty=order["filled_qty"],
            filled_avg_price=order["filled_avg_price"],
        )

    def cancel_order(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if not order:
            self._logger.info("Simulator cancel_order id=%s not_found", order_id)
            return False
        if order["status"] != "open":
            self._logger.info("Simulator cancel_order id=%s status=%s", order_id, order["status"])
            return False
        order["status"] = "canceled"
        order["updated_at"] = datetime.utcnow()
        self._logger.info("Simulator cancel_order id=%s canceled", order_id)
        return True

    def get_order_status(self, order_id: str) -> OrderStatus:
        order = self._orders.get(order_id)
        if not order:
            raise ValueError(f"Order not found: {order_id}")
        if order["status"] == "open":
            self._maybe_fill_order(order)
        filled_qty = order["filled_qty"]
        filled_avg_price = order["filled_avg_price"]
        remaining_qty = None
        if filled_qty is not None:
            remaining_qty = max(0.0, float(order["qty"]) - float(filled_qty))
        return OrderStatus(
            order_id=order_id,
            status=str(order["status"]),
            filled_qty=filled_qty,
            filled_avg_price=filled_avg_price,
            remaining_qty=remaining_qty,
            updated_at=order["updated_at"],
        )

    def is_market_open(self) -> bool:
        return True

    def _initial_price(self, symbol: str) -> float:
        base = 100.0 + (self._rng.random() - 0.5) * 10.0
        self._logger.debug("Simulator init price %s %.4f", symbol, base)
        return max(self._price_floor, base)

    def _position_market_value(self, symbol: str) -> float:
        pos = self._positions.get(symbol)
        if not pos:
            return 0.0
        qty = float(pos.get("qty") or 0.0)
        if qty == 0.0:
            return 0.0
        return qty * self.get_last_price(symbol)

    def _maybe_fill_order(self, order: Dict[str, object]) -> None:
        if order["status"] not in {"new", "open"}:
            return
        symbol = str(order["symbol"])
        side = str(order["side"])
        qty = float(order["qty"] or 0.0)
        order_type = str(order["order_type"])
        limit_price = order.get("limit_price")
        market_price = self.get_last_price(symbol)
        should_fill = order_type == "market"
        if order_type == "limit":
            if side == "buy" and market_price <= float(limit_price):
                should_fill = True
            elif side == "sell" and market_price >= float(limit_price):
                should_fill = True
            else:
                order["status"] = "open"
                order["updated_at"] = datetime.utcnow()
                return
        if should_fill:
            fill_price = self._apply_slippage(market_price, side)
            if limit_price is not None:
                if side == "buy":
                    fill_price = min(fill_price, float(limit_price))
                else:
                    fill_price = max(fill_price, float(limit_price))
            if not self._has_funds(side, qty, fill_price):
                order["status"] = "rejected"
                order["updated_at"] = datetime.utcnow()
                self._logger.warning(
                    "Simulator order rejected id=%s symbol=%s side=%s qty=%.4f price=%.4f",
                    order["order_id"],
                    symbol,
                    side,
                    qty,
                    fill_price,
                )
                return
            self._apply_fill(symbol, side, qty, fill_price, order["order_id"])
            order["status"] = "filled"
            order["filled_qty"] = qty
            order["filled_avg_price"] = fill_price
            order["updated_at"] = datetime.utcnow()

    def _apply_slippage(self, price: float, side: str) -> float:
        if side == "buy":
            return price * (1.0 + self._slippage)
        return price * (1.0 - self._slippage)

    def _has_funds(self, side: str, qty: float, price: float) -> bool:
        if side == "buy":
            return self._cash >= qty * price
        return True

    def _apply_fill(self, symbol: str, side: str, qty: float, price: float, order_id: str) -> None:
        signed_qty = qty if side == "buy" else -qty
        pos = self._positions.get(symbol, {"qty": 0.0, "avg_price": None})
        prev_qty = float(pos.get("qty") or 0.0)
        prev_avg = pos.get("avg_price")
        new_qty = prev_qty + signed_qty

        if side == "buy":
            self._cash -= qty * price
        else:
            self._cash += qty * price

        if new_qty == 0.0:
            new_avg = None
        elif prev_qty == 0.0 or (prev_qty > 0 and signed_qty > 0) or (prev_qty < 0 and signed_qty < 0):
            total_cost = (abs(prev_qty) * (prev_avg or price)) + (abs(signed_qty) * price)
            new_avg = total_cost / abs(new_qty)
        else:
            if abs(new_qty) > 0:
                new_avg = price
            else:
                new_avg = None

        pos["qty"] = new_qty
        pos["avg_price"] = new_avg
        self._positions[symbol] = pos

        trade = {
            "timestamp": datetime.utcnow(),
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": price,
        }
        self._trade_history.append(trade)
        self._logger.info(
            "Simulator fill id=%s symbol=%s side=%s qty=%.4f price=%.4f cash=%.2f",
            order_id,
            symbol,
            side,
            qty,
            price,
            self._cash,
        )
