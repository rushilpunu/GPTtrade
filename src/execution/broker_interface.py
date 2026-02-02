"""Abstract broker interface and shared execution models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class AccountInfo(BaseModel):
    equity: float = Field(..., gt=0.0, description="Total account equity.")
    buying_power: float = Field(..., ge=0.0, description="Available buying power.")
    pdt_flag: bool = Field(..., description="Whether the account is flagged as PDT.")
    day_trades_remaining: int = Field(
        ...,
        ge=0,
        description="Remaining day trades before PDT violation.",
    )


class Position(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=32, description="Symbol.")
    qty: float = Field(..., description="Position quantity (signed for long/short).")
    avg_price: Optional[float] = Field(
        None,
        gt=0.0,
        description="Average entry price.",
    )
    market_value: Optional[float] = Field(
        None,
        description="Current market value.",
    )
    unrealized_pl: Optional[float] = Field(
        None,
        description="Unrealized profit/loss.",
    )


class OrderResult(BaseModel):
    order_id: str = Field(..., min_length=1, max_length=128, description="Order id.")
    status: str = Field(..., min_length=1, max_length=32, description="Order status.")
    submitted_at: Optional[datetime] = Field(
        None,
        description="Timestamp when the order was submitted.",
    )
    filled_qty: Optional[float] = Field(
        None,
        ge=0.0,
        description="Filled quantity.",
    )
    filled_avg_price: Optional[float] = Field(
        None,
        gt=0.0,
        description="Average fill price.",
    )


class OrderStatus(BaseModel):
    order_id: str = Field(..., min_length=1, max_length=128, description="Order id.")
    status: str = Field(..., min_length=1, max_length=32, description="Order status.")
    filled_qty: Optional[float] = Field(
        None,
        ge=0.0,
        description="Filled quantity.",
    )
    filled_avg_price: Optional[float] = Field(
        None,
        gt=0.0,
        description="Average fill price.",
    )
    remaining_qty: Optional[float] = Field(
        None,
        ge=0.0,
        description="Remaining quantity.",
    )
    updated_at: Optional[datetime] = Field(
        None,
        description="Last status update timestamp.",
    )


class BrokerInterface(ABC):
    @abstractmethod
    def get_account(self) -> AccountInfo:
        raise NotImplementedError

    @abstractmethod
    def get_positions(self) -> List[Position]:
        raise NotImplementedError

    @abstractmethod
    def get_last_price(self, symbol: str) -> float:
        raise NotImplementedError

    @abstractmethod
    def submit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str,
        limit_price: Optional[float] = None,
    ) -> OrderResult:
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderStatus:
        raise NotImplementedError

    @abstractmethod
    def is_market_open(self) -> bool:
        raise NotImplementedError
