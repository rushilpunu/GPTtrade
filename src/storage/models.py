"""Pydantic models for storage records."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class DecisionAction(str, Enum):
    STRONG_SELL = "STRONG_SELL"
    SELL = "SELL"
    HOLD = "HOLD"
    BUY = "BUY"
    STRONG_BUY = "STRONG_BUY"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class DecisionRecord(BaseModel):
    timestamp: datetime = Field(
        ...,
        description="UTC timestamp when the decision was made.",
        examples=["2026-01-30T14:05:12Z"],
    )
    symbol: str = Field(
        ...,
        min_length=1,
        max_length=32,
        description="Instrument symbol.",
        examples=["AAPL"],
    )
    action: DecisionAction = Field(
        ...,
        description="Recommended trading action.",
        examples=["BUY"],
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence from 0 to 1.",
        examples=[0.82],
    )
    rationale: str = Field(
        ...,
        min_length=1,
        description="Human-readable explanation for the decision.",
        examples=["Momentum and earnings revisions improved."],
    )
    feature_snapshot: Dict[str, Any] = Field(
        ...,
        description="Raw features at decision time.",
        examples=[{"rsi_14": 61.4, "macd": 0.13, "volume_z": 1.8}],
    )
    policy_type: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Policy or strategy identifier.",
        examples=["risk_parity_v2"],
    )


class OrderRecord(BaseModel):
    order_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Broker or internal order identifier.",
        examples=["ord_9f2e7a"],
    )
    symbol: str = Field(
        ...,
        min_length=1,
        max_length=32,
        description="Instrument symbol.",
        examples=["AAPL"],
    )
    side: OrderSide = Field(
        ...,
        description="Order side.",
        examples=["BUY"],
    )
    qty: float = Field(
        ...,
        gt=0.0,
        description="Order quantity.",
        examples=[10.0],
    )
    order_type: str = Field(
        ...,
        min_length=1,
        max_length=32,
        description="Order type (e.g., MARKET, LIMIT).",
        examples=["LIMIT"],
    )
    limit_price: Optional[float] = Field(
        None,
        gt=0.0,
        description="Limit price for limit orders.",
        examples=[192.5],
    )
    status: str = Field(
        ...,
        min_length=1,
        max_length=32,
        description="Broker order status.",
        examples=["FILLED"],
    )
    submitted_at: datetime = Field(
        ...,
        description="Timestamp when the order was submitted.",
        examples=["2026-01-30T14:06:02Z"],
    )
    filled_at: Optional[datetime] = Field(
        None,
        description="Timestamp when the order was fully filled.",
        examples=["2026-01-30T14:06:05Z"],
    )
    filled_qty: Optional[float] = Field(
        None,
        ge=0.0,
        description="Filled quantity.",
        examples=[10.0],
    )
    filled_price: Optional[float] = Field(
        None,
        gt=0.0,
        description="Average filled price.",
        examples=[192.45],
    )
    broker: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Broker or venue name.",
        examples=["alpaca"],
    )


class PositionSnapshot(BaseModel):
    timestamp: datetime = Field(
        ...,
        description="Snapshot timestamp.",
        examples=["2026-01-30T14:10:00Z"],
    )
    symbol: str = Field(
        ...,
        min_length=1,
        max_length=32,
        description="Instrument symbol.",
        examples=["AAPL"],
    )
    qty: float = Field(
        ...,
        description="Position quantity (signed for long/short).",
        examples=[25.0],
    )
    market_value: float = Field(
        ...,
        description="Current market value of the position.",
        examples=[4810.5],
    )
    unrealized_pnl: float = Field(
        ...,
        description="Unrealized PnL for the position.",
        examples=[125.3],
    )


class DailyPnLRecord(BaseModel):
    trading_date: date = Field(
        ...,
        description="Trading date.",
        examples=["2026-01-30"],
    )
    realized_pnl: float = Field(
        ...,
        description="Realized PnL for the day.",
        examples=[-42.75],
    )
    unrealized_pnl: float = Field(
        ...,
        description="End-of-day unrealized PnL.",
        examples=[18.25],
    )
    kill_switch_triggered: bool = Field(
        ...,
        description="Whether the kill switch was triggered on this date.",
        examples=[False],
    )


class ConstraintViolation(BaseModel):
    timestamp: datetime = Field(
        ...,
        description="Timestamp of the violation.",
        examples=["2026-01-30T13:55:00Z"],
    )
    constraint_name: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Name of the violated constraint.",
        examples=["max_position_size"],
    )
    symbol: Optional[str] = Field(
        None,
        min_length=1,
        max_length=32,
        description="Related instrument symbol, if applicable.",
        examples=["AAPL"],
    )
    reason: str = Field(
        ...,
        min_length=1,
        description="Reason for the violation.",
        examples=["Order would exceed max position size."],
    )
    action_blocked: bool = Field(
        ...,
        description="Whether the action was blocked due to the violation.",
        examples=[True],
    )
