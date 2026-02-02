"""SQLite-backed storage for trading records."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any, Dict, Iterable, Optional, Type, TypeVar

from pydantic import BaseModel
from sqlite_utils import Database

from .models import (
    ConstraintViolation,
    DailyPnLRecord,
    DecisionRecord,
    OrderRecord,
    PositionSnapshot,
)

ModelT = TypeVar("ModelT", bound=BaseModel)


class TradingDatabase:
    """SQLite storage layer backed by sqlite-utils."""

    def __init__(self, db_path: str) -> None:
        self._db = Database(db_path)
        self._ensure_tables()

    def save_decision(
        self, decision: DecisionRecord, correlation_id: Optional[str] = None
    ) -> None:
        row = self._model_to_row(decision, correlation_id)
        self._db["decisions"].insert(row)

    def save_order(self, order: OrderRecord, correlation_id: Optional[str] = None) -> None:
        row = self._model_to_row(order, correlation_id)
        self._db["orders"].upsert(row, pk="order_id")

    def save_position_snapshot(
        self, snapshot: PositionSnapshot, correlation_id: Optional[str] = None
    ) -> None:
        row = self._model_to_row(snapshot, correlation_id)
        self._db["positions"].insert(row)

    def save_daily_pnl(
        self, record: DailyPnLRecord, correlation_id: Optional[str] = None
    ) -> None:
        row = self._model_to_row(record, correlation_id)
        self._db["daily_pnl"].upsert(row, pk="date")

    def save_constraint_violation(
        self, violation: ConstraintViolation, correlation_id: Optional[str] = None
    ) -> None:
        row = self._model_to_row(violation, correlation_id)
        self._db["constraint_violations"].insert(row)

    def get_decisions_by_symbol(
        self, symbol: str, limit: Optional[int] = None
    ) -> list[DecisionRecord]:
        rows = self._db["decisions"].rows_where(
            "symbol = ?",
            [symbol],
            order_by="timestamp desc",
            limit=limit,
        )
        return [self._row_to_model(DecisionRecord, row) for row in rows]

    def get_orders_today(self) -> list[OrderRecord]:
        today = date.today()
        start, end = self._day_bounds(today)
        rows = self._db["orders"].rows_where(
            "submitted_at >= ? AND submitted_at < ?",
            [start, end],
            order_by="submitted_at desc",
        )
        return [self._row_to_model(OrderRecord, row) for row in rows]

    def get_daily_pnl(self, day: date) -> Optional[DailyPnLRecord]:
        rows = list(self._db["daily_pnl"].rows_where("date = ?", [day.isoformat()], limit=1))
        if not rows:
            return None
        return self._row_to_model(DailyPnLRecord, rows[0])

    def get_latest_positions(self) -> list[PositionSnapshot]:
        query = """
            SELECT p.timestamp, p.symbol, p.qty, p.market_value, p.unrealized_pnl, p.correlation_id
            FROM positions p
            JOIN (
                SELECT symbol, MAX(timestamp) AS max_ts
                FROM positions
                GROUP BY symbol
            ) latest
            ON p.symbol = latest.symbol AND p.timestamp = latest.max_ts
            ORDER BY p.symbol
        """
        rows = self._db.query(query)
        return [self._row_to_model(PositionSnapshot, row) for row in rows]

    def _ensure_tables(self) -> None:
        self._ensure_table(
            "decisions",
            {
                "timestamp": "text",
                "symbol": "text",
                "action": "text",
                "confidence": "float",
                "rationale": "text",
                "feature_snapshot": "text",
                "policy_type": "text",
                "correlation_id": "text",
            },
        )
        self._ensure_table(
            "orders",
            {
                "order_id": "text",
                "symbol": "text",
                "side": "text",
                "qty": "float",
                "order_type": "text",
                "limit_price": "float",
                "status": "text",
                "submitted_at": "text",
                "filled_at": "text",
                "filled_qty": "float",
                "filled_price": "float",
                "broker": "text",
                "correlation_id": "text",
            },
            pk="order_id",
        )
        self._ensure_table(
            "positions",
            {
                "timestamp": "text",
                "symbol": "text",
                "qty": "float",
                "market_value": "float",
                "unrealized_pnl": "float",
                "correlation_id": "text",
            },
        )
        self._ensure_table(
            "daily_pnl",
            {
                "date": "text",
                "realized_pnl": "float",
                "unrealized_pnl": "float",
                "kill_switch_triggered": "integer",
                "correlation_id": "text",
            },
            pk="date",
        )
        self._ensure_table(
            "constraint_violations",
            {
                "timestamp": "text",
                "constraint_name": "text",
                "symbol": "text",
                "reason": "text",
                "action_blocked": "integer",
                "correlation_id": "text",
            },
        )

    def _ensure_table(
        self, name: str, columns: Dict[str, str], pk: Optional[str] = None
    ) -> None:
        col_defs = []
        for col_name, col_type in columns.items():
            if pk and col_name == pk:
                col_defs.append(f"{col_name} {col_type.upper()} PRIMARY KEY")
            else:
                col_defs.append(f"{col_name} {col_type.upper()}")
        create_sql = f"CREATE TABLE IF NOT EXISTS {name} ({', '.join(col_defs)})"
        self._db.execute(create_sql)

        table = self._db.table(name)
        existing = {col.name for col in table.columns}
        for column, col_type in columns.items():
            if column not in existing:
                table.add_column(column, col_type)

    @staticmethod
    def _day_bounds(day: date) -> tuple[str, str]:
        start = datetime.combine(day, time.min).isoformat()
        end = datetime.combine(day + timedelta(days=1), time.min).isoformat()
        return start, end

    @staticmethod
    def _model_to_row(model: BaseModel, correlation_id: Optional[str]) -> Dict[str, Any]:
        data = model.model_dump()
        for key, value in data.items():
            data[key] = TradingDatabase._serialize_value(value)
        data["correlation_id"] = correlation_id
        return data

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, date):
            return value.isoformat()
        return value

    @staticmethod
    def _row_to_model(model_cls: Type[ModelT], row: Iterable[tuple[str, Any]] | Dict[str, Any]) -> ModelT:
        row_dict = dict(row)
        filtered = {field: row_dict.get(field) for field in model_cls.model_fields}
        return model_cls.model_validate(filtered)
