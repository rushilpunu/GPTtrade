"""FastAPI monitoring server for the trading system."""

from __future__ import annotations

import json
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from execution.broker_interface import BrokerInterface, Position
from storage.db import TradingDatabase
from storage.models import DecisionRecord, OrderRecord


APP_VERSION = "0.1.0"


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    version: str = Field(..., examples=[APP_VERSION])
    timestamp: datetime = Field(..., description="UTC timestamp")


class StatusResponse(BaseModel):
    mode: str
    last_run: Optional[datetime]
    next_run: Optional[datetime]
    market_open: bool
    positions_summary: Dict[str, Any]
    pnl_summary: Dict[str, Any]


class DecisionResponse(DecisionRecord):
    pass


class OrderResponse(OrderRecord):
    pass


class PositionResponse(Position):
    pass


class LogsResponse(BaseModel):
    lines: List[Dict[str, Any]]


class MonitoringServer:
    def __init__(
        self,
        database: TradingDatabase,
        broker: BrokerInterface,
        config: Dict[str, Any],
        log_file: str | None = None,
    ) -> None:
        self._database = database
        self._broker = broker
        self._config = dict(config)
        self._log_file = log_file
        self._app = _create_app(database, broker, self._config, log_file)
        self._thread: Optional[threading.Thread] = None
        self._server: Optional[uvicorn.Server] = None

    @property
    def app(self) -> FastAPI:
        return self._app

    def start(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        if self._thread and self._thread.is_alive():
            return

        config = uvicorn.Config(
            self._app,
            host=host,
            port=port,
            log_level="info",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        if self._server is None:
            return
        self._server.run()

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.should_exit = True
        if self._thread:
            self._thread.join(timeout=5)
        self._thread = None
        self._server = None


def _create_app(
    database: TradingDatabase,
    broker: BrokerInterface,
    config: Dict[str, Any],
    log_file: str | None,
) -> FastAPI:
    app = FastAPI(title="Trading Monitoring API", version=APP_VERSION)

    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^https?://(localhost|127\\.0\\.0\\.1)(:\\d+)?$",
        allow_credentials=True,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=APP_VERSION,
            timestamp=datetime.now(timezone.utc),
        )

    @app.get("/status", response_model=StatusResponse)
    def status() -> StatusResponse:
        try:
            market_open = broker.is_market_open()
        except Exception as exc:  # pragma: no cover - depends on broker
            raise HTTPException(status_code=503, detail="Broker unavailable") from exc

        positions = _safe_positions(broker)
        positions_summary = _positions_summary(positions)
        pnl_summary = _pnl_summary(database, positions_summary)

        return StatusResponse(
            mode=str(config.get("trading_mode", "paper")),
            last_run=_coerce_datetime(config.get("last_run")),
            next_run=_coerce_datetime(config.get("next_run")),
            market_open=market_open,
            positions_summary=positions_summary,
            pnl_summary=pnl_summary,
        )

    @app.get("/decisions", response_model=List[DecisionResponse])
    def decisions(limit: int = Query(50, ge=1, le=500)) -> List[DecisionResponse]:
        try:
            rows = database._db["decisions"].rows_where(
                "1=1", order_by="timestamp desc", limit=limit
            )
            return [DecisionResponse.model_validate(dict(row)) for row in rows]
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Failed to load decisions") from exc

    @app.get("/orders", response_model=List[OrderResponse])
    def orders(limit: int = Query(50, ge=1, le=500)) -> List[OrderResponse]:
        try:
            rows = database._db["orders"].rows_where(
                "1=1", order_by="submitted_at desc", limit=limit
            )
            return [OrderResponse.model_validate(dict(row)) for row in rows]
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Failed to load orders") from exc

    @app.get("/positions", response_model=List[PositionResponse])
    def positions() -> List[PositionResponse]:
        try:
            items = broker.get_positions()
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Broker unavailable") from exc
        return [PositionResponse.model_validate(item.model_dump()) for item in items]

    @app.get("/logs/recent", response_model=LogsResponse)
    def logs_recent(lines: int = Query(200, ge=1, le=2000)) -> LogsResponse:
        if not log_file:
            raise HTTPException(status_code=404, detail="Log file not configured")

        path = Path(log_file)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Log file not found")

        try:
            entries = _tail_json_lines(path, lines)
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Failed to read logs") from exc
        return LogsResponse(lines=entries)

    return app


def _coerce_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _safe_positions(broker: BrokerInterface) -> List[Position]:
    try:
        return broker.get_positions()
    except Exception:
        return []


def _positions_summary(positions: Iterable[Position]) -> Dict[str, Any]:
    total_market_value = 0.0
    total_unrealized = 0.0
    long_count = 0
    short_count = 0
    count = 0

    for position in positions:
        count += 1
        if position.qty > 0:
            long_count += 1
        elif position.qty < 0:
            short_count += 1
        if position.market_value is not None:
            total_market_value += float(position.market_value)
        if position.unrealized_pl is not None:
            total_unrealized += float(position.unrealized_pl)

    return {
        "count": count,
        "long_count": long_count,
        "short_count": short_count,
        "market_value": total_market_value,
        "unrealized_pnl": total_unrealized,
    }


def _pnl_summary(database: TradingDatabase, positions_summary: Dict[str, Any]) -> Dict[str, Any]:
    summary = {
        "date": None,
        "realized_pnl": None,
        "unrealized_pnl": positions_summary.get("unrealized_pnl", 0.0),
        "kill_switch_triggered": None,
    }

    try:
        table = database._db["daily_pnl"]
    except Exception:
        return summary

    row = _latest_daily_pnl_row(table, order_by="date desc")
    if row is None:
        row = _latest_daily_pnl_row(table, order_by="trading_date desc")

    if row:
        date_value = row.get("date") or row.get("trading_date")
        summary.update(
            {
                "date": date_value,
                "realized_pnl": row.get("realized_pnl"),
                "unrealized_pnl": row.get("unrealized_pnl", summary["unrealized_pnl"]),
                "kill_switch_triggered": row.get("kill_switch_triggered"),
            }
        )

    return summary


def _latest_daily_pnl_row(table: Any, order_by: str) -> Optional[Dict[str, Any]]:
    try:
        rows = table.rows_where("1=1", order_by=order_by, limit=1)
        for row in rows:
            return dict(row)
    except Exception:
        return None
    return None


def _tail_json_lines(path: Path, max_lines: int) -> List[Dict[str, Any]]:
    buffer: deque[str] = deque(maxlen=max_lines)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                buffer.append(line)

    entries: List[Dict[str, Any]] = []
    for line in buffer:
        try:
            payload = json.loads(line)
            if isinstance(payload, dict):
                entries.append(payload)
            else:
                entries.append({"message": payload})
        except json.JSONDecodeError:
            entries.append({"message": line})
    return entries
