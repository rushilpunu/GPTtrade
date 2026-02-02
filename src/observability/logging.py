"""Structured logging utilities for the trading system."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Mapping

import structlog

_CORRELATION_ID_KEY = "correlation_id"


def _coerce_level(level: str | int | None) -> int:
    if isinstance(level, int):
        return level
    if not level:
        return logging.INFO
    return logging.getLevelName(str(level).upper())


def setup_logging(config: Mapping[str, Any] | None = None) -> None:
    """Configure structlog with JSONL output and optional log rotation."""

    config = config or {}
    logging_cfg = config.get("logging", config)

    level = _coerce_level(logging_cfg.get("level"))
    to_console = bool(logging_cfg.get("to_console", True))
    to_file = bool(logging_cfg.get("to_file", False))
    log_file = logging_cfg.get("file", "logs/trading.log")
    max_bytes = int(logging_cfg.get("max_bytes", 10 * 1024 * 1024))
    backup_count = int(logging_cfg.get("backup_count", 5))

    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
        foreign_pre_chain=processors,
    )

    handlers: list[logging.Handler] = []
    if to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)

    if to_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)
    for handler in handlers:
        root_logger.addHandler(handler)

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger with the given name."""

    return structlog.get_logger(name)


def with_correlation_id(correlation_id: str):
    """Bind a correlation id to the current context for request tracing."""

    return structlog.contextvars.bound_contextvars(
        **{_CORRELATION_ID_KEY: correlation_id}
    )
