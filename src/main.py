"""Main entrypoint for the trading system."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import logging
import os
import signal
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import yaml
from dotenv import load_dotenv

from agent.llm_clients import LLMClientError, create_llm_client
from agent.policy import LLMPolicy, RulesPolicy
from data.market_data import AlpacaMarketData, MarketDataProvider, YahooFinanceMarketData
from execution.alpaca_broker import AlpacaBroker
from execution.broker_interface import BrokerInterface
from execution.simulator_broker import SimulatorBroker
from features.behavioral_features import BehavioralFeatureCalculator
from observability.notify import create_notifier
from observability.web import MonitoringServer
from risk.position_sizing import PositionSizer
from risk.risk_gate import RiskGate
from scheduler import TradingScheduler
from storage.db import TradingDatabase
from storage.models import (
    ConstraintViolation,
    DecisionAction,
    DecisionRecord,
    OrderRecord,
    OrderSide,
    PositionSnapshot,
)


DEFAULT_CONFIG_PATH = "config.yaml"
DEFAULT_DB_PATH = "data/trading.db"


class TradingSystem:
    def __init__(
        self,
        config: Mapping[str, Any],
        broker: BrokerInterface,
        market_data: MarketDataProvider,
        feature_calculator: BehavioralFeatureCalculator,
        policy: Any,
        risk_gate: RiskGate,
        position_sizer: PositionSizer,
        database: TradingDatabase,
        notifier: Any,
        dry_run: bool = False,
    ) -> None:
        self._config = dict(config)
        self._broker = broker
        self._market_data = market_data
        self._feature_calculator = feature_calculator
        self._policy = policy
        self._risk_gate = risk_gate
        self._position_sizer = position_sizer
        self._database = database
        self._notifier = notifier
        self._dry_run = dry_run

        self._symbols = [str(sym) for sym in (self._config.get("symbols") or [])]
        self._logger = logging.getLogger(__name__)

        self._ohlcv_cache: Dict[str, Any] = {}
        self._quote_cache: Dict[str, Any] = {}
        self._features_cache: Dict[str, Dict[str, Any]] = {}
        self._decisions_cache: Dict[str, Any] = {}
        self._pending_trade_notifications: list[Dict[str, Any]] = []
        self._pending_risk_notifications: list[Dict[str, Any]] = []

    def refresh_data(self) -> None:
        if not self._symbols:
            self._logger.warning("No symbols configured; skipping data refresh")
            return

        lookback_days = _coerce_int(self._config.get("lookback_days"), default=120)
        end = datetime.utcnow()
        start = end - timedelta(days=lookback_days)

        for symbol in self._symbols:
            try:
                ohlcv = self._market_data.get_ohlcv(symbol, start=start, end=end)
                self._ohlcv_cache[symbol] = ohlcv
            except Exception:
                self._logger.exception("Failed to load OHLCV for %s", symbol)

            try:
                quote = self._market_data.get_quote(symbol)
                self._quote_cache[symbol] = quote
            except Exception:
                self._logger.exception("Failed to load quote for %s", symbol)

    def compute_features(self) -> None:
        if not self._ohlcv_cache:
            self._logger.warning("No OHLCV data cached; skipping feature computation")
            return

        for symbol in self._symbols:
            ohlcv = self._ohlcv_cache.get(symbol)
            if ohlcv is None or getattr(ohlcv, "empty", False):
                self._logger.warning("No OHLCV data for %s", symbol)
                continue
            try:
                features = self._feature_calculator.compute_all_features(symbol, ohlcv)
                self._features_cache[symbol] = features
            except Exception:
                self._logger.exception("Failed to compute features for %s", symbol)

    def make_decisions(self) -> None:
        if not self._features_cache:
            self._logger.warning("No features available; skipping decisions")
            return

        policy_type = str(self._config.get("policy_type", "rules"))
        now = datetime.utcnow()

        for symbol, features in self._features_cache.items():
            try:
                context = self._build_context(symbol)
                decision = self._policy.decide(features, context)
                self._decisions_cache[symbol] = decision

                record = DecisionRecord(
                    timestamp=now,
                    symbol=symbol,
                    action=DecisionAction(decision.action.value),
                    confidence=float(decision.confidence),
                    rationale=str(decision.rationale),
                    feature_snapshot=features,
                    policy_type=policy_type,
                )
                self._database.save_decision(record)
            except Exception:
                self._logger.exception("Decision failed for %s", symbol)

    def execute_orders(self) -> None:
        if self._dry_run:
            self._logger.info("Dry-run enabled; skipping order execution")
            return

        if not self._decisions_cache:
            self._logger.warning("No decisions available; skipping order execution")
            return

        account = self._broker.get_account()
        positions = self._broker.get_positions()
        trades_today = len(self._database.get_orders_today())

        base_account_info = account.model_dump()
        base_account_info["trades_today"] = trades_today

        for symbol, decision in self._decisions_cache.items():
            action = decision.action.value
            if action == "HOLD":
                continue

            price = self._get_current_price(symbol)
            if price is None or price <= 0:
                self._logger.warning("Missing price for %s", symbol)
                continue

            delta = self._position_sizer.calculate_shares(
                symbol,
                action,
                price,
                account.equity,
                positions,
            )
            if delta == 0:
                continue

            side = "buy" if delta > 0 else "sell"
            qty = abs(delta)

            account_info = dict(base_account_info)
            account_info["expected_edge"] = float(decision.confidence)

            approved, reasons = self._risk_gate.check_order(
                symbol,
                side.upper(),
                qty,
                positions,
                account_info,
            )
            if not approved:
                self._record_risk_block(symbol, side, reasons)
                continue

            order_type = "limit" if self._config.get("use_limit_orders", False) else "market"
            limit_price = None
            if order_type == "limit":
                limit_price = self._calculate_limit_price(side, price)

            try:
                result = self._broker.submit_order(
                    symbol=symbol,
                    side=side,
                    qty=qty,
                    order_type=order_type,
                    limit_price=limit_price,
                )
            except Exception:
                self._logger.exception("Order submission failed for %s", symbol)
                continue

            self._record_order(result, symbol, side, qty, order_type, limit_price)
            self._queue_trade_notification(
                symbol=symbol,
                side=side,
                qty=qty,
                price=limit_price or price,
                confidence=decision.confidence,
                rationale=str(decision.rationale),
                correlation_id=str(result.order_id) if result.order_id else None,
            )

        self._snapshot_positions()

    def run_cycle(self) -> None:
        self._reset_notifications()
        self.refresh_data()
        self.compute_features()
        self.make_decisions()
        self.execute_orders()
        self._dispatch_notifications()
        self._check_large_moves()

    def _build_context(self, symbol: str) -> Dict[str, Any]:
        quote = self._quote_cache.get(symbol)
        return {
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "quote": quote.model_dump() if quote is not None else None,
        }

    def _get_current_price(self, symbol: str) -> Optional[float]:
        quote = self._quote_cache.get(symbol)
        if quote is not None:
            for value in (quote.last, quote.bid, quote.ask):
                if value is not None and value > 0:
                    return float(value)
        try:
            return float(self._broker.get_last_price(symbol))
        except Exception:
            return None

    def _calculate_limit_price(self, side: str, current_price: float) -> float:
        offset = _coerce_float(self._config.get("limit_order_offset"), default=0.0)
        if side == "buy":
            return current_price * (1.0 + offset)
        return current_price * (1.0 - offset)

    def _record_risk_block(self, symbol: str, side: str, reasons: Iterable[str]) -> None:
        now = datetime.utcnow()
        reason_list = [str(reason) for reason in reasons]
        if reason_list:
            self._pending_risk_notifications.append(
                {
                    "symbol": symbol,
                    "action": side.upper(),
                    "reasons": reason_list,
                    "correlation_id": self._build_correlation_id(symbol),
                }
            )

        for reason in reason_list:
            try:
                record = ConstraintViolation(
                    timestamp=now,
                    constraint_name=str(reason),
                    symbol=symbol,
                    reason=f"risk_gate_blocked_{side}",
                    action_blocked=True,
                )
                self._database.save_constraint_violation(record)
            except Exception:
                self._logger.exception("Failed to persist constraint violation for %s", symbol)

    def _record_order(
        self,
        result: Any,
        symbol: str,
        side: str,
        qty: float,
        order_type: str,
        limit_price: Optional[float],
    ) -> None:
        submitted_at = result.submitted_at or datetime.utcnow()
        record = OrderRecord(
            order_id=str(result.order_id),
            symbol=symbol,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            qty=float(qty),
            order_type=order_type.upper(),
            limit_price=limit_price,
            status=str(result.status).upper(),
            submitted_at=submitted_at,
            filled_at=None,
            filled_qty=result.filled_qty,
            filled_price=result.filled_avg_price,
            broker=_broker_name(self._broker),
        )
        self._database.save_order(record)

    def _snapshot_positions(self) -> None:
        now = datetime.utcnow()
        try:
            positions = self._broker.get_positions()
        except Exception:
            self._logger.exception("Failed to load positions for snapshot")
            return

        for pos in positions:
            try:
                market_value = pos.market_value
                if market_value is None:
                    price = self._get_current_price(pos.symbol)
                    if price is not None:
                        market_value = price * pos.qty
                unrealized = pos.unrealized_pl
                if unrealized is None and market_value is not None and pos.avg_price is not None:
                    unrealized = (market_value - (pos.avg_price * pos.qty))
                record = PositionSnapshot(
                    timestamp=now,
                    symbol=pos.symbol,
                    qty=float(pos.qty),
                    market_value=float(market_value or 0.0),
                    unrealized_pnl=float(unrealized or 0.0),
                )
                self._database.save_position_snapshot(record)
            except Exception:
                self._logger.exception("Failed to persist position snapshot for %s", pos.symbol)

    def _reset_notifications(self) -> None:
        self._pending_trade_notifications.clear()
        self._pending_risk_notifications.clear()

    def _queue_trade_notification(
        self,
        *,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        confidence: float,
        rationale: str,
        correlation_id: Optional[str],
    ) -> None:
        self._pending_trade_notifications.append(
            {
                "symbol": symbol,
                "side": side,
                "qty": float(qty),
                "price": float(price),
                "confidence": float(confidence),
                "rationale": rationale,
                "correlation_id": correlation_id or self._build_correlation_id(symbol),
            }
        )

    def _dispatch_notifications(self) -> None:
        for item in list(self._pending_trade_notifications):
            try:
                self._notifier.notify_trade(
                    symbol=item["symbol"],
                    side=item["side"],
                    qty=item["qty"],
                    price=item["price"],
                    confidence=item["confidence"],
                    rationale=item["rationale"],
                    correlation_id=item["correlation_id"],
                )
            except Exception:
                self._logger.exception("Failed to notify trade for %s", item.get("symbol"))

        for item in list(self._pending_risk_notifications):
            try:
                self._notifier.notify_risk_block(
                    symbol=item["symbol"],
                    action=item["action"],
                    reasons=item["reasons"],
                    correlation_id=item["correlation_id"],
                )
            except Exception:
                self._logger.exception(
                    "Failed to notify risk block for %s", item.get("symbol")
                )

    def _check_large_moves(self) -> None:
        for symbol in self._symbols:
            ohlcv = self._ohlcv_cache.get(symbol)
            if ohlcv is None:
                continue
            try:
                last_close, prev_close = self._extract_recent_closes(ohlcv)
                if last_close is None or prev_close is None or prev_close <= 0:
                    continue
                price_change_pct = ((last_close - prev_close) / prev_close) * 100.0
                current_price = self._get_current_price(symbol) or last_close
                self._notifier.notify_large_move(
                    symbol=symbol,
                    price_change_pct=price_change_pct,
                    current_price=float(current_price),
                )
            except Exception:
                self._logger.exception("Failed to evaluate large move for %s", symbol)

    @staticmethod
    def _extract_recent_closes(ohlcv: Any) -> tuple[Optional[float], Optional[float]]:
        if getattr(ohlcv, "empty", False):
            return None, None

        try:
            if hasattr(ohlcv, "__getitem__") and "close" in ohlcv:
                closes = ohlcv["close"]
                if hasattr(closes, "iloc") and len(closes) >= 2:
                    return float(closes.iloc[-1]), float(closes.iloc[-2])
                if isinstance(closes, (list, tuple)) and len(closes) >= 2:
                    return float(closes[-1]), float(closes[-2])
        except Exception:
            pass

        if isinstance(ohlcv, (list, tuple)) and len(ohlcv) >= 2:
            last = ohlcv[-1]
            prev = ohlcv[-2]
            try:
                last_close = float(last.get("close")) if isinstance(last, Mapping) else None
                prev_close = float(prev.get("close")) if isinstance(prev, Mapping) else None
                return last_close, prev_close
            except Exception:
                return None, None

        return None, None

    @staticmethod
    def _build_correlation_id(symbol: str) -> str:
        timestamp = datetime.utcnow().isoformat()
        return f"{symbol}-{timestamp}"


def _broker_name(broker: BrokerInterface) -> str:
    if isinstance(broker, AlpacaBroker):
        return "alpaca"
    if isinstance(broker, SimulatorBroker):
        return "simulator"
    return broker.__class__.__name__.lower()


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_config(config_path: Path) -> Dict[str, Any]:
    env_path = config_path.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
    else:
        load_dotenv(override=False)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if not isinstance(config, dict):
        raise ValueError("Config file must contain a mapping at the top level.")

    merged = dict(config)
    for key, value in os.environ.items():
        if key in merged or key.isupper():
            merged[key] = value
    return merged


def _setup_logging(config: Mapping[str, Any]) -> None:
    log_config = config.get("logging", {}) if isinstance(config, Mapping) else {}
    level_name = str(log_config.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = log_config.get("format", "%(asctime)s %(levelname)s %(name)s: %(message)s")

    handlers: list[logging.Handler] = []

    if log_config.get("to_console", True):
        handlers.append(logging.StreamHandler())

    if log_config.get("to_file", False):
        log_path = Path(str(log_config.get("file", "logs/trading.log")))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            RotatingFileHandler(
                log_path,
                maxBytes=_coerce_int(log_config.get("max_bytes"), default=10 * 1024 * 1024),
                backupCount=_coerce_int(log_config.get("backup_count"), default=5),
            )
        )

    if not handlers:
        handlers.append(logging.StreamHandler())

    formatter = logging.Formatter(fmt)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)
    for handler in handlers:
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)


def _build_broker(config: Mapping[str, Any]) -> BrokerInterface:
    trading_mode = str(config.get("trading_mode", "paper")).lower()
    if trading_mode == "live":
        _validate_live_confirmation(config)
        return AlpacaBroker(config)

    return SimulatorBroker(
        starting_equity=_coerce_float(config.get("starting_equity"), default=100000.0),
        symbols=config.get("symbols"),
        slippage=_coerce_float(config.get("simulator_slippage"), default=0.001),
        seed=config.get("simulator_seed"),
        price_volatility=_coerce_float(config.get("simulator_price_volatility"), default=0.002),
        price_floor=_coerce_float(config.get("simulator_price_floor"), default=0.01),
    )


def _validate_live_confirmation(config: Mapping[str, Any]) -> None:
    token = os.getenv("LIVE_TRADING_CONFIRMATION_TOKEN")
    expected_hash = _live_confirmation_hash(config)
    if not token or not expected_hash:
        raise PermissionError("Live trading confirmation token missing.")
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    if not hmac.compare_digest(token_hash, expected_hash):
        raise PermissionError("Live trading confirmation token mismatch.")


def _live_confirmation_hash(config: Mapping[str, Any]) -> Optional[str]:
    for key in (
        "LIVE_TRADING_CONFIRMATION_HASH",
        "live_trading_confirmation_hash",
        "LIVE_TRADING_CONFIRMATION_TOKEN_HASH",
        "live_trading_confirmation_token_hash",
    ):
        value = config.get(key)
        if value:
            return str(value)
        env_value = os.getenv(key)
        if env_value:
            return str(env_value)
    return None


def _build_market_data(config: Mapping[str, Any]) -> MarketDataProvider:
    provider = str(config.get("market_data_provider", "")).lower()
    if provider in {"alpaca", "alpaca_market_data", "alpaca_market"}:
        return AlpacaMarketData(config)
    if provider in {"yahoo", "yahoo_finance", "yfinance"}:
        return YahooFinanceMarketData(config)

    trading_mode = str(config.get("trading_mode", "paper")).lower()
    if trading_mode == "live":
        return AlpacaMarketData(config)

    try:
        return YahooFinanceMarketData(config)
    except Exception:
        return AlpacaMarketData(config)


def _build_policy(config: Mapping[str, Any]) -> Any:
    policy_type = str(config.get("policy_type", "rules")).lower()
    if policy_type == "llm":
        provider = str(config.get("llm_provider", "openai"))
        model = str(config.get("llm_model", "gpt-4o-mini"))
        temperature = _coerce_float(config.get("llm_temperature"), default=0.1)
        max_tokens = _coerce_int(config.get("llm_max_tokens"), default=1024)

        provider_key = provider.strip().lower()
        env_key = "GEMINI_API_KEY" if provider_key in {"gemini", "google", "gcp"} else "OPENAI_API_KEY"
        api_key = os.getenv(env_key)

        if not api_key:
            logging.getLogger(__name__).warning(
                "LLM policy selected but %s is missing; falling back to rules policy",
                env_key,
            )
            return RulesPolicy()

        try:
            llm_client = create_llm_client(
                provider=provider,
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except LLMClientError:
            logging.getLogger(__name__).exception(
                "Failed to initialize LLM client; falling back to rules policy"
            )
            return RulesPolicy()

        return LLMPolicy(llm_client=llm_client, fallback_policy=RulesPolicy())

    return RulesPolicy()


def _build_database(config: Mapping[str, Any]) -> TradingDatabase:
    db_path = Path(str(config.get("db_path", DEFAULT_DB_PATH)))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return TradingDatabase(str(db_path))


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPTTrade main entrypoint")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to config YAML file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without submitting orders.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single trading cycle and exit.",
    )
    return parser.parse_args(argv)


def _resolve_log_file(config: Mapping[str, Any]) -> Optional[str]:
    log_config = config.get("logging", {}) if isinstance(config, Mapping) else {}
    if log_config.get("to_file", False):
        return str(log_config.get("file", "logs/trading.log"))
    return None


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = _parse_args(argv)
    config_path = Path(args.config)
    config = _load_config(config_path)
    _setup_logging(config)

    broker = _build_broker(config)
    market_data = _build_market_data(config)
    feature_calculator = BehavioralFeatureCalculator()
    policy = _build_policy(config)
    risk_gate = RiskGate(config, broker)
    position_sizer = PositionSizer(config)
    database = _build_database(config)
    notifier = create_notifier(config)

    system = TradingSystem(
        config,
        broker,
        market_data,
        feature_calculator,
        policy,
        risk_gate,
        position_sizer,
        database,
        notifier,
        dry_run=args.dry_run,
    )

    if args.once:
        system.run_cycle()
        return

    intervals = config.get("scheduler_intervals") or config.get("intervals")
    default_interval = _coerce_int(config.get("decision_interval_minutes"), default=60)

    scheduler = TradingScheduler(
        data_refresh=system.refresh_data,
        compute_features=system.compute_features,
        make_decisions=system.make_decisions,
        execute_orders=system.execute_orders,
        intervals=intervals,
        default_interval_minutes=default_interval,
        dry_run=False,
        install_signal_handlers=False,
    )

    enable_web = bool(config.get("enable_web", False))
    monitoring_server: Optional[MonitoringServer] = None
    if enable_web:
        log_file = _resolve_log_file(config)
        monitoring_server = MonitoringServer(database, broker, dict(config), log_file)
        monitoring_server.start(
            host=str(config.get("web_host", "127.0.0.1")),
            port=_coerce_int(config.get("web_port"), default=8000),
        )

    prior_handlers: Dict[int, object] = {}
    stop_requested = False
    logger = logging.getLogger(__name__)

    def _handle_signal(signum: int, _frame: object) -> None:
        nonlocal stop_requested
        if stop_requested:
            return
        stop_requested = True
        logger.info("Received signal %s, shutting down", signum)
        scheduler.stop()
        if monitoring_server:
            monitoring_server.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            prior_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, _handle_signal)
        except ValueError:
            logger.warning("Signal handlers must be set in main thread")
            break

    try:
        scheduler.start()
    finally:
        if monitoring_server:
            monitoring_server.stop()
        for sig, handler in prior_handlers.items():
            try:
                signal.signal(sig, handler)
            except ValueError:
                logger.warning("Cannot restore signal handlers from non-main thread")
                break


if __name__ == "__main__":
    main()
