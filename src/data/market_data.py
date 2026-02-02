"""Market data providers with caching and fallback support."""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, Field

try:
    from alpaca_trade_api import REST
    from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit
except Exception:  # pragma: no cover - optional import path
    REST = None
    APIError = None
    TimeFrame = None
    TimeFrameUnit = None


class Quote(BaseModel):
    """Best-effort quote snapshot."""

    bid: Optional[float] = Field(None, description="Best bid price.")
    ask: Optional[float] = Field(None, description="Best ask price.")
    last: Optional[float] = Field(None, description="Last traded price.")
    volume: Optional[float] = Field(None, description="Most recent trade volume.")
    timestamp: datetime = Field(..., description="Timestamp for the quote.")


class MarketDataProvider(ABC):
    """Abstract base class for market data sources."""

    @abstractmethod
    def get_ohlcv(self, symbol: str, start: Any, end: Any) -> pd.DataFrame:
        """Return OHLCV dataframe indexed by timestamp."""

    @abstractmethod
    def get_quote(self, symbol: str) -> Quote:
        """Return the latest quote snapshot."""


class _TTLCache:
    def __init__(self, ttl_seconds: float, max_items: int = 256) -> None:
        self._ttl_seconds = float(ttl_seconds)
        self._max_items = int(max_items)
        self._store: Dict[Any, Tuple[float, Any]] = {}

    def get(self, key: Any) -> Optional[Any]:
        entry = self._store.get(key)
        if not entry:
            return None
        expires_at, value = entry
        if time.time() >= expires_at:
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: Any, value: Any, ttl_seconds: Optional[float] = None) -> None:
        if len(self._store) >= self._max_items:
            self._purge_expired()
        ttl = self._ttl_seconds if ttl_seconds is None else float(ttl_seconds)
        self._store[key] = (time.time() + ttl, value)

    def _purge_expired(self) -> None:
        now = time.time()
        stale = [key for key, (expires_at, _) in self._store.items() if now >= expires_at]
        for key in stale:
            self._store.pop(key, None)


class YahooFinanceMarketData(MarketDataProvider):
    """Fallback data provider using Yahoo Finance (yfinance)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._config = config or {}
        self._logger = logging.getLogger(__name__)
        self._quote_cache = _TTLCache(self._config.get("quote_cache_ttl_seconds", 15))
        self._ohlcv_cache = _TTLCache(self._config.get("ohlcv_cache_ttl_seconds", 60))
        try:
            import yfinance as yf
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("yfinance is required for YahooFinanceMarketData.") from exc
        self._yf = yf

        self._max_retries = int(self._config.get("yfinance_max_retries", 3))
        self._backoff_base = float(self._config.get("yfinance_backoff_base_seconds", 1.0))
        self._backoff_max = float(self._config.get("yfinance_backoff_max_seconds", 10.0))

    def get_ohlcv(self, symbol: str, start: Any, end: Any) -> pd.DataFrame:
        cache_key = (symbol, str(start), str(end), "yfinance")
        cached = self._ohlcv_cache.get(cache_key)
        if cached is not None:
            return cached.copy()

        def _download() -> pd.DataFrame:
            return self._yf.download(
                symbol,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                threads=False,
            )

        df = self._call_with_backoff(_download, "yfinance.download")
        if df is None or df.empty:
            raise ValueError(f"No OHLCV data returned for {symbol}.")

        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )
        df = df[[col for col in ["open", "high", "low", "close", "volume", "adj_close"] if col in df]]
        df = df.sort_index()
        self._ohlcv_cache.set(cache_key, df)
        return df.copy()

    def get_quote(self, symbol: str) -> Quote:
        cache_key = (symbol, "yfinance")
        cached = self._quote_cache.get(cache_key)
        if cached is not None:
            return cached

        def _fetch() -> Quote:
            ticker = self._yf.Ticker(symbol)
            info = getattr(ticker, "fast_info", None) or {}
            bid = info.get("bid")
            ask = info.get("ask")
            last = info.get("last_price") or info.get("last")
            volume = info.get("last_volume") or info.get("volume")
            timestamp = info.get("last_time")
            if timestamp is None:
                info_full = getattr(ticker, "info", {}) or {}
                bid = bid or info_full.get("bid")
                ask = ask or info_full.get("ask")
                last = last or info_full.get("regularMarketPrice")
                volume = volume or info_full.get("regularMarketVolume")
                timestamp = info_full.get("regularMarketTime")
            ts = self._coerce_datetime(timestamp) or datetime.utcnow()
            return Quote(
                bid=self._coerce_float(bid),
                ask=self._coerce_float(ask),
                last=self._coerce_float(last),
                volume=self._coerce_float(volume),
                timestamp=ts,
            )

        quote = self._call_with_backoff(_fetch, "yfinance.quote")
        self._quote_cache.set(cache_key, quote)
        return quote

    def _call_with_backoff(self, fn: Any, name: str) -> Any:
        attempt = 0
        backoff = self._backoff_base
        while True:
            try:
                return fn()
            except Exception as exc:
                if attempt < self._max_retries and self._is_rate_limit_error(exc):
                    self._logger.warning(
                        "Yahoo Finance rate limit on %s; retrying in %.2fs (attempt %s/%s)",
                        name,
                        backoff,
                        attempt + 1,
                        self._max_retries,
                    )
                    time.sleep(backoff)
                    attempt += 1
                    backoff = min(backoff * 2.0, self._backoff_max)
                    continue
                self._logger.exception("Yahoo Finance call failed %s", name)
                raise

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return "rate limit" in message or "too many requests" in message or "429" in message

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
            if isinstance(value, (int, float)):
                return datetime.utcfromtimestamp(value)
            return datetime.fromisoformat(str(value))
        except (TypeError, ValueError):
            return None


class AlpacaMarketData(MarketDataProvider):
    """Market data provider for Alpaca with optional fallback."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, fallback_provider: Optional[MarketDataProvider] = None) -> None:
        self._config = config or {}
        self._logger = logging.getLogger(__name__)

        self._max_retries = int(self._config.get("alpaca_max_retries", 4))
        self._backoff_base = float(self._config.get("alpaca_backoff_base_seconds", 1.0))
        self._backoff_max = float(self._config.get("alpaca_backoff_max_seconds", 20.0))

        self._quote_cache = _TTLCache(self._config.get("quote_cache_ttl_seconds", 10))
        self._ohlcv_cache = _TTLCache(self._config.get("ohlcv_cache_ttl_seconds", 60))

        self._api = self._init_api()
        self._fallback = fallback_provider
        if self._fallback is None:
            try:
                self._fallback = YahooFinanceMarketData(self._config)
            except Exception:
                self._fallback = None
                self._logger.info("Yahoo Finance fallback not available (missing dependency).")

    def get_ohlcv(self, symbol: str, start: Any, end: Any) -> pd.DataFrame:
        timeframe = self._resolve_timeframe(self._config.get("alpaca_timeframe", "1Min"))
        cache_key = (symbol, str(start), str(end), str(timeframe))
        cached = self._ohlcv_cache.get(cache_key)
        if cached is not None:
            return cached.copy()

        if self._api is None:
            return self._fallback_ohlcv(symbol, start, end)

        try:
            df = self._call_api(
                self._api.get_bars,
                symbol,
                timeframe,
                start=start,
                end=end,
                adjustment=self._config.get("alpaca_adjustment", "raw"),
            )
            bars = getattr(df, "df", df)
            normalized = self._normalize_ohlcv(bars, symbol)
            if normalized.empty:
                raise ValueError(f"Alpaca returned empty OHLCV for {symbol}.")
            self._ohlcv_cache.set(cache_key, normalized)
            return normalized.copy()
        except Exception as exc:
            self._logger.warning("Alpaca OHLCV failed for %s: %s", symbol, exc)
            return self._fallback_ohlcv(symbol, start, end)

    def get_quote(self, symbol: str) -> Quote:
        cache_key = (symbol, "alpaca")
        cached = self._quote_cache.get(cache_key)
        if cached is not None:
            return cached

        if self._api is None:
            return self._fallback_quote(symbol)

        try:
            quote = self._call_api(self._api.get_latest_quote, symbol)
            trade = self._call_api(self._api.get_latest_trade, symbol)
            bid = getattr(quote, "bid_price", None) or getattr(quote, "bp", None)
            ask = getattr(quote, "ask_price", None) or getattr(quote, "ap", None)
            quote_ts = getattr(quote, "timestamp", None) or getattr(quote, "t", None)
            last = getattr(trade, "price", None) or getattr(trade, "p", None)
            volume = getattr(trade, "size", None) or getattr(trade, "s", None)
            trade_ts = getattr(trade, "timestamp", None) or getattr(trade, "t", None)
            ts = self._coerce_datetime(trade_ts) or self._coerce_datetime(quote_ts) or datetime.utcnow()
            result = Quote(
                bid=self._coerce_float(bid),
                ask=self._coerce_float(ask),
                last=self._coerce_float(last),
                volume=self._coerce_float(volume),
                timestamp=ts,
            )
            self._quote_cache.set(cache_key, result)
            return result
        except Exception as exc:
            self._logger.warning("Alpaca quote failed for %s: %s", symbol, exc)
            return self._fallback_quote(symbol)

    def _fallback_quote(self, symbol: str) -> Quote:
        if self._fallback is None:
            raise RuntimeError("Alpaca is unavailable and no fallback provider is configured.")
        return self._fallback.get_quote(symbol)

    def _fallback_ohlcv(self, symbol: str, start: Any, end: Any) -> pd.DataFrame:
        if self._fallback is None:
            raise RuntimeError("Alpaca is unavailable and no fallback provider is configured.")
        return self._fallback.get_ohlcv(symbol, start, end)

    def _init_api(self) -> Optional[Any]:
        if REST is None:
            self._logger.warning("alpaca-trade-api is not installed; AlpacaMarketData disabled.")
            return None

        key_id = os.getenv("ALPACA_API_KEY_ID") or os.getenv("APCA_API_KEY_ID")
        secret_key = os.getenv("ALPACA_API_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
        if not key_id or not secret_key:
            self._logger.warning("Missing Alpaca API credentials; AlpacaMarketData disabled.")
            return None

        base_url = self._config.get("ALPACA_DATA_URL") or self._config.get("alpaca_data_url")
        return REST(key_id, secret_key, base_url=base_url)

    def _call_api(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        name = getattr(fn, "__name__", "api_call")
        attempt = 0
        backoff = self._backoff_base
        while True:
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                if attempt < self._max_retries and self._is_rate_limit_error(exc):
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

    def _resolve_timeframe(self, timeframe: Any) -> Any:
        if TimeFrame is None:
            return timeframe
        if isinstance(timeframe, TimeFrame):
            return timeframe
        tf = str(timeframe).strip()
        lower = tf.lower()
        if lower in {"1min", "1m", "minute"}:
            return TimeFrame.Minute
        if lower in {"1hour", "1h", "hour"}:
            return TimeFrame.Hour
        if lower in {"1day", "1d", "day"}:
            return TimeFrame.Day
        for suffix, unit in (("min", TimeFrameUnit.Minute), ("m", TimeFrameUnit.Minute)):
            if lower.endswith(suffix) and lower[:-len(suffix)].isdigit():
                return TimeFrame(int(lower[:-len(suffix)]), unit)
        if lower.endswith("h") and lower[:-1].isdigit():
            return TimeFrame(int(lower[:-1]), TimeFrameUnit.Hour)
        if lower.endswith("d") and lower[:-1].isdigit():
            return TimeFrame(int(lower[:-1]), TimeFrameUnit.Day)
        return timeframe

    def _normalize_ohlcv(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.index, pd.MultiIndex):
            if symbol in df.index.get_level_values(0):
                df = df.xs(symbol)
            else:
                df = df.droplevel(0)
        df = df.rename(
            columns={
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
                "v": "volume",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
            }
        )
        df = df[[col for col in ["open", "high", "low", "close", "volume"] if col in df]]
        return df.sort_index()

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
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
            if isinstance(value, (int, float)):
                return datetime.utcfromtimestamp(value)
            return datetime.fromisoformat(str(value))
        except (TypeError, ValueError):
            return None


__all__ = ["MarketDataProvider", "Quote", "AlpacaMarketData", "YahooFinanceMarketData"]
