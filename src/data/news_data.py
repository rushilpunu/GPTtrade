"""News data providers with caching, rate limiting, and fallback support."""

from __future__ import annotations

import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests


@dataclass(frozen=True)
class NewsItem:
    title: str
    summary: Optional[str]
    source: str
    published_at: datetime
    url: Optional[str]
    symbols: List[str]
    sentiment_score: Optional[float]


class NewsProvider(ABC):
    """Abstract base class for news providers."""

    @abstractmethod
    def get_headlines(self, symbol: str, limit: int = 10) -> List[NewsItem]:
        """Return news headlines for a specific symbol."""

    @abstractmethod
    def get_market_news(self, limit: int = 20) -> List[NewsItem]:
        """Return broader market news headlines."""


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


class AlphaVantageError(RuntimeError):
    """Raised when Alpha Vantage news retrieval fails."""


class AlphaVantageNewsProvider(NewsProvider):
    """News provider using the Alpha Vantage NEWS_SENTIMENT endpoint."""

    _BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, config: Optional[Dict[str, Any]] = None, api_key: Optional[str] = None) -> None:
        self._config = config or {}
        self._logger = logging.getLogger(__name__)
        self._api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        if not self._api_key:
            raise AlphaVantageError("Missing Alpha Vantage API key.")

        self._cache = _TTLCache(self._config.get("news_cache_ttl_seconds", 300))
        self._min_interval_seconds = float(self._config.get("alpha_vantage_min_interval_seconds", 12))
        self._last_call_time: Optional[float] = None
        self._timeout_seconds = float(self._config.get("alpha_vantage_timeout_seconds", 10))
        self._market_topics = self._config.get("alpha_vantage_market_topics", "financial_markets")

    def get_headlines(self, symbol: str, limit: int = 10) -> List[NewsItem]:
        cache_key = ("headlines", symbol.upper(), int(limit))
        cached = self._cache.get(cache_key)
        if cached is not None:
            return list(cached)

        items = self._fetch_news({"tickers": symbol.upper()}, limit=limit, symbol=symbol.upper())
        self._cache.set(cache_key, items)
        return list(items)

    def get_market_news(self, limit: int = 20) -> List[NewsItem]:
        cache_key = ("market", int(limit))
        cached = self._cache.get(cache_key)
        if cached is not None:
            return list(cached)

        params: Dict[str, Any] = {}
        if self._market_topics:
            params["topics"] = self._market_topics
        items = self._fetch_news(params, limit=limit)
        self._cache.set(cache_key, items)
        return list(items)

    def _fetch_news(self, extra_params: Dict[str, Any], limit: int, symbol: Optional[str] = None) -> List[NewsItem]:
        params = {
            "function": "NEWS_SENTIMENT",
            "apikey": self._api_key,
            "sort": "LATEST",
            "limit": int(limit),
        }
        params.update(extra_params)

        self._respect_rate_limit()
        try:
            response = requests.get(self._BASE_URL, params=params, timeout=self._timeout_seconds)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            self._logger.exception("Alpha Vantage news request failed")
            raise AlphaVantageError("Failed to fetch Alpha Vantage news") from exc

        if not isinstance(payload, dict):
            raise AlphaVantageError("Alpha Vantage response was not JSON")

        if "Note" in payload or "Information" in payload:
            message = payload.get("Note") or payload.get("Information")
            self._logger.warning("Alpha Vantage rate limit or info: %s", message)
            raise AlphaVantageError(message or "Alpha Vantage rate limit hit")

        feed = payload.get("feed")
        if not isinstance(feed, list):
            raise AlphaVantageError("Alpha Vantage response missing feed data")

        items: List[NewsItem] = []
        for entry in feed[: int(limit)]:
            title = (entry.get("title") or "").strip()
            if not title:
                continue
            summary = entry.get("summary")
            source = entry.get("source") or entry.get("source_domain") or "Alpha Vantage"
            published_at = _parse_av_time(entry.get("time_published"))
            url = entry.get("url")
            symbols = _extract_symbols(entry.get("ticker_sentiment"), fallback_symbol=symbol)
            sentiment = _extract_sentiment(entry, symbol)
            items.append(
                NewsItem(
                    title=title,
                    summary=summary,
                    source=source,
                    published_at=published_at,
                    url=url,
                    symbols=symbols,
                    sentiment_score=sentiment,
                )
            )

        return items

    def _respect_rate_limit(self) -> None:
        if self._last_call_time is None:
            self._last_call_time = time.time()
            return
        elapsed = time.time() - self._last_call_time
        if elapsed < self._min_interval_seconds:
            sleep_for = self._min_interval_seconds - elapsed
            self._logger.debug("Sleeping %.2fs to respect Alpha Vantage rate limits", sleep_for)
            time.sleep(sleep_for)
        self._last_call_time = time.time()


class RSSNewsProvider(NewsProvider):
    """RSS-based news provider using feedparser."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._config = config or {}
        self._logger = logging.getLogger(__name__)
        try:
            import feedparser
        except Exception as exc:
            raise ImportError("feedparser is required for RSSNewsProvider") from exc
        self._feedparser = feedparser

        self._feeds = self._config.get(
            "rss_feeds",
            [
                "https://finance.yahoo.com/rss/topstories",
                "https://feeds.reuters.com/reuters/marketsNews",
                "https://feeds.reuters.com/reuters/businessNews",
            ],
        )
        self._symbol_feed_template = self._config.get(
            "rss_symbol_feed_template",
            "https://finance.yahoo.com/rss/headline?s={symbol}",
        )

    def get_headlines(self, symbol: str, limit: int = 10) -> List[NewsItem]:
        feed_url = self._symbol_feed_template.format(symbol=symbol.upper())
        entries = self._read_feed(feed_url)
        return self._entries_to_items(entries, limit=limit, symbol=symbol.upper())

    def get_market_news(self, limit: int = 20) -> List[NewsItem]:
        all_entries: List[Dict[str, Any]] = []
        for feed_url in self._feeds:
            all_entries.extend(self._read_feed(feed_url))
        return self._entries_to_items(all_entries, limit=limit)

    def _read_feed(self, url: str) -> List[Dict[str, Any]]:
        try:
            parsed = self._feedparser.parse(url)
        except Exception as exc:
            self._logger.warning("RSS feed failed: %s", url, exc_info=exc)
            return []

        if getattr(parsed, "bozo", False):
            self._logger.debug("RSS feed parse issues: %s", url)
        return list(getattr(parsed, "entries", []) or [])

    def _entries_to_items(
        self, entries: Sequence[Dict[str, Any]], limit: int, symbol: Optional[str] = None
    ) -> List[NewsItem]:
        items: List[NewsItem] = []
        for entry in entries:
            if len(items) >= int(limit):
                break
            title = (entry.get("title") or "").strip()
            if not title:
                continue
            summary = entry.get("summary") or entry.get("description")
            source = _rss_source(entry)
            published_at = _rss_published(entry)
            url = entry.get("link")
            symbols = [symbol] if symbol else []
            items.append(
                NewsItem(
                    title=title,
                    summary=summary,
                    source=source,
                    published_at=published_at,
                    url=url,
                    symbols=symbols,
                    sentiment_score=None,
                )
            )
        return items


class CompositeNewsProvider(NewsProvider):
    """News provider that tries Alpha Vantage first, then RSS fallback."""

    def __init__(
        self,
        primary: Optional[NewsProvider] = None,
        fallback: Optional[NewsProvider] = None,
        similarity_threshold: float = 0.9,
    ) -> None:
        self._primary = primary
        self._fallback = fallback or RSSNewsProvider()
        self._similarity_threshold = float(similarity_threshold)
        self._logger = logging.getLogger(__name__)

    def get_headlines(self, symbol: str, limit: int = 10) -> List[NewsItem]:
        items = self._fetch_with_fallback(lambda: self._primary.get_headlines(symbol, limit))
        if items is None:
            items = self._fallback.get_headlines(symbol, limit)
        return self._post_process(items, limit)

    def get_market_news(self, limit: int = 20) -> List[NewsItem]:
        items = self._fetch_with_fallback(lambda: self._primary.get_market_news(limit))
        if items is None:
            items = self._fallback.get_market_news(limit)
        return self._post_process(items, limit)

    def _fetch_with_fallback(self, fn: Any) -> Optional[List[NewsItem]]:
        if not self._primary:
            return None
        try:
            return fn()
        except Exception as exc:
            self._logger.warning("Primary news provider failed; falling back", exc_info=exc)
            return None

    def _post_process(self, items: Iterable[NewsItem], limit: int) -> List[NewsItem]:
        deduped: List[NewsItem] = []
        for item in sorted(items, key=lambda x: x.published_at, reverse=True):
            if not _is_duplicate(item, deduped, self._similarity_threshold):
                deduped.append(item)
            if len(deduped) >= int(limit):
                break
        return deduped


def create_news_provider(config: Optional[Dict[str, Any]] = None) -> NewsProvider:
    """Factory for NewsProvider implementations."""
    config = config or {}
    provider = str(config.get("provider", "composite")).strip().lower()

    if provider in {"alpha", "alphavantage", "alpha_vantage"}:
        return AlphaVantageNewsProvider(config=config)
    if provider in {"rss", "feed", "feeds"}:
        return RSSNewsProvider(config=config)
    if provider in {"composite", "auto", "default"}:
        primary: Optional[NewsProvider] = None
        try:
            primary = AlphaVantageNewsProvider(config=config)
        except Exception as exc:
            logging.getLogger(__name__).warning("Alpha Vantage unavailable: %s", exc)
        fallback = RSSNewsProvider(config=config)
        return CompositeNewsProvider(primary=primary, fallback=fallback)

    raise ValueError(f"Unsupported news provider: {provider}")


def _parse_av_time(value: Any) -> datetime:
    if not value:
        return datetime.utcnow()
    if isinstance(value, datetime):
        return value
    text = str(value)
    for fmt in ("%Y%m%dT%H%M%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return datetime.utcnow()


def _extract_symbols(ticker_sentiment: Any, fallback_symbol: Optional[str]) -> List[str]:
    symbols: List[str] = []
    if isinstance(ticker_sentiment, list):
        for entry in ticker_sentiment:
            ticker = entry.get("ticker") if isinstance(entry, dict) else None
            if ticker:
                symbols.append(str(ticker).upper())
    if not symbols and fallback_symbol:
        symbols = [fallback_symbol]
    return symbols


def _extract_sentiment(entry: Dict[str, Any], symbol: Optional[str]) -> Optional[float]:
    if symbol:
        ticker_list = entry.get("ticker_sentiment")
        if isinstance(ticker_list, list):
            for ticker_entry in ticker_list:
                if not isinstance(ticker_entry, dict):
                    continue
                ticker = str(ticker_entry.get("ticker", "")).upper()
                if ticker == symbol.upper():
                    return _coerce_float(ticker_entry.get("ticker_sentiment_score"))
    return _coerce_float(entry.get("overall_sentiment_score"))


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rss_published(entry: Dict[str, Any]) -> datetime:
    published = entry.get("published_parsed") or entry.get("updated_parsed")
    if published is None:
        return datetime.utcnow()
    try:
        return datetime.utcfromtimestamp(time.mktime(published))
    except Exception:
        return datetime.utcnow()


def _rss_source(entry: Dict[str, Any]) -> str:
    source = entry.get("source")
    if isinstance(source, dict):
        title = source.get("title")
        if title:
            return str(title)
    return str(entry.get("feedburner_origlink") or entry.get("author") or "RSS")


def _normalize_title(title: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", title.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _is_duplicate(item: NewsItem, existing: Sequence[NewsItem], threshold: float) -> bool:
    if not existing:
        return False
    title = _normalize_title(item.title)
    for other in existing:
        other_title = _normalize_title(other.title)
        if not title or not other_title:
            continue
        if title == other_title:
            return True
        similarity = SequenceMatcher(None, title, other_title).ratio()
        if similarity >= threshold:
            return True
    return False
