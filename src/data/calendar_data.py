"""Calendar data provider for earnings dates and macro events."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Mapping, Optional, Set


@dataclass(frozen=True)
class CalendarEvent:
    """A calendar event that may affect trading."""
    event_type: str  # "earnings", "fed", "macro", "holiday"
    symbol: Optional[str]  # None for market-wide events
    event_date: date
    description: str
    is_blackout: bool = False  # If True, risk gate should block/reduce


class CalendarProvider:
    """
    Provides calendar-based flags for trading decisions.

    Supports:
    - Earnings blackout windows (configurable days before/after)
    - Fed meeting dates
    - Macro event flags
    - Trading holidays
    """

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self._config = dict(config or {})
        self._logger = logging.getLogger(__name__)

        # Blackout window configuration
        self._earnings_blackout_days_before = int(
            self._config.get("earnings_blackout_days_before", 2)
        )
        self._earnings_blackout_days_after = int(
            self._config.get("earnings_blackout_days_after", 1)
        )

        # Manual earnings dates (symbol -> list of dates)
        self._earnings_dates: Dict[str, List[date]] = {}
        self._load_manual_earnings()

        # Known Fed meeting dates for 2024-2026 (can be extended)
        self._fed_dates: Set[date] = self._load_fed_dates()

        # US market holidays
        self._holidays: Set[date] = self._load_holidays()

    def _load_manual_earnings(self) -> None:
        """Load manually configured earnings dates."""
        earnings_config = self._config.get("earnings_dates", {})
        if isinstance(earnings_config, dict):
            for symbol, dates in earnings_config.items():
                if isinstance(dates, list):
                    parsed = []
                    for d in dates:
                        try:
                            if isinstance(d, date):
                                parsed.append(d)
                            elif isinstance(d, str):
                                parsed.append(date.fromisoformat(d))
                        except Exception:
                            pass
                    if parsed:
                        self._earnings_dates[str(symbol).upper()] = parsed

    def _load_fed_dates(self) -> Set[date]:
        """Load Fed meeting dates."""
        # 2024-2026 FOMC meeting dates (announcement days)
        fed_dates = [
            # 2024
            "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
            "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
            # 2025
            "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
            "2025-07-30", "2025-09-17", "2025-11-05", "2025-12-17",
            # 2026
            "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
            "2026-07-29", "2026-09-16", "2026-11-04", "2026-12-16",
        ]
        result = set()
        for d in fed_dates:
            try:
                result.add(date.fromisoformat(d))
            except Exception:
                pass
        return result

    def _load_holidays(self) -> Set[date]:
        """Load US market holidays."""
        # Major US market holidays 2024-2026
        holidays = [
            # 2024
            "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29",
            "2024-05-27", "2024-06-19", "2024-07-04", "2024-09-02",
            "2024-11-28", "2024-12-25",
            # 2025
            "2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18",
            "2025-05-26", "2025-06-19", "2025-07-04", "2025-09-01",
            "2025-11-27", "2025-12-25",
            # 2026
            "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03",
            "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07",
            "2026-11-26", "2026-12-25",
        ]
        result = set()
        for d in holidays:
            try:
                result.add(date.fromisoformat(d))
            except Exception:
                pass
        return result

    def is_earnings_blackout(self, symbol: str, check_date: Optional[date] = None) -> bool:
        """Check if symbol is in earnings blackout window."""
        check_date = check_date or date.today()
        symbol = symbol.upper()

        earnings_list = self._earnings_dates.get(symbol, [])
        for earnings_date in earnings_list:
            start = earnings_date - timedelta(days=self._earnings_blackout_days_before)
            end = earnings_date + timedelta(days=self._earnings_blackout_days_after)
            if start <= check_date <= end:
                return True
        return False

    def is_fed_day(self, check_date: Optional[date] = None) -> bool:
        """Check if date is a Fed meeting announcement day."""
        check_date = check_date or date.today()
        return check_date in self._fed_dates

    def is_fed_week(self, check_date: Optional[date] = None) -> bool:
        """Check if date is within a Fed meeting week (Mon-Fri of announcement)."""
        check_date = check_date or date.today()
        for fed_date in self._fed_dates:
            # Get Monday of Fed week
            days_since_monday = fed_date.weekday()
            week_start = fed_date - timedelta(days=days_since_monday)
            week_end = week_start + timedelta(days=4)  # Friday
            if week_start <= check_date <= week_end:
                return True
        return False

    def is_holiday(self, check_date: Optional[date] = None) -> bool:
        """Check if date is a market holiday."""
        check_date = check_date or date.today()
        return check_date in self._holidays

    def get_calendar_flags(
        self, symbol: str, check_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """
        Get all calendar flags for a symbol on a date.

        Returns:
            earnings_soon: bool - in earnings blackout window
            fed_day: bool - Fed announcement day
            fed_week: bool - during Fed meeting week
            macro_event_soon: bool - any significant event
            is_holiday: bool - market holiday
            blackout_active: bool - should reduce/block trading
        """
        check_date = check_date or date.today()

        earnings_soon = self.is_earnings_blackout(symbol, check_date)
        fed_day = self.is_fed_day(check_date)
        fed_week = self.is_fed_week(check_date)
        is_holiday = self.is_holiday(check_date)

        macro_event_soon = fed_day or fed_week
        blackout_active = earnings_soon or fed_day or is_holiday

        return {
            "earnings_soon": earnings_soon,
            "fed_day": fed_day,
            "fed_week": fed_week,
            "macro_event_soon": macro_event_soon,
            "is_holiday": is_holiday,
            "blackout_active": blackout_active,
        }

    def add_earnings_date(self, symbol: str, earnings_date: date) -> None:
        """Manually add an earnings date for a symbol."""
        symbol = symbol.upper()
        if symbol not in self._earnings_dates:
            self._earnings_dates[symbol] = []
        if earnings_date not in self._earnings_dates[symbol]:
            self._earnings_dates[symbol].append(earnings_date)
            self._earnings_dates[symbol].sort()


def create_calendar_provider(config: Optional[Mapping[str, Any]] = None) -> CalendarProvider:
    """Factory for CalendarProvider."""
    return CalendarProvider(config)
