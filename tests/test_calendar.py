"""Tests for calendar data provider."""

from datetime import date, timedelta

import pytest

from src.data.calendar_data import CalendarProvider, create_calendar_provider


class TestCalendarProvider:
    """Tests for CalendarProvider."""

    def test_fed_day_detection(self):
        provider = CalendarProvider()
        # 2025-01-29 is a known Fed meeting date
        assert provider.is_fed_day(date(2025, 1, 29)) is True
        assert provider.is_fed_day(date(2025, 1, 28)) is False

    def test_fed_week_detection(self):
        provider = CalendarProvider()
        # 2025-01-29 (Wednesday) is Fed day, so Mon-Fri of that week should be fed_week
        fed_date = date(2025, 1, 29)
        # Monday of that week
        monday = fed_date - timedelta(days=fed_date.weekday())
        assert provider.is_fed_week(monday) is True
        assert provider.is_fed_week(monday + timedelta(days=4)) is True  # Friday
        # Week before should not be fed week
        assert provider.is_fed_week(monday - timedelta(days=7)) is False

    def test_holiday_detection(self):
        provider = CalendarProvider()
        # 2025-12-25 is Christmas
        assert provider.is_holiday(date(2025, 12, 25)) is True
        assert provider.is_holiday(date(2025, 12, 24)) is False

    def test_earnings_blackout_window(self):
        config = {
            "earnings_blackout_days_before": 2,
            "earnings_blackout_days_after": 1,
            "earnings_dates": {
                "AAPL": ["2025-02-10"],
            },
        }
        provider = CalendarProvider(config)

        # 2 days before earnings
        assert provider.is_earnings_blackout("AAPL", date(2025, 2, 8)) is True
        # Day of earnings
        assert provider.is_earnings_blackout("AAPL", date(2025, 2, 10)) is True
        # 1 day after earnings
        assert provider.is_earnings_blackout("AAPL", date(2025, 2, 11)) is True
        # 2 days after - outside window
        assert provider.is_earnings_blackout("AAPL", date(2025, 2, 12)) is False
        # 3 days before - outside window
        assert provider.is_earnings_blackout("AAPL", date(2025, 2, 7)) is False

    def test_earnings_blackout_case_insensitive(self):
        config = {
            "earnings_dates": {
                "AAPL": ["2025-02-10"],
            },
        }
        provider = CalendarProvider(config)
        assert provider.is_earnings_blackout("aapl", date(2025, 2, 10)) is True
        assert provider.is_earnings_blackout("Aapl", date(2025, 2, 10)) is True

    def test_no_earnings_configured(self):
        provider = CalendarProvider()
        # Should return False if no earnings dates configured
        assert provider.is_earnings_blackout("AAPL", date(2025, 2, 10)) is False

    def test_get_calendar_flags_all_fields(self):
        config = {
            "earnings_dates": {
                "AAPL": ["2025-01-29"],  # Same as Fed day for testing
            },
        }
        provider = CalendarProvider(config)
        flags = provider.get_calendar_flags("AAPL", date(2025, 1, 29))

        assert "earnings_soon" in flags
        assert "fed_day" in flags
        assert "fed_week" in flags
        assert "macro_event_soon" in flags
        assert "is_holiday" in flags
        assert "blackout_active" in flags

        # On this date, earnings_soon and fed_day should both be True
        assert flags["earnings_soon"] is True
        assert flags["fed_day"] is True
        assert flags["blackout_active"] is True

    def test_blackout_active_on_fed_day(self):
        provider = CalendarProvider()
        flags = provider.get_calendar_flags("SPY", date(2025, 1, 29))
        assert flags["fed_day"] is True
        assert flags["blackout_active"] is True

    def test_blackout_active_on_holiday(self):
        provider = CalendarProvider()
        flags = provider.get_calendar_flags("SPY", date(2025, 12, 25))
        assert flags["is_holiday"] is True
        assert flags["blackout_active"] is True

    def test_add_earnings_date_manually(self):
        provider = CalendarProvider()
        provider.add_earnings_date("MSFT", date(2025, 3, 15))
        assert provider.is_earnings_blackout("MSFT", date(2025, 3, 15)) is True

    def test_factory_function(self):
        provider = create_calendar_provider({"earnings_blackout_days_before": 3})
        assert provider._earnings_blackout_days_before == 3


class TestCalendarBlackoutInRiskGate:
    """Integration tests for calendar blackout in risk gate."""

    def test_blackout_blocks_new_buy(self):
        from src.risk.risk_gate import RiskGate

        class MockBroker:
            pass

        config = {}
        gate = RiskGate(config, MockBroker())

        account_info = {"equity": 100000, "prices": {"AAPL": 150.0}}
        positions = []

        # With blackout active, new buys should be blocked
        features = {"blackout_active": True}
        approved, reasons = gate.check_order(
            symbol="AAPL",
            side="BUY",
            qty=10,
            current_positions=positions,
            account_info=account_info,
            features=features,
        )

        assert "calendar_blackout" in reasons

    def test_blackout_allows_reducing_position(self):
        from src.risk.risk_gate import RiskGate

        class MockBroker:
            pass

        config = {"blackout_reduce_only": True}
        gate = RiskGate(config, MockBroker())

        account_info = {"equity": 100000, "prices": {"AAPL": 150.0}}
        # Existing long position
        positions = [{"symbol": "AAPL", "qty": 100, "market_value": 15000}]

        # With blackout active, selling existing position should be allowed
        features = {"blackout_active": True}
        approved, reasons = gate.check_order(
            symbol="AAPL",
            side="SELL",
            qty=50,
            current_positions=positions,
            account_info=account_info,
            features=features,
        )

        assert "calendar_blackout" not in reasons

    def test_no_blackout_allows_trading(self):
        from src.risk.risk_gate import RiskGate

        class MockBroker:
            pass

        config = {}
        gate = RiskGate(config, MockBroker())

        account_info = {"equity": 100000, "prices": {"AAPL": 150.0}}
        positions = []

        # Without blackout, buys should be allowed
        features = {"blackout_active": False}
        approved, reasons = gate.check_order(
            symbol="AAPL",
            side="BUY",
            qty=10,
            current_positions=positions,
            account_info=account_info,
            features=features,
        )

        assert "calendar_blackout" not in reasons
