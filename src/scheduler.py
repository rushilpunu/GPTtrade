"""Trading scheduler for recurring system jobs."""

from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional

import schedule


JobFn = Callable[[], None]


@dataclass(frozen=True)
class JobConfig:
    name: str
    interval_minutes: int
    action: JobFn


class TradingScheduler:
    """Run trading system jobs on configurable intervals using `schedule`."""

    _DEFAULT_JOBS = ("data_refresh", "compute_features", "make_decisions", "execute_orders")

    def __init__(
        self,
        data_refresh: JobFn,
        compute_features: JobFn,
        make_decisions: JobFn,
        execute_orders: JobFn,
        *,
        intervals: Optional[Mapping[str, int]] = None,
        default_interval_minutes: int = 60,
        loop_sleep_seconds: float = 1.0,
        dry_run: bool = False,
        scheduler: Optional[schedule.Scheduler] = None,
        install_signal_handlers: bool = True,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._scheduler = scheduler or schedule.Scheduler()
        self._loop_sleep_seconds = loop_sleep_seconds
        self._dry_run = dry_run
        self._should_install_signal_handlers = install_signal_handlers
        self._stop_requested = False
        self._registered = False
        self._prior_signal_handlers: Dict[int, object] = {}

        intervals = dict(intervals or {})
        self._jobs = {
            "data_refresh": JobConfig(
                name="data_refresh",
                interval_minutes=self._resolve_interval(
                    intervals, "data_refresh", default_interval_minutes
                ),
                action=data_refresh,
            ),
            "compute_features": JobConfig(
                name="compute_features",
                interval_minutes=self._resolve_interval(
                    intervals, "compute_features", default_interval_minutes
                ),
                action=compute_features,
            ),
            "make_decisions": JobConfig(
                name="make_decisions",
                interval_minutes=self._resolve_interval(
                    intervals, "make_decisions", default_interval_minutes
                ),
                action=make_decisions,
            ),
            "execute_orders": JobConfig(
                name="execute_orders",
                interval_minutes=self._resolve_interval(
                    intervals, "execute_orders", default_interval_minutes
                ),
                action=execute_orders,
            ),
        }

        self._log_unknown_intervals(intervals)

    def register_jobs(self) -> None:
        if self._registered:
            return
        for job in self._jobs.values():
            self._scheduler.every(job.interval_minutes).minutes.do(
                self._run_job, job.name, job.action
            )
            self._logger.info(
                "Scheduled job %s every %d minutes", job.name, job.interval_minutes
            )
        self._registered = True

    def start(self) -> None:
        """Start the scheduler loop and block until stopped."""
        self.register_jobs()
        if self._should_install_signal_handlers:
            self._install_signal_handlers()
        self._logger.info("TradingScheduler started dry_run=%s", self._dry_run)
        try:
            while not self._stop_requested:
                self._scheduler.run_pending()
                time.sleep(self._loop_sleep_seconds)
        finally:
            if self._should_install_signal_handlers:
                self._restore_signal_handlers()
            self._logger.info("TradingScheduler stopped")

    def stop(self) -> None:
        """Request a graceful stop of the scheduler loop."""
        self._stop_requested = True

    def _install_signal_handlers(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._prior_signal_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, self._handle_signal)
            except ValueError:
                self._logger.warning("Signal handlers must be set in main thread")
                break

    def _restore_signal_handlers(self) -> None:
        for sig, handler in self._prior_signal_handlers.items():
            try:
                signal.signal(sig, handler)
            except ValueError:
                self._logger.warning("Cannot restore signal handlers from non-main thread")
                break
        self._prior_signal_handlers.clear()

    def _handle_signal(self, signum: int, _frame: object) -> None:
        self._logger.info("Received signal %s, shutting down", signum)
        self.stop()

    def _run_job(self, name: str, action: JobFn) -> None:
        start = time.perf_counter()
        self._logger.info("Job %s start", name)
        if self._dry_run:
            elapsed = time.perf_counter() - start
            self._logger.info("Job %s dry-run completed in %.2fs", name, elapsed)
            return
        try:
            action()
        except Exception:
            self._logger.exception("Job %s failed", name)
        finally:
            elapsed = time.perf_counter() - start
            self._logger.info("Job %s finished in %.2fs", name, elapsed)

    @staticmethod
    def _resolve_interval(
        intervals: Mapping[str, int], name: str, default_interval_minutes: int
    ) -> int:
        interval = intervals.get(name, default_interval_minutes)
        if interval <= 0:
            raise ValueError(f"Interval for {name} must be positive, got {interval}")
        return interval

    def _log_unknown_intervals(self, intervals: Mapping[str, int]) -> None:
        for name in intervals:
            if name not in self._jobs:
                self._logger.warning("Unknown interval job %s ignored", name)
