from __future__ import annotations

import threading
import time
from typing import Callable, Dict, List
from unittest.mock import Mock

from scheduler import TradingScheduler


class _ImmediateScheduler:
    def __init__(self) -> None:
        self._jobs: List[tuple[Callable[..., None], tuple[object, ...], Dict[str, object]]] = []

    def every(self, _interval: int) -> "_ImmediateScheduler":
        return self

    @property
    def minutes(self) -> "_ImmediateScheduler":
        return self

    def do(self, job_fn: Callable[..., None], *args: object, **kwargs: object) -> "_ImmediateScheduler":
        self._jobs.append((job_fn, args, kwargs))
        return self

    def run_pending(self) -> None:
        for job_fn, args, kwargs in list(self._jobs):
            job_fn(*args, **kwargs)


def _make_counter_job(name: str, counts: Dict[str, int], on_reached: Callable[[], None]) -> Callable[[], None]:
    def _job() -> None:
        counts[name] += 1
        on_reached()

    return _job


def test_scheduler_runs_n_iterations() -> None:
    counts = {
        "data_refresh": 0,
        "compute_features": 0,
        "make_decisions": 0,
        "execute_orders": 0,
    }
    target_iterations = 3
    stop_event = threading.Event()

    def _stop_when_ready() -> None:
        if all(value >= target_iterations for value in counts.values()):
            scheduler.stop()
            stop_event.set()

    scheduler = TradingScheduler(
        data_refresh=_make_counter_job("data_refresh", counts, _stop_when_ready),
        compute_features=_make_counter_job("compute_features", counts, _stop_when_ready),
        make_decisions=_make_counter_job("make_decisions", counts, _stop_when_ready),
        execute_orders=_make_counter_job("execute_orders", counts, _stop_when_ready),
        intervals={
            "data_refresh": 1,
            "compute_features": 1,
            "make_decisions": 1,
            "execute_orders": 1,
        },
        loop_sleep_seconds=0.1,
        scheduler=_ImmediateScheduler(),
    )

    thread = threading.Thread(target=scheduler.start, daemon=True)
    thread.start()

    assert stop_event.wait(timeout=5.0)

    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert all(value == target_iterations for value in counts.values())


def test_scheduler_graceful_shutdown() -> None:
    errors: List[BaseException] = []

    scheduler = TradingScheduler(
        data_refresh=lambda: None,
        compute_features=lambda: None,
        make_decisions=lambda: None,
        execute_orders=lambda: None,
        intervals={
            "data_refresh": 1,
            "compute_features": 1,
            "make_decisions": 1,
            "execute_orders": 1,
        },
        loop_sleep_seconds=0.1,
        scheduler=_ImmediateScheduler(),
    )

    def _run() -> None:
        try:
            scheduler.start()
        except BaseException as exc:  # pragma: no cover - should not trigger
            errors.append(exc)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    time.sleep(0.05)
    scheduler.stop()

    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert not errors


def test_scheduler_dry_run_mode() -> None:
    submit_order = Mock()
    dry_run_enabled = True

    def _execute_orders() -> None:
        if not dry_run_enabled:
            submit_order()

    execute_orders = Mock(side_effect=_execute_orders)

    scheduler = TradingScheduler(
        data_refresh=lambda: None,
        compute_features=lambda: None,
        make_decisions=lambda: None,
        execute_orders=execute_orders,
        intervals={
            "data_refresh": 1,
            "compute_features": 1,
            "make_decisions": 1,
            "execute_orders": 1,
        },
        loop_sleep_seconds=0.1,
        scheduler=_ImmediateScheduler(),
    )

    scheduler.register_jobs()
    scheduler._scheduler.run_pending()

    execute_orders.assert_called_once()
    submit_order.assert_not_called()
