# orchestrator/scheduler.py — Day 60 | Task Scheduler
# ============================================================
# Time-based task scheduling for autonomous routines.
# Supports recurring tasks and one-time scheduled events.
#
# Scheduled Tasks:
#   - Morning briefing (daily)
#   - Research cycle (every N hours)
#   - Monte Carlo simulation (weekly)
#   - Position monitoring (every 5 minutes)
#   - State backup (every 30 minutes)
#   - Weekly report (Sunday)
# ============================================================

import time
from datetime import datetime, timezone, timedelta
from typing import Callable, Optional

from utils.logger import get_logger

log = get_logger("scheduler")


class ScheduledTask:
    """A task scheduled to run at specific intervals."""

    def __init__(
        self,
        name: str,
        func: Callable,
        interval_seconds: int = None,
        run_at_hour: int = None,
        run_at_minute: int = None,
        run_on_weekday: int = None,  # 0=Monday, 6=Sunday
        one_time: bool = False,
        enabled: bool = True,
    ):
        self.name = name
        self.func = func
        self.interval_seconds = interval_seconds
        self.run_at_hour = run_at_hour
        self.run_at_minute = run_at_minute or 0
        self.run_on_weekday = run_on_weekday
        self.one_time = one_time
        self.enabled = enabled
        self.last_run = None
        self.run_count = 0
        self.last_error = None

    def should_run(self) -> bool:
        """Check if the task should run now."""
        if not self.enabled:
            return False
        if self.one_time and self.run_count > 0:
            return False

        now = datetime.now(timezone.utc)

        # Day-of-week check
        if self.run_on_weekday is not None and now.weekday() != self.run_on_weekday:
            return False

        # Time-of-day check
        if self.run_at_hour is not None:
            if now.hour != self.run_at_hour or now.minute < self.run_at_minute:
                return False
            # Day 102+ CRITICAL hotfix: don't run twice in the same hour.
            # Previously: only blocked same MINUTE (last_run.hour==now.hour
            # AND last_run.minute==now.minute). After firing at 09:30, at
            # 09:31 the check passed (different minute) and the task fired
            # AGAIN — repeating every minute through 09:59, so tasks ran
            # ~30× per hour instead of once.
            # Fix: block if already ran in this same (date, hour) bucket.
            if self.last_run:
                if (self.last_run.date() == now.date()
                        and self.last_run.hour == now.hour):
                    return False

        # Interval check
        if self.interval_seconds and self.last_run:
            elapsed = (now - self.last_run).total_seconds()
            if elapsed < self.interval_seconds:
                return False

        return True

    def execute(self) -> dict:
        """Execute the task and record timing."""
        start = time.time()
        try:
            result = self.func()
            self.last_run = datetime.now(timezone.utc)
            self.run_count += 1
            self.last_error = None
            return {
                "task": self.name,
                "success": True,
                "duration": round(time.time() - start, 2),
                "result": result,
            }
        except Exception as e:
            self.last_run = datetime.now(timezone.utc)
            self.run_count += 1
            self.last_error = str(e)
            return {
                "task": self.name,
                "success": False,
                "error": str(e),
                "duration": round(time.time() - start, 2),
            }


class TaskScheduler:
    """
    Central task scheduler for autonomous routines.
    """

    def __init__(self):
        self._tasks: list[ScheduledTask] = []
        self._running = False

    def schedule(
        self,
        name: str,
        func: Callable,
        interval_seconds: int = None,
        run_at_hour: int = None,
        run_at_minute: int = None,
        run_on_weekday: int = None,
        one_time: bool = False,
        enabled: bool = True,
    ) -> ScheduledTask:
        """Schedule a new task."""
        task = ScheduledTask(
            name=name,
            func=func,
            interval_seconds=interval_seconds,
            run_at_hour=run_at_hour,
            run_at_minute=run_at_minute,
            run_on_weekday=run_on_weekday,
            one_time=one_time,
            enabled=enabled,
        )
        self._tasks.append(task)
        log.debug(f"[Scheduler] Scheduled: {name}")
        return task

    def tick(self) -> list[dict]:
        """
        Check all scheduled tasks and execute any that are due.
        Returns list of execution results.
        """
        if not self._running:
            return []

        results = []
        for task in self._tasks:
            if task.should_run():
                log.info(f"[Scheduler] Running task: {task.name}")
                result = task.execute()
                results.append(result)
                if not result["success"]:
                    log.warning(f"[Scheduler] Task '{task.name}' failed: {result.get('error')}")
        return results

    def start(self) -> None:
        self._running = True
        log.info(f"[Scheduler] Started with {len(self._tasks)} tasks")

    def stop_all(self) -> None:
        self._running = False
        log.info("[Scheduler] Stopped")

    def get_tasks(self) -> list[dict]:
        """Get all scheduled tasks with status."""
        return [
            {
                "name": t.name,
                "enabled": t.enabled,
                "last_run": t.last_run.isoformat() if t.last_run else None,
                "run_count": t.run_count,
                "interval_seconds": t.interval_seconds,
                "run_at_hour": t.run_at_hour,
                "last_error": t.last_error,
            }
            for t in self._tasks
        ]

    def get_stats(self) -> dict:
        return {
            "total_tasks": len(self._tasks),
            "enabled_tasks": sum(1 for t in self._tasks if t.enabled),
            "running": self._running,
            "total_executions": sum(t.run_count for t in self._tasks),
        }
