# orchestrator/daily_routine.py — Day 60 | Autonomous Daily Routine
# ============================================================
# AI schedule across the trading week:
#
# Morning (pre-market):
#   - Economic calendar check
#   - Market briefing generation
#   - Risk adjustment based on overnight gaps
#
# Trading Hours:
#   - Scan → Analyze → Trade → Monitor
#   - Position monitoring loop
#
# Evening (post-market):
#   - Performance review
#   - Mistake analysis
#   - Learning update
#
# Sunday (pre-week):
#   - Strategy research
#   - Optimization
#   - Weekly report
# ============================================================

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from utils.logger import get_logger

log = get_logger("daily_routine")

if TYPE_CHECKING:
    from orchestrator.trading_orchestrator import TradingOrchestrator


class DailyRoutineManager:
    """
    Manages autonomous daily, weekly, and periodic routines.
    Scheduled via TaskScheduler within the orchestrator.
    """

    def __init__(self, orchestrator: "TradingOrchestrator"):
        self.orch = orchestrator
        self._morning_done = False
        self._evening_done = False
        self._sunday_done = False
        self._last_morning_date = None
        self._last_evening_date = None
        self._last_sunday_date = None

    def setup(self):
        """Register all routine tasks with the scheduler."""
        # Morning briefing — every day at 06:00 UTC (before London open)
        self.orch.scheduler.schedule(
            name="morning_briefing",
            func=self._morning_routine,
            run_at_hour=6,
            run_at_minute=0,
        )

        # Evening review — every day at 21:00 UTC (after NY close)
        self.orch.scheduler.schedule(
            name="evening_review",
            func=self._evening_routine,
            run_at_hour=21,
            run_at_minute=0,
        )

        # Sunday research — every Sunday at 18:00 UTC (before week open)
        self.orch.scheduler.schedule(
            name="sunday_research",
            func=self._sunday_routine,
            run_on_weekday=6,  # Sunday
            run_at_hour=18,
            run_at_minute=0,
        )

        # Position monitoring — every 5 minutes during trading
        self.orch.scheduler.schedule(
            name="position_monitor",
            func=self._position_monitor,
            interval_seconds=300,
        )

        # State backup — every 30 minutes
        self.orch.scheduler.schedule(
            name="state_backup",
            func=self._state_backup,
            interval_seconds=1800,
        )

        # Research cycle — every 4 hours
        self.orch.scheduler.schedule(
            name="research_cycle",
            func=self._research_cycle,
            interval_seconds=14400,
        )

        log.info("[DailyRoutine] All routines scheduled")

    def execute_scheduled_tasks(self) -> list[dict]:
        """Execute all due scheduled tasks. Called at start of each trading cycle."""
        return self.orch.scheduler.tick()

    # ──────────────────────────────────────────────────
    # MORNING ROUTINE
    # ──────────────────────────────────────────────────

    def _morning_routine(self) -> dict:
        """
        Morning routine (06:00 UTC):
        1. Economic calendar check
        2. Market briefing
        3. Risk adjustment
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        if self._last_morning_date == today:
            return {"status": "already_done_today"}

        self._last_morning_date = today
        log.info("=" * 50)
        log.info("  MORNING ROUTINE")
        log.info("=" * 50)

        results = {}

        # 1. Economic Calendar
        try:
            from broker.economic_calendar import EconomicCalendar
            ec = EconomicCalendar()
            events = ec.get_today_events()
            results["calendar_events"] = len(events) if events else 0
            log.info(f"  [Morning] Economic events today: {results['calendar_events']}")

            if events:
                self.orch.bus.publish_message(
                    source="daily_routine",
                    msg_type="system_event",
                    data={"event": "morning_briefing", "high_impact_events": len(events)},
                )
        except Exception as e:
            log.warning(f"[Morning] Calendar check failed: {e}")
            results["calendar_error"] = str(e)

        # 2. Market Briefing
        try:
            briefing = self._generate_briefing()
            results["briefing"] = briefing
            log.info(f"  [Morning] Market briefing: {briefing.get('summary', 'N/A')}")
        except Exception as e:
            log.warning(f"[Morning] Briefing failed: {e}")

        # 3. Risk Adjustment
        try:
            if self.orch._risk_manager:
                rm_stats = self.orch._risk_manager.get_stats()
                results["risk_mode"] = rm_stats.get("current_mode", "NORMAL")
                log.info(f"  [Morning] Risk mode: {results['risk_mode']}")
        except Exception as e:
            log.warning(f"[Morning] Risk adjustment failed: {e}")

        log.info("  Morning routine complete")
        log.info("=" * 50)
        return results

    # ──────────────────────────────────────────────────
    # EVENING ROUTINE
    # ──────────────────────────────────────────────────

    def _evening_routine(self) -> dict:
        """
        Evening routine (21:00 UTC):
        1. Performance review
        2. Mistake analysis
        3. Learning update
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        if self._last_evening_date == today:
            return {"status": "already_done_today"}

        self._last_evening_date = today
        log.info("=" * 50)
        log.info("  EVENING ROUTINE")
        log.info("=" * 50)

        results = {}

        # 1. Performance Review
        try:
            if self.orch._paper_trader:
                dashboard = self.orch._paper_trader.get_dashboard()
                results["performance"] = dashboard
                log.info(f"  [Evening] Balance: ${dashboard.get('balance', 0):,.2f}")
                log.info(f"  [Evening] Win Rate: {dashboard.get('win_rate', 0)}%")
                log.info(f"  [Evening] Total Trades: {dashboard.get('total_trades', 0)}")
        except Exception as e:
            log.warning(f"[Evening] Performance review failed: {e}")

        # 2. Mistake Analysis
        try:
            if self.orch._learning_agent:
                stats = self.orch._learning_agent.get_performance_stats()
                results["learning"] = stats
                log.info(f"  [Evening] Learning stats: {stats.get('win_rate', 0)}% win rate")
        except Exception as e:
            log.warning(f"[Evening] Mistake analysis failed: {e}")

        # 3. Decision Journal Review
        try:
            journal_stats = self.orch.journal.get_stats()
            results["journal"] = journal_stats
            log.info(f"  [Evening] Journal entries: {journal_stats['total_entries']}")
            if journal_stats.get("entries_with_outcome", 0) > 0:
                log.info(f"  [Evening] Journal win rate: {journal_stats['win_rate']}%")
        except Exception as e:
            log.warning(f"[Evening] Journal review failed: {e}")

        log.info("  Evening routine complete")
        log.info("=" * 50)
        return results

    # ──────────────────────────────────────────────────
    # SUNDAY ROUTINE
    # ──────────────────────────────────────────────────

    def _sunday_routine(self) -> dict:
        """
        Sunday routine (18:00 UTC):
        1. Strategy research
        2. Optimization
        3. Weekly report
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        if self._last_sunday_date == today:
            return {"status": "already_done_this_sunday"}

        self._last_sunday_date = today
        log.info("=" * 50)
        log.info("  SUNDAY RESEARCH ROUTINE")
        log.info("=" * 50)

        results = {}

        # 1. Research Cycle
        try:
            if self.orch._research_agent:
                research_report = self.orch._research_agent.run_research_cycle(n_experiments=3)
                results["research"] = {
                    "experiments": len(research_report.get("experiments", [])),
                    "approved": len(research_report.get("strategies_approved", [])),
                    "rejected": len(research_report.get("strategies_rejected", [])),
                }
                log.info(f"  [Sunday] Research: {results['research']['experiments']} experiments, "
                         f"{results['research']['approved']} approved")
        except Exception as e:
            log.warning(f"[Sunday] Research failed: {e}")

        # 2. Monte Carlo Simulation
        try:
            if self.orch._risk_manager:
                mc_result = self.orch._risk_manager.run_risk_simulation(
                    n_simulations=5000, n_trades=100
                )
                results["monte_carlo"] = mc_result
                log.info(f"  [Sunday] Monte Carlo simulation complete")
        except Exception as e:
            log.warning(f"[Sunday] Monte Carlo failed: {e}")

        # 3. Weekly Report
        try:
            if self.orch._risk_manager:
                capital_report = self.orch._risk_manager.generate_capital_report()
                results["capital_report"] = "generated"
                log.info(f"  [Sunday] Capital report generated")
        except Exception as e:
            log.warning(f"[Sunday] Capital report failed: {e}")

        # 4. Audit Trail Summary
        try:
            audit_stats = self.orch.audit.get_stats()
            results["audit"] = audit_stats
            log.info(f"  [Sunday] Audit events this week: {audit_stats['total_events']}")
        except Exception as e:
            log.warning(f"[Sunday] Audit summary failed: {e}")

        log.info("  Sunday routine complete")
        log.info("=" * 50)
        return results

    # ──────────────────────────────────────────────────
    # PERIODIC ROUTINES
    # ──────────────────────────────────────────────────

    def _position_monitor(self) -> dict:
        """Monitor open positions — check for SL/TP."""
        try:
            if self.orch._paper_trader and self.orch._paper_trader.open_positions:
                positions = self.orch._paper_trader.open_positions
                self.orch.state_mgr.update(
                    active_trades=len(positions),
                    current_task=f"Monitoring {len(positions)} position(s)",
                )
                return {"active_positions": len(positions)}
            return {"active_positions": 0}
        except Exception as e:
            log.warning(f"[Monitor] Position monitor error: {e}")
            return {"error": str(e)}

    def _state_backup(self) -> dict:
        """Periodic state backup."""
        try:
            self.orch.bus.save_history()
            self.orch.journal.save()
            self.orch.audit.save()
            return {"status": "saved"}
        except Exception as e:
            return {"error": str(e)}

    def _research_cycle(self) -> dict:
        """Periodic research cycle."""
        try:
            if self.orch._research_agent:
                report = self.orch._research_agent.run_research_cycle(n_experiments=1)
                return {
                    "experiments": len(report.get("experiments", [])),
                    "approved": len(report.get("strategies_approved", [])),
                }
            return {"status": "research_disabled"}
        except Exception as e:
            return {"error": str(e)}

    # ──────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────

    def _generate_briefing(self) -> dict:
        """Generate morning market briefing."""
        state = self.orch.state_mgr.state
        return {
            "summary": f"Market {state.market_status}, Risk {state.risk_mode}",
            "mode": state.mode,
            "balance": state.balance,
            "daily_pnl_pct": state.daily_pnl_pct,
        }
