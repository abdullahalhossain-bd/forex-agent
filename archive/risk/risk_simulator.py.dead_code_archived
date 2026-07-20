# risk/risk_simulator.py — Day 58 | Risk Scenario Simulator
# ============================================================
# "What if" analysis engine. Simulates specific scenarios
# to understand account impact before they happen.
#
# Scenarios:
#   - Consecutive losses (5, 10, 15 losses in a row)
#   - Consecutive wins
#   - Worst day / best day
#   - Worst week
#   - Black swan events (extreme market moves)
#   - Strategy failure (win rate drops suddenly)
# ============================================================

from utils.logger import get_logger

log = get_logger("risk_simulator")


class RiskScenarioSimulator:
    """
    Risk Scenario Simulator — "What If" Analysis Engine.

    AI আগে simulate করে দেখবে বিভিন্ন scenario-তে
    account-এর কী হতে পারে।

    Example questions this answers:
      - "5টি trade পরপর loss হলে কী হবে?"
      - "Worst case দিনে কতটুকু loss হতে পারে?"
      - "Strategy fail করলে account survive করবে?"
      - "Black swan event-এ impact কত?"

    Usage:
        sim = RiskScenarioSimulator(balance=10000, risk_pct=1.0)
        result = sim.consecutive_losses(n=5)
        result = sim.black_swan(move_pct=5.0)
    """

    def __init__(
        self,
        balance: float = 10000.0,
        initial_balance: float = 10000.0,
        risk_pct: float = 1.0,    # risk per trade as %
        avg_rr: float = 2.0,      # average reward:risk ratio
    ):
        self.balance = balance
        self.initial_balance = initial_balance
        self.risk_pct = risk_pct
        self.avg_rr = avg_rr

        log.info(
            f"[RiskSimulator] Initialized | "
            f"Balance: ${balance:,.2f} | "
            f"Risk: {risk_pct}% | Avg RR: 1:{avg_rr}"
        )

    # ═══════════════════════════════════════════════════════
    # CONSECUTIVE LOSS SCENARIO
    # ═══════════════════════════════════════════════════════

    def consecutive_losses(self, n: int = 5) -> dict:
        """
        Simulate N consecutive losses.

        Example:
            n=5, risk=1%, balance=$10,000
            Each loss = $100 (1% of current balance)
            After 5 losses: $9,509.90
            Impact: -4.9%

        Returns:
            Scenario analysis with survival assessment.
        """
        balance = self.balance
        risk_per_trade = self.risk_pct / 100

        losses = []
        for i in range(n):
            loss = balance * risk_per_trade
            balance -= loss
            balance = max(0, balance)
            losses.append({
                "trade": i + 1,
                "loss_usd": round(loss, 2),
                "balance_after": round(balance, 2),
                "dd_pct": round(
                    (self.balance - balance) / self.balance * 100, 2
                ),
            })

        total_loss = self.balance - balance
        impact_pct = total_loss / self.balance * 100
        dd_pct = impact_pct
        survival = balance > self.initial_balance * 0.5

        result = {
            "scenario": f"{n} consecutive losses",
            "n_losses": n,
            "starting_balance": round(self.balance, 2),
            "ending_balance": round(balance, 2),
            "total_loss_usd": round(total_loss, 2),
            "impact_pct": round(impact_pct, 2),
            "drawdown_pct": round(dd_pct, 2),
            "survival": survival,
            "survival_threshold": "Balance > 50% of initial",
            "trade_by_trade": losses,
            "risk_after": "DEFENSIVE" if impact_pct > 5 else (
                "NORMAL" if impact_pct < 3 else "CAUTION"
            ),
        }

        log.info(
            f"[RiskSimulator] Scenario: {n} losses | "
            f"Impact: {impact_pct:.1f}% | "
            f"Survival: {'YES' if survival else 'NO'}"
        )

        return result

    def consecutive_wins(self, n: int = 5) -> dict:
        """
        Simulate N consecutive wins.

        Example:
            n=5, risk=1%, rr=2.0, balance=$10,000
            Each win = $200 (1% risk * 2.0 RR)
            After 5 wins: $11,041
        """
        balance = self.balance
        risk_per_trade = self.risk_pct / 100

        wins = []
        for i in range(n):
            gain = balance * risk_per_trade * self.avg_rr
            balance += gain
            wins.append({
                "trade": i + 1,
                "gain_usd": round(gain, 2),
                "balance_after": round(balance, 2),
                "gain_pct": round(
                    (balance - self.balance) / self.balance * 100, 2
                ),
            })

        total_gain = balance - self.balance
        gain_pct = total_gain / self.balance * 100

        result = {
            "scenario": f"{n} consecutive wins",
            "n_wins": n,
            "starting_balance": round(self.balance, 2),
            "ending_balance": round(balance, 2),
            "total_gain_usd": round(total_gain, 2),
            "gain_pct": round(gain_pct, 2),
            "trade_by_trade": wins,
        }

        return result

    # ═══════════════════════════════════════════════════════
    # EXTREME SCENARIOS
    # ═══════════════════════════════════════════════════════

    def worst_day(self, n_trades: int = 5, all_losses: bool = True) -> dict:
        """
        Simulate worst possible trading day.
        """
        if all_losses:
            return self.consecutive_losses(n_trades)

        # Worst day with mixed results (80% losses)
        balance = self.balance
        risk_per_trade = self.risk_pct / 100
        trades = []

        for i in range(n_trades):
            if i < int(n_trades * 0.8):  # 80% losses
                loss = balance * risk_per_trade
                balance -= loss
                trades.append({"trade": i+1, "result": "LOSS", "pnl": -round(loss, 2)})
            else:
                gain = balance * risk_per_trade * self.avg_rr * 0.5
                balance += gain
                trades.append({"trade": i+1, "result": "WIN", "pnl": round(gain, 2)})

        total_pnl = balance - self.balance
        return {
            "scenario": f"Worst day ({n_trades} trades, 80% losses)",
            "total_pnl_usd": round(total_pnl, 2),
            "impact_pct": round(total_pnl / self.balance * 100, 2),
            "ending_balance": round(balance, 2),
            "trades": trades,
        }

    def best_day(self, n_trades: int = 5) -> dict:
        """Simulate best possible trading day."""
        balance = self.balance
        risk_per_trade = self.risk_pct / 100
        trades = []

        for i in range(n_trades):
            gain = balance * risk_per_trade * self.avg_rr
            balance += gain
            trades.append({"trade": i+1, "result": "WIN", "pnl": round(gain, 2)})

        total_pnl = balance - self.balance
        return {
            "scenario": f"Best day ({n_trades} trades, all wins)",
            "total_pnl_usd": round(total_pnl, 2),
            "gain_pct": round(total_pnl / self.balance * 100, 2),
            "ending_balance": round(balance, 2),
            "trades": trades,
        }

    def worst_week(
        self,
        trades_per_day: int = 3,
        losing_days: int = 4,
    ) -> dict:
        """
        Simulate worst possible trading week.

        Example: 4 losing days with 3 trades each, 1 breakeven day.
        """
        balance = self.balance
        risk_per_trade = self.risk_pct / 100
        day_results = []

        for day in range(5):
            if day < losing_days:
                # Losing day
                day_loss = 0
                for _ in range(trades_per_day):
                    loss = balance * risk_per_trade
                    balance -= loss
                    day_loss += loss
                day_results.append({
                    "day": day + 1, "type": "LOSING",
                    "pnl": -round(day_loss, 2),
                    "balance": round(balance, 2),
                })
            else:
                # Breakeven day
                day_results.append({
                    "day": day + 1, "type": "BREAKEVEN",
                    "pnl": 0, "balance": round(balance, 2),
                })

        total_pnl = balance - self.balance
        return {
            "scenario": f"Worst week ({losing_days}/5 losing days, {trades_per_day} trades/day)",
            "total_pnl_usd": round(total_pnl, 2),
            "impact_pct": round(total_pnl / self.balance * 100, 2),
            "ending_balance": round(balance, 2),
            "daily_results": day_results,
        }

    # ═══════════════════════════════════════════════════════
    # BLACK SWAN & STRATEGY FAILURE
    # ═══════════════════════════════════════════════════════

    def black_swan(self, move_pct: float = 5.0, positions: int = 2) -> dict:
        """
        Simulate a black swan event (extreme market move).

        Example: Market suddenly moves 5% against you while
        holding 2 positions with 1% risk each.

        In a real black swan, SL may not trigger at expected level.
        We simulate 2x-5x normal loss.
        """
        balance = self.balance
        normal_risk = self.risk_pct / 100

        # Black swan: SL fails, loss is multiplied
        sl_failure_multiplier = 3  # lose 3x expected
        total_loss = 0

        position_results = []
        for i in range(positions):
            normal_loss = balance * normal_risk
            actual_loss = normal_loss * sl_failure_multiplier
            balance -= actual_loss
            balance = max(0, balance)
            total_loss += actual_loss

            position_results.append({
                "position": i + 1,
                "normal_loss": round(normal_loss, 2),
                "actual_loss": round(actual_loss, 2),
                "sl_failure_multiplier": sl_failure_multiplier,
            })

        impact_pct = total_loss / self.balance * 100
        survival = balance > self.initial_balance * 0.3  # 30% survival threshold

        return {
            "scenario": f"Black Swan ({move_pct}% move, {positions} positions)",
            "market_move_pct": move_pct,
            "sl_failure_multiplier": sl_failure_multiplier,
            "starting_balance": round(self.balance, 2),
            "ending_balance": round(balance, 2),
            "total_loss_usd": round(total_loss, 2),
            "impact_pct": round(impact_pct, 2),
            "survival": survival,
            "position_results": position_results,
            "recovery_trades_needed": round(
                total_loss / (self.balance * normal_risk * self.avg_rr), 1
            ) if balance > 0 else float("inf"),
        }

    def strategy_failure(
        self,
        current_wr: float = 0.55,
        new_wr: float = 0.35,
        n_trades: int = 50,
    ) -> dict:
        """
        Simulate strategy failure (win rate drops suddenly).

        This is what happens when a strategy that worked in backtesting
        suddenly stops working in live markets.
        """
        balance = self.balance
        risk_per_trade = self.risk_pct / 100

        # Phase 1: Normal performance
        for _ in range(n_trades // 2):
            if __import__("random").random() < current_wr:
                balance += balance * risk_per_trade * self.avg_rr
            else:
                balance -= balance * risk_per_trade

        balance_after_normal = balance

        # Phase 2: Degraded performance
        for _ in range(n_trades - n_trades // 2):
            if __import__("random").random() < new_wr:
                balance += balance * risk_per_trade * self.avg_rr
            else:
                balance -= balance * risk_per_trade

        normal_pnl = balance_after_normal - self.balance
        failure_pnl = balance - balance_after_normal
        total_pnl = balance - self.balance

        return {
            "scenario": (
                f"Strategy failure: WR drops from "
                f"{current_wr*100:.0f}% to {new_wr*100:.0f}%"
            ),
            "phase1_wr": current_wr,
            "phase2_wr": new_wr,
            "balance_after_normal": round(balance_after_normal, 2),
            "ending_balance": round(balance, 2),
            "normal_phase_pnl": round(normal_pnl, 2),
            "failure_phase_pnl": round(failure_pnl, 2),
            "total_pnl": round(total_pnl, 2),
            "total_impact_pct": round(total_pnl / self.balance * 100, 2),
        }

    # ═══════════════════════════════════════════════════════
    # COMPREHENSIVE REPORT
    # ═══════════════════════════════════════════════════════

    def run_all_scenarios(self) -> dict:
        """
        Run all scenarios and produce a comprehensive risk report.
        """
        results = {
            "timestamp": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ).isoformat(),
            "balance": self.balance,
            "risk_pct": self.risk_pct,
            "avg_rr": self.avg_rr,
            "scenarios": {
                "5_consecutive_losses": self.consecutive_losses(5),
                "10_consecutive_losses": self.consecutive_losses(10),
                "5_consecutive_wins": self.consecutive_wins(5),
                "worst_day": self.worst_day(),
                "best_day": self.best_day(),
                "worst_week": self.worst_week(),
                "black_swan": self.black_swan(),
            },
        }

        # Summary assessment
        losses_5 = results["scenarios"]["5_consecutive_losses"]
        losses_10 = results["scenarios"]["10_consecutive_losses"]
        black_swan = results["scenarios"]["black_swan"]

        overall_risk = self._assess_overall_risk_from_scenarios(
            losses_5, losses_10, black_swan
        )

        results["summary"] = {
            "can_survive_5_losses": losses_5["survival"],
            "can_survive_10_losses": losses_10["survival"],
            "can_survive_black_swan": black_swan["survival"],
            "impact_5_losses_pct": losses_5["impact_pct"],
            "impact_10_losses_pct": losses_10["impact_pct"],
            "impact_black_swan_pct": black_swan["impact_pct"],
            "overall_risk_level": overall_risk,
        }

        return results

    def _assess_overall_risk_from_scenarios(
        self, losses_5: dict, losses_10: dict, black_swan: dict
    ) -> str:
        """Assess overall risk level from individual scenario results."""
        if not losses_5["survival"]:
            return "CRITICAL — Cannot survive 5 consecutive losses"
        elif not losses_10["survival"]:
            return "HIGH — Cannot survive 10 consecutive losses"
        elif not black_swan["survival"]:
            return "HIGH — Cannot survive black swan event"
        elif losses_5["impact_pct"] > 10:
            return "ELEVATED — 5 losses would cause >10% drawdown"
        elif losses_5["impact_pct"] > 5:
            return "MODERATE — 5 losses would cause >5% drawdown"
        else:
            return "LOW — Account well protected against scenarios"

    def print_all_scenarios(self) -> None:
        """Print all scenario results."""
        results = self.run_all_scenarios()
        bar = "=" * 52
        print(f"\n{bar}")
        print("  RISK SCENARIO SIMULATION REPORT")
        print(bar)
        print(f"  Balance: ${self.balance:,.2f} | Risk: {self.risk_pct}% | RR: 1:{self.avg_rr}")
        print()

        for name, result in results["scenarios"].items():
            if "impact_pct" in result:
                print(f"  {name:.<30} {result['impact_pct']:+.1f}%")
            elif "gain_pct" in result:
                print(f"  {name:.<30} {result['gain_pct']:+.1f}%")

        print()
        summary = results["summary"]
        print(f"  Overall Risk Level: {summary['overall_risk_level']}")
        print(f"  Survive 5 losses  : {'YES' if summary['can_survive_5_losses'] else 'NO'}")
        print(f"  Survive 10 losses : {'YES' if summary['can_survive_10_losses'] else 'NO'}")
        print(f"  Survive black swan: {'YES' if summary['can_survive_black_swan'] else 'NO'}")
        print(bar + "\n")
