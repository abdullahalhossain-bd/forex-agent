# research/research_agent.py — Day 57 | Autonomous Research Agent
# ============================================================
# AI Trader-এর Research Department — Master Orchestrator.
#
# Architecture:
#   Market Data → Autonomous Research Agent → Research Results → Knowledge Base
#
# Sub-components:
#   StrategyGenerator — discovers new strategies & mutations
#   HypothesisEngine — creates & evaluates testable hypotheses
#   ExperimentRunner — runs backtests and evaluates results
#   ResearchReportGenerator — produces weekly research reports
#
# Research Loop:
#   Observe Market → Find Problem → Create Hypothesis → Build Strategy
#   → Backtest → Evaluate → Store Knowledge → Improve
# ============================================================

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from utils.logger import get_logger
from research.strategy_generator import StrategyGenerator
from research.hypothesis_engine import HypothesisEngine, Hypothesis
from research.experiment_runner import ExperimentRunner, Experiment
from research.research_report import ResearchReportGenerator

log = get_logger("research.agent")

# ── Research State Path ─────────────────────────────────────────
from core.constants import MEMORY_DIR
RESEARCH_STATE_PATH = MEMORY_DIR / "research_state.json"
RESEARCH_KNOWLEDGE_PATH = MEMORY_DIR / "research_knowledge.json"


class ResearchAgent:
    """
    Autonomous Research Agent — AI Trader-এর Research Department.

    এটা সবচেয়ে গুরুত্বপূর্ণ Day 57 অংশ।
    AI নিজে নতুন trading idea খুঁজবে, পরীক্ষা করবে,
    এবং প্রমাণিত হলে নিজের knowledge base-এ যুক্ত করবে।

    Usage:
        agent = ResearchAgent()
        agent.run_research_cycle()
        report = agent.generate_weekly_report()
        agent.print_status()
    """

    def __init__(self, enable_auto_research: bool = True):
        self.generator = StrategyGenerator()
        self.hypothesis_engine = HypothesisEngine()
        self.experiment_runner = ExperimentRunner()
        self.report_generator = ResearchReportGenerator()
        self.enable_auto_research = enable_auto_research

        # State tracking
        self._state = self._load_state()
        self._knowledge = self._load_knowledge()

        log.info(
            f"[ResearchAgent] Initialized | "
            f"Auto research: {'ON' if enable_auto_research else 'OFF'} | "
            f"Total experiments: {self.experiment_runner.get_stats()['total_experiments']} | "
            f"Active strategies: {len(self.generator.get_active_strategies())}"
        )

    # ═══════════════════════════════════════════════════════
    # MAIN RESEARCH CYCLE
    # ═══════════════════════════════════════════════════════

    def run_research_cycle(
        self,
        n_experiments: int = 3,
        pair: str = "EURUSD",
        timeframe: str = "H1",
    ) -> dict:
        """
        Execute a complete research cycle.

        Steps:
          1. Observe current market (analyze patterns)
          2. Generate hypotheses
          3. Create strategies (novel + mutations)
          4. Run experiments (backtest)
          5. Evaluate results
          6. Approve/reject strategies
          7. Store knowledge
          8. Update state

        Args:
            n_experiments: Number of experiments to run
            pair: Primary currency pair
            timeframe: Primary timeframe

        Returns:
            Complete research cycle results dict.
        """
        if not self.enable_auto_research:
            log.info("[ResearchAgent] Auto research disabled — skipping cycle")
            return {"status": "disabled", "experiments": []}

        cycle_id = f"CYCLE-{self._state.get('cycle_count', 0) + 1}"
        started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        t0 = datetime.now(timezone.utc)

        log.info(f"[ResearchAgent] Starting {cycle_id} — {n_experiments} experiments on {pair} {timeframe}")

        # Step 1: Generate hypotheses
        log.info(f"[{cycle_id}] Step 1/6: Generating hypotheses...")
        hypotheses = self.hypothesis_engine.generate_batch(n=n_experiments, strategy_name=pair)

        # Step 2: Generate strategies
        log.info(f"[{cycle_id}] Step 2/6: Generating strategies...")
        strategies = []
        for hyp in hypotheses:
            strategy = self.generator.generate_novel(pair=pair, timeframe=timeframe)
            strategies.append((hyp, strategy))

        # Also add mutations of existing strategies
        active = self.generator.get_active_strategies()
        if active:
            base = random.choice(active) if hasattr(__import__('random'), 'choice') else active[0]
            mutation = self.generator.generate_mutation(base["name"], "medium")
            strategies.append((hypotheses[0], mutation))

        # Step 3: Run experiments
        log.info(f"[{cycle_id}] Step 3/6: Running {len(strategies)} experiments...")
        experiment_results = []
        approved_names = []
        rejected_names = []

        for i, (hypothesis, strategy) in enumerate(strategies[:n_experiments]):
            log.info(f"[{cycle_id}] Experiment {i+1}/{len(strategies)}: {strategy.get('name')}")
            try:
                experiment = self.experiment_runner.create_experiment(
                    hypothesis=hypothesis,
                    strategy=strategy,
                    pair=pair,
                    timeframe=timeframe,
                )
                result = self.experiment_runner.run_experiment(experiment)
                experiment_results.append(result)

                if result.get("status") == "APPROVED":
                    approved_names.append(strategy.get("name"))
                    self._add_knowledge(
                        topic=f"strategy_approved:{strategy.get('name')}",
                        content=f"Strategy '{strategy.get('name')}' approved via backtest on {pair} {timeframe}. "
                                f"Trades: {result.get('backtest_result', {}).get('summary', {}).get('trades', 0)}. "
                                f"Win rate: {result.get('backtest_result', {}).get('summary', {}).get('win_rate', 0)}%.",
                        metadata={"type": "approved_strategy", "pair": pair, "timeframe": timeframe},
                    )
                else:
                    rejected_names.append(strategy.get("name"))

            except Exception as e:
                log.error(f"[{cycle_id}] Experiment failed: {e}", exc_info=True)

        # Step 4: Evaluate hypotheses
        log.info(f"[{cycle_id}] Step 4/6: Evaluating hypotheses...")
        hypothesis_results = self._evaluate_hypotheses(hypotheses, experiment_results)

        # Step 5: Analyze market behavior
        log.info(f"[{cycle_id}] Step 5/6: Analyzing market behavior...")
        market_findings = self._analyze_market_behavior(pair, timeframe)

        # Step 6: Store knowledge
        log.info(f"[{cycle_id}] Step 6/6: Storing research knowledge...")
        self._store_research_knowledge(experiment_results, hypothesis_results, market_findings)

        # Update state
        elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
        self._state["cycle_count"] = self._state.get("cycle_count", 0) + 1
        self._state["last_cycle_at"] = started_at
        self._state["total_experiments_run"] = self._state.get("total_experiments_run", 0) + len(experiment_results)
        self._state["total_strategies_approved"] = self._state.get("total_strategies_approved", 0) + len(approved_names)
        self._save_state()

        # Store in knowledge memory (ChromaDB) if available
        self._store_in_vector_memory(experiment_results, approved_names)

        completed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        log.info(
            f"[ResearchAgent] {cycle_id} complete in {elapsed:.1f}s — "
            f"{len(experiment_results)} experiments, "
            f"{len(approved_names)} approved, "
            f"{len(rejected_names)} rejected"
        )

        return {
            "cycle_id": cycle_id,
            "started_at": started_at,
            "completed_at": completed_at,
            "duration_seconds": round(elapsed, 1),
            "pair": pair,
            "timeframe": timeframe,
            "experiments": experiment_results,
            "hypotheses": [h.to_dict() for h in hypotheses],
            "hypothesis_results": hypothesis_results,
            "market_findings": market_findings,
            "strategies_approved": approved_names,
            "strategies_rejected": rejected_names,
            "stats": {
                "experiment_stats": self.experiment_runner.get_stats(),
                "hypothesis_stats": self.hypothesis_engine.get_stats(),
                "strategy_stats": {
                    "active": len(self.generator.get_active_strategies()),
                    "testing": len(self.generator.get_testing_strategies()),
                    "rejected": len(self.generator.get_rejected_strategies()),
                },
            },
        }

    # ═══════════════════════════════════════════════════════
    # MARKET BEHAVIOR RESEARCH
    # ═══════════════════════════════════════════════════════

    def analyze_market(self, pair: str, timeframe: str = "H1") -> dict:
        """
        Analyze current market behavior for a pair.

        Returns market behavior research findings that feed into
        hypothesis generation and strategy creation.
        """
        log.info(f"[ResearchAgent] Analyzing market behavior: {pair} {timeframe}")

        try:
            from data.fetcher import DataFetcher
            from data.indicators import Indicators
            import numpy as np

            fetcher = DataFetcher()
            df = fetcher.fetch_ohlcv(pair, timeframe.lower(), limit=500)

            if df is None or len(df) < 100:
                return {"error": "insufficient_data"}

            ind = Indicators()
            df = ind.add_all(df)

            # Calculate session-based analysis
            session_analysis = self._session_analysis(df)
            volatility_analysis = self._volatility_analysis(df)
            trend_analysis = self._trend_analysis(df)

            finding = {
                "pair": pair,
                "timeframe": timeframe,
                "session_analysis": session_analysis,
                "volatility_analysis": volatility_analysis,
                "trend_analysis": trend_analysis,
                "analyzed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "recommendation": self._market_recommendation(session_analysis, volatility_analysis),
            }

            log.info(f"[ResearchAgent] Market analysis complete for {pair}")
            return finding

        except Exception as e:
            log.error(f"[ResearchAgent] Market analysis failed: {e}")
            return {"error": str(e)}

    def generate_hypothesis_from_market(self, pair: str, timeframe: str = "H1") -> list[Hypothesis]:
        """Generate hypotheses based on current market analysis."""
        market_data = self.analyze_market(pair, timeframe)
        if "error" in market_data:
            return self.hypothesis_engine.generate_batch(n=2)

        hypotheses = self.hypothesis_engine.generate_batch(n=3, strategy_name=pair)

        # Add market-observation-based hypothesis
        if market_data.get("recommendation"):
            obs_hypothesis = self.hypothesis_engine.generate_from_market_observation(
                {
                    "pair": pair,
                    "observation": market_data["recommendation"],
                    "evidence": {
                        "session": market_data.get("session_analysis", {}),
                        "volatility": market_data.get("volatility_analysis", {}),
                    },
                }
            )
            hypotheses.append(obs_hypothesis)

        return hypotheses

    # ═══════════════════════════════════════════════════════
    # TEST & EVALUATE
    # ═══════════════════════════════════════════════════════

    def test_strategy(self, strategy: dict, pair: str = "EURUSD", timeframe: str = "H1") -> dict:
        """Test a single strategy through backtest."""
        hypothesis = self.hypothesis_engine.generate()
        experiment = self.experiment_runner.create_experiment(
            hypothesis=hypothesis,
            strategy=strategy,
            pair=pair,
            timeframe=timeframe,
        )
        return self.experiment_runner.run_experiment(experiment)

    def evaluate_result(self, experiment_result: dict) -> dict:
        """Evaluate an experiment result and provide recommendations."""
        return self.report_generator.generate_single_experiment_report(experiment_result)

    # ═══════════════════════════════════════════════════════
    # REPORT GENERATION
    # ═══════════════════════════════════════════════════════

    def generate_weekly_report(self) -> dict:
        """Generate a comprehensive weekly research report."""
        all_experiments = self.experiment_runner.get_all_experiments()
        all_hypotheses = self.hypothesis_engine.get_history()
        active = self.generator.get_active_strategies()
        rejected = self.generator.get_rejected_strategies()

        report = self.report_generator.generate_weekly(
            experiment_results=all_experiments[-20:],  # Last 20 experiments
            hypothesis_results=[h for h in all_hypotheses if h.get("result")],
            strategies_approved=[s["name"] for s in active],
            strategies_rejected=[s.get("name", "") for s in rejected],
        )

        # Save report
        self.report_generator.save_report(report)
        log.info("[ResearchAgent] Weekly research report generated and saved")
        return report

    # ═══════════════════════════════════════════════════════
    # STRATEGY MUTATION ENGINE
    # ═══════════════════════════════════════════════════════

    def mutate_best_strategy(self, strength: str = "medium") -> dict:
        """Mutate the best performing active strategy."""
        active = self.generator.get_active_strategies()
        if not active:
            log.warning("[ResearchAgent] No active strategies to mutate")
            return {"error": "no_active_strategies"}

        best = max(
            active,
            key=lambda s: s.get("approval_backtest", {}).get("metrics", {}).get("profit_factor", 0)
        )
        mutation = self.generator.generate_mutation(best["name"], strength)
        return self.test_strategy(mutation)

    # ═══════════════════════════════════════════════════════
    # STATUS & STATS
    # ═══════════════════════════════════════════════════════

    def print_status(self) -> None:
        """Print current research agent status."""
        stats = self.get_stats()
        active = self.generator.get_active_strategies()
        testing = self.generator.get_testing_strategies()

        bar = "=" * 56
        sep = "-" * 56

        print()
        print(bar)
        print("  RESEARCH AGENT STATUS (Day 57)")
        print(bar)
        print(f"  Status           : {'ACTIVE' if self.enable_auto_research else 'DISABLED'}")
        print(f"  Research Cycles  : {stats.get('cycle_count', 0)}")
        print(f"  Total Experiments: {stats.get('total_experiments', 0)}")
        print(f"  Approved         : {stats.get('total_approved', 0)}")
        print(sep)
        print(f"  Active Strategies  : {len(active)}")
        print(f"  Testing Strategies : {len(testing)}")
        print(f"  Hypothesis Stats   : {stats.get('hypothesis_stats', {})}")
        print(sep)

        if active:
            print("  Active Strategies:")
            for s in active[:5]:
                print(f"    + {s.get('name', 'unknown')}")

        if testing:
            print("  Testing:")
            for s in testing[:3]:
                print(f"    ~ {s.get('name', 'unknown')}")

        print(bar)
        print()

    def get_stats(self) -> dict:
        """Return comprehensive research agent statistics."""
        return {
            "enabled": self.enable_auto_research,
            "cycle_count": self._state.get("cycle_count", 0),
            "total_experiments": self._state.get("total_experiments_run", 0),
            "total_approved": self._state.get("total_strategies_approved", 0),
            "last_cycle_at": self._state.get("last_cycle_at"),
            "experiment_stats": self.experiment_runner.get_stats(),
            "hypothesis_stats": self.hypothesis_engine.get_stats(),
            "active_strategies": len(self.generator.get_active_strategies()),
            "testing_strategies": len(self.generator.get_testing_strategies()),
            "rejected_strategies": len(self.generator.get_rejected_strategies()),
            "knowledge_entries": len(self._knowledge),
            "failed_archive": len(self.experiment_runner.get_failed_archive()),
        }

    # ═══════════════════════════════════════════════════════
    # KNOWLEDGE MANAGEMENT
    # ═══════════════════════════════════════════════════════

    def query_knowledge(self, topic: str = None) -> list[dict]:
        """Query research knowledge base."""
        if topic:
            return [k for k in self._knowledge if topic.lower() in k.get("topic", "").lower()]
        return list(self._knowledge)

    def get_research_memory_graph(self) -> dict:
        """
        Research Memory Graph — relationships between patterns,
        market regimes, and results.
        """
        graph = {
            "nodes": [],
            "edges": [],
        }

        # Add experiment nodes
        for exp in self.experiment_runner.get_all_experiments():
            node_id = f"exp_{exp.get('id', '')}"
            graph["nodes"].append({
                "id": node_id,
                "type": "experiment",
                "label": exp.get("strategy_name", "unknown"),
                "status": exp.get("status", "unknown"),
                "pair": exp.get("pair"),
                "timeframe": exp.get("timeframe"),
            })

        # Add knowledge relationship edges
        for i, k in enumerate(self._knowledge):
            graph["edges"].append({
                "source": k.get("topic", ""),
                "target": k.get("metadata", {}).get("type", ""),
                "relationship": k.get("content", "")[:50],
            })

        return graph

    # ═══════════════════════════════════════════════════════
    # INTERNAL HELPERS
    # ═══════════════════════════════════════════════════════

    def _evaluate_hypotheses(
        self, hypotheses: list[Hypothesis], experiment_results: list[dict]
    ) -> list[dict]:
        """Evaluate hypotheses against experiment results."""
        results = []
        for i, hyp in enumerate(hypotheses):
            if i < len(experiment_results):
                ctrl_result = {"summary": {"win_rate": 50, "profit_factor": 1.0, "max_drawdown": 10}}
                treat_result = experiment_results[i].get("backtest_result", ctrl_result)
                result = self.hypothesis_engine.evaluate_hypothesis(hyp, ctrl_result, treat_result)
                results.append(result)
        return results

    def _analyze_market_behavior(self, pair: str, timeframe: str) -> list[dict]:
        """Analyze market behavior and return findings."""
        findings = []
        try:
            analysis = self.analyze_market(pair, timeframe)
            if "error" not in analysis:
                finding = self.hypothesis_engine.analyze_market_behavior(
                    pair,
                    {
                        "session": "auto",
                        "avg_ATR": analysis.get("volatility_analysis", {}).get("avg_atr_pct", 0),
                        "win_rate": 50,  # Will be filled from actual data
                        "avg_spread": 1.5,
                        "session_movement": analysis.get("trend_analysis", {}).get("trend_strength", 0),
                    },
                )
                findings.append(finding)
        except Exception as e:
            log.warning(f"[ResearchAgent] Market behavior analysis failed: {e}")
        return findings

    def _session_analysis(self, df) -> dict:
        """Analyze session-based performance patterns."""
        import pandas as pd

        try:
            if not isinstance(df.index, pd.DatetimeIndex):
                return {"available": False}

            df["hour"] = df.index.hour
            sessions = {
                "tokyo": (0, 8),
                "london": (7, 16),
                "ny": (13, 22),
                "overlap": (13, 16),
            }

            result = {}
            for name, (start, end) in sessions.items():
                session_df = df[(df["hour"] >= start) & (df["hour"] < end)]
                if len(session_df) > 10:
                    avg_range = (session_df["high"] - session_df["low"]).mean()
                    avg_atr = session_df["atr"].mean() if "atr" in session_df.columns else 0
                    result[name] = {
                        "candles": len(session_df),
                        "avg_range": round(float(avg_range), 6),
                        "avg_atr": round(float(avg_atr), 6),
                        "pct_of_total": round(len(session_df) / len(df) * 100, 1),
                    }
            return result
        except Exception:
            return {"available": False}

    def _volatility_analysis(self, df) -> dict:
        """Analyze volatility patterns."""
        try:
            atr = df["atr"].dropna()
            return {
                "avg_atr": round(float(atr.mean()), 6),
                "std_atr": round(float(atr.std()), 6),
                "current_atr": round(float(atr.iloc[-1]), 6),
                "avg_atr_pct": round(float(atr.iloc[-1] / df["close"].iloc[-1] * 100), 3) if df["close"].iloc[-1] > 0 else 0,
                "regime": "high" if float(atr.iloc[-1]) > float(atr.mean()) + float(atr.std()) else (
                    "low" if float(atr.iloc[-1]) < float(atr.mean()) - float(atr.std()) else "normal"
                ),
            }
        except Exception:
            return {"available": False}

    def _trend_analysis(self, df) -> dict:
        """Analyze trend patterns."""
        try:
            ema9 = df["ema_9"].iloc[-1] if "ema_9" in df.columns else 0
            ema21 = df["ema_21"].iloc[-1] if "ema_21" in df.columns else 0
            sma50 = df["sma_50"].iloc[-1] if "sma_50" in df.columns else 0
            sma200 = df["sma_200"].iloc[-1] if "sma_200" in df.columns else 0
            close = df["close"].iloc[-1]

            if close > ema9 > ema21 > sma50:
                trend = "strong_bullish"
            elif close > ema9 > ema21:
                trend = "bullish"
            elif close < ema9 < ema21 < sma50:
                trend = "strong_bearish"
            elif close < ema9 < ema21:
                trend = "bearish"
            else:
                trend = "sideways"

            return {
                "trend": trend,
                "trend_strength": round(abs(float(ema9 - ema21) / float(close) * 10000), 1),
                "above_sma200": bool(close > sma200),
            }
        except Exception:
            return {"available": False}

    def _market_recommendation(self, session_analysis: dict, volatility_analysis: dict) -> str:
        """Generate market recommendation based on analysis."""
        regime = volatility_analysis.get("regime", "normal")
        if regime == "high":
            return "High volatility detected — widen stops or wait for calmer conditions."
        elif regime == "low":
            return "Low volatility — tight ranges may not justify the spread cost."
        else:
            return "Normal market conditions — standard strategy parameters apply."

    def _store_research_knowledge(
        self,
        experiments: list[dict],
        hypotheses: list[dict],
        findings: list[dict],
    ) -> None:
        """Store research results in knowledge base."""
        for exp in experiments:
            self._add_knowledge(
                topic=f"experiment:{exp.get('id', '')}",
                content=f"Experiment {exp.get('id')}: {exp.get('strategy_name', '')} "
                        f"on {exp.get('pair')} {exp.get('timeframe')} — "
                        f"Result: {exp.get('status')}",
                metadata={"type": "experiment", "status": exp.get("status")},
            )

        for finding in findings:
            self._add_knowledge(
                topic=f"market:{finding.get('pair', '')}",
                content=finding.get("finding", ""),
                metadata={"type": "market_finding", "pair": finding.get("pair")},
            )

    def _store_in_vector_memory(self, experiments: list[dict], approved_names: list[str]) -> None:
        """Store research results in vector memory (ChromaDB) for future retrieval."""
        if not approved_names:
            return

        try:
            from memory.knowledge_store import KnowledgeStore

            store = KnowledgeStore()
            for name in approved_names:
                store.add_memory(
                    f"Research approved strategy: {name}. "
                    f"This strategy passed all backtest criteria and is ready for paper trading.",
                    metadata={"type": "research_approved", "strategy": name},
                )
            log.info(f"[ResearchAgent] Stored {len(approved_names)} entries in vector memory")
        except Exception as e:
            log.debug(f"[ResearchAgent] Vector memory storage skipped: {e}")

    def _add_knowledge(self, topic: str, content: str, metadata: dict = None) -> None:
        """Add an entry to the research knowledge base."""
        entry = {
            "topic": topic,
            "content": content,
            "metadata": metadata or {},
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self._knowledge.append(entry)
        self._save_knowledge()

    # ═══════════════════════════════════════════════════════
    # PERSISTENCE
    # ═══════════════════════════════════════════════════════

    def _load_state(self) -> dict:
        if RESEARCH_STATE_PATH.exists():
            try:
                with open(RESEARCH_STATE_PATH) as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "cycle_count": 0,
            "total_experiments_run": 0,
            "total_strategies_approved": 0,
            "last_cycle_at": None,
        }

    def _save_state(self) -> None:
        RESEARCH_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RESEARCH_STATE_PATH, "w") as f:
            json.dump(self._state, f, indent=2)

    def _load_knowledge(self) -> list:
        if RESEARCH_KNOWLEDGE_PATH.exists():
            try:
                with open(RESEARCH_KNOWLEDGE_PATH) as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def _save_knowledge(self) -> None:
        RESEARCH_KNOWLEDGE_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Keep last 500 entries
        with open(RESEARCH_KNOWLEDGE_PATH, "w") as f:
            json.dump(self._knowledge[-500:], f, indent=2)


# ── Import random at module level for strategy mutation ──────────
import random
