# memory/trade_memory.py  —  Day 33 | Full Memory Bridge & Self-Learning Vector Layer
# ====================================================================================
# এই মডিউলে Day 16-এর Database/Pattern ট্র্যাকিং এবং Day 33-এর Local Vector Memory
# (sentence-transformers) একসাথে মিক্স করা হয়েছে যাতে কোনো ডুপ্লিকেট ক্লাস এরর না আসে।
#
# এটি যা করে:
#   - প্রতিটা closed trade-কে তার entry context (RSI, trend, regime, pattern) +
#     outcome (WIN/LOSS, pnl) সহ ডেটাবেজে এবং ভেক্টর হিসেবে .npy ফাইলে স্টোর করে।
#   - নতুন setup আসলে similar past trades রিট্রিভ করে টেক্সট আকারে AIAnalyst-এ পাঠায়।
# ====================================================================================

import os
import json
import numpy as np
from datetime import datetime
from utils.logger import get_logger
from memory.database import Database
from memory.pattern_memory import PatternMemory

log = get_logger("trade_memory")

MEMORY_DIR = "memory/trade_vectors"
os.makedirs(MEMORY_DIR, exist_ok=True)
VECTORS_PATH = os.path.join(MEMORY_DIR, "vectors.npy")
METADATA_PATH = os.path.join(MEMORY_DIR, "metadata.json")

# ── Embedding model setup (graceful fallback) ──────────────────────────────────
EMBEDDINGS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer as _ST

    def _load_embedding_model(model_name: str):
        """
        Model load করো।
        - আগে local cache চেক করে (offline-safe)
        - না পেলে download করার চেষ্টা করে
        - সব fail হলে None return করে (system চলতে থাকে)
        """
        # Local cache path (sentence-transformers default cache)
        cache_dir = os.path.join(
            os.path.expanduser("~"),
            ".cache", "torch", "sentence_transformers",
            model_name.replace("/", "_"),
        )
        try:
            if os.path.isdir(cache_dir):
                log.info(f"[TradeMemory] Loading embedding model from cache: {cache_dir}")
                return _ST(cache_dir)
            else:
                log.info(f"[TradeMemory] Downloading embedding model: {model_name} (~80MB, once only)")
                return _ST(model_name)
        except Exception as e:
            log.warning(
                f"[TradeMemory] Embedding model load failed: {e}\n"
                f"  Vector memory disabled — trading continues normally.\n"
                f"  To enable: pip install sentence-transformers  then re-run."
            )
            return None

    EMBEDDINGS_AVAILABLE = True

except ImportError:
    log.warning(
        "[TradeMemory] sentence-transformers not installed — vector memory disabled.\n"
        "  To enable: pip install sentence-transformers"
    )

    def _load_embedding_model(model_name: str):
        return None


class TradeMemory:
    """
    SQL Database + Closed trade lessons-এর local vector memory (Day 16 & Day 33 Combined).
    """

    MODEL_NAME = "all-MiniLM-L6-v2"
    TOP_K = 3

    def __init__(self, seed_rules: bool = False):
        # Day 16 Storage Core
        self.db      = Database()
        self.pattern = PatternMemory()

        # Day 33 Vector Engine Initialization
        self._model = None
        self._vectors: np.ndarray | None = None
        self._metadata: list[dict] = []

        # Day 81+ — Use shared model cache so the SentenceTransformer
        # downloads exactly once across TradeMemory + KnowledgeStore +
        # MistakeAnalyzer. Saves ~6s per duplicate boot.
        try:
            from memory.sentence_model_cache import get_sentence_model
            self._model = get_sentence_model()
        except Exception:
            self._model = _load_embedding_model(self.MODEL_NAME)  # legacy fallback
        if self._model is not None:
            log.info(f"[TradeMemory] Embedding model ready: {self.MODEL_NAME}")
        else:
            log.warning("[TradeMemory] Running without vector memory (embedding model unavailable)")

        self._load_vector_data()

        # Day 37 fix: previously the `seed_rules` argument was silently ignored.
        # Now we honor it by seeding the KnowledgeStore with canonical trading
        # rules exactly once (idempotent — KnowledgeStore.seed_trading_rules
        # internally checks if already seeded and skips).
        if seed_rules:
            try:
                from memory.knowledge_store import KnowledgeStore
                ks = KnowledgeStore()
                if hasattr(ks, "seed_trading_rules"):
                    ks.seed_trading_rules()
                    log.info("[TradeMemory] Trading rules seeding completed")
            except Exception as e:
                log.warning(f"[TradeMemory] Rule seeding skipped (non-critical): {e}")

    def _has_model(self) -> bool:
        """Embedding model available কিনা চেক করো।"""
        return self._model is not None

    # ── Called from trader.py / signal_pipeline.py ─────────────────────────────

    def on_signal_generated(self, result: dict, market_out: dict, analysis_out: dict):
        final_action = result.get("final_action", "NO TRADE")

        ind     = market_out.get("ind_ctx", {})
        pattern = "none"
        df      = market_out.get("df")
        if df is not None and "pattern_name" in df.columns:
            pattern = df.iloc[-1].get("pattern_name", "none")

        self.db.save_analysis({
            "pair":        result.get("symbol"),
            "timeframe":   result.get("timeframe"),
            "rsi":         result.get("rsi"),
            "macd":        ind.get("macd"),
            "trend":       result.get("trend"),
            "regime":      result.get("regime"),
            "pattern":     pattern,
            "sr_location": analysis_out.get("sr_result", {}).get("location", "unknown"),
            "mtf_bias":    str(result.get("mtf_bias", "")),
            "decision":    result.get("decision"),
            "confidence":  result.get("confidence"),
            "indicators":  ind,
        })

        if final_action in ("BUY", "SELL"):
            trade_id = self.db.save_trade({
                "pair":       result.get("symbol"),
                "signal":     final_action,
                "entry":      result.get("entry"),
                "sl":         result.get("sl"),
                "tp":         result.get("tp"),
                "lot":        result.get("lot"),
                "result":     "OPEN",
                "pnl":        0,
                "rr_ratio":   result.get("rr"),
                "confidence": result.get("confidence"),
                "chart_snapshot": {
                    "rsi":     result.get("rsi"),
                    "trend":   result.get("trend"),
                    "regime":  result.get("regime"),
                    "pattern": pattern,
                    "mtf":     str(result.get("mtf_bias", "")),
                },
            })
            return trade_id

        return None

    def on_trade_closed(self, trade_id: int, result: str, pnl: float, pnl_pips: float = 0.0, close_reason: str = "TP/SL"):
        # Day 16: Database performance update
        self.db.update_trade_result(trade_id, result, pnl)
        self.db.update_daily_performance()

        trade = self.db.get_trade_by_id(trade_id)
        snapshot = (trade.get("chart_snapshot") or {}) if trade else {}
        lesson = self._generate_lesson(trade) if trade else "No trade context found."

        if result == "LOSS":
            self.db.auto_log_mistake(trade_id)
            self.pattern.add_losing_pattern({
                "pair":    trade.get("pair") if trade else "unknown",
                "signal":  trade.get("signal") if trade else "unknown",
                "pattern": snapshot.get("pattern", "unknown"),
                "regime":  snapshot.get("regime", "unknown"),
                "rsi":     snapshot.get("rsi"),
                "pnl":     pnl,
            }, lesson=lesson)
        else:
            self.pattern.add_winning_pattern({
                "pair":    trade.get("pair") if trade else "unknown",
                "signal":  trade.get("signal") if trade else "unknown",
                "pattern": snapshot.get("pattern", "unknown"),
                "regime":  snapshot.get("regime", "unknown"),
                "rsi":     snapshot.get("rsi"),
                "pnl":     pnl,
                "rr":      trade.get("rr_ratio", 0) if trade else 0,
            })

        # Day 33: Vector memory lesson
        if self._has_model() and trade:
            closed_trade_payload = {
                "pair":         trade.get("pair"),
                "type":         trade.get("signal"),
                "result":       result,
                "pnl":          pnl,
                "pnl_pips":     pnl_pips,
                "close_reason": close_reason,
                "context": {
                    "regime":   snapshot.get("regime", "unknown"),
                    "trend":    snapshot.get("trend", "unknown"),
                    "mtf_bias": snapshot.get("mtf", "unknown"),
                    "rsi":      snapshot.get("rsi")
                }
            }
            self.add_vector_lesson(closed_trade_payload)

        log.info(f"Trade #{trade_id} closed: {result} | PnL: ${pnl}")

    def _generate_lesson(self, trade: dict) -> str:
        conf = trade.get("confidence", 0)
        rr   = trade.get("rr_ratio", 0)
        snap = trade.get("chart_snapshot") or {}
        regime = snap.get("regime", "")

        if conf < 60:
            return f"Confidence was {conf}% — too low. Minimum 65% required before entry."
        elif rr < 1.5:
            return f"Risk-reward was 1:{rr} — too low. Never enter below 1:2."
        elif "BEAR" in str(regime) and trade.get("signal") == "BUY":
            return "Bought against bearish regime. Do not fight the higher timeframe trend."
        else:
            return "Setup looked valid but market moved against. Review entry timing."

    # ── Day 33 Vector Core Processing Methods ──────────────────────────────────

    def add_vector_lesson(self, closed_trade: dict) -> None:
        """Closed trade-কে ভেক্টরাইজ করে local রিদমে সেভ করে।"""
        if not self._has_model():
            return

        description = self._describe_setup(closed_trade)
        vector = self._model.encode([description])[0]

        lesson = {
            "pair":         closed_trade.get("pair"),
            "type":         closed_trade.get("type"),
            "result":       closed_trade.get("result"),
            "pnl":          closed_trade.get("pnl"),
            "pnl_pips":     closed_trade.get("pnl_pips"),
            "close_reason": closed_trade.get("close_reason"),
            "context":      closed_trade.get("context", {}),
            "description":  description,
            "saved_at":     datetime.utcnow().isoformat(timespec="seconds"),
        }

        self._metadata.append(lesson)
        if self._vectors is None:
            self._vectors = vector.reshape(1, -1)
        else:
            self._vectors = np.vstack([self._vectors, vector])

        self._save_vector_data()
        log.info(f"[TradeMemory Vector] Lesson saved: {lesson['pair']} | total: {len(self._metadata)}")

    def find_similar(self, current_context: dict, top_k: int = None) -> list[dict]:
        """করেন্ট মার্কেটের সাথে সবচেয়ে মিল থাকা অতীতের ট্রেডগুলো টপ-কে ভেক্টর ম্যাচিং করে আনে।"""
        if not self._has_model() or self._vectors is None or len(self._metadata) == 0:
            return []

        top_k = top_k or self.TOP_K
        query_desc = self._describe_context(current_context)
        query_vec = self._model.encode([query_desc])[0]

        similarities = self._cosine_similarity(query_vec, self._vectors)
        top_idx = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_idx:
            results.append({
                "lesson": self._metadata[idx],
                "similarity": round(float(similarities[idx]), 3),
            })
        return results

    def get_memory_context_text(self, current_context: dict) -> str:
        """AIAnalyst প্রম্পটে ইনজেক্ট করার জন্য একটি টেক্সট ব্লকে কনভার্ট করে।"""
        similar = self.find_similar(current_context)
        if not similar:
            return "No past lessons found (memory empty or vector model unavailable)."

        lines = ["-- PAST SIMILAR TRADES (TradeMemory) --"]
        for item in similar:
            lesson = item["lesson"]
            lines.append(
                f"  * {lesson['pair']} {lesson['type']} -> {lesson['result']} "
                f"(${lesson['pnl']}, {lesson.get('pnl_pips', 0)} pips) "
                f"[similarity {item['similarity']}] — {lesson.get('close_reason', '')}"
            )
        return "\n".join(lines)

    # ── Context Text Builders & Math ──────────────────────────────────────────

    def _describe_setup(self, closed_trade: dict) -> str:
        ctx = closed_trade.get("context", {})
        return (
            f"{closed_trade.get('pair')} {closed_trade.get('type')} trade "
            f"in {ctx.get('regime', 'unknown')} regime, trend {ctx.get('trend', 'unknown')}, "
            f"mtf bias {ctx.get('mtf_bias', 'unknown')}, result {closed_trade.get('result')} via {closed_trade.get('close_reason')}"
        )

    def _describe_context(self, current_context: dict) -> str:
        return (
            f"{current_context.get('symbol', 'unknown')} {current_context.get('signal', 'unknown')} trade "
            f"in {current_context.get('regime', 'unknown')} regime, trend {current_context.get('trend', 'unknown')}, "
            f"mtf bias {current_context.get('mtf_bias', 'unknown')}"
        )

    def _cosine_similarity(self, query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
        return matrix_norm @ query_norm

    # ── Context Fetchers for AI / Pipeline ─────────────────────────────────────

    def get_context_for_ai(self, pair: str) -> dict:
        stats   = self.db.get_overall_stats()
        lessons = self.db.get_lessons(pair=pair, limit=5)
        recent  = self.db.get_recent_trades(limit=5)
        return {
            "overall_win_rate": stats.get("win_rate", 0),
            "total_trades":     stats.get("total", 0),
            "total_pnl":        stats.get("total_pnl", 0),
            "lessons":          [l.get("lesson") for l in lessons],
            "recent_results":   [t.get("result") for t in recent],
        }

    def get_pattern_context(
        self,
        pair: str,
        regime: str = "",
        pattern: str = "",
        trend: str = None,
        rsi: float = None,
    ) -> dict:
        """
        Get pattern context for decision making.
        
        Supports both old signature (pair, regime, pattern) and new signature
        (pair, trend, rsi, pattern, regime) for backward compatibility.
        """
        # Use pattern_memory's get_summary_for_decision with available params
        return self.pattern.get_summary_for_decision(pair, regime, pattern)

    def get_stats(self) -> dict:
        return self.db.get_overall_stats()

    def diagnose_persistence(self) -> dict:
        """
        P2 audit diagnostic — verify trade memory is actually persisting.

        Returns a dict with:
          - db_path: where the SQLite DB lives
          - db_exists: bool
          - db_size_bytes: int
          - trades_table_count: row count of trades table
          - vector_lessons: count of vector lessons on disk
          - vectors_npy_exists: bool
          - metadata_json_exists: bool

        Call this when "0 closed / 0 total" appears in stats despite
        trades having been executed. The most common causes are:
          1. DB path is read-only or unwritable (silent failure).
          2. on_trade_closed() was never called by the orchestrator.
          3. Database() is being constructed in-memory (some test paths).
        """
        import os
        diag = {
            "db_path": getattr(self.db, "db_path", "unknown"),
            "db_exists": False,
            "db_size_bytes": 0,
            "trades_table_count": -1,
            "vector_lessons": len(self._metadata or []),
            "vectors_npy_exists": os.path.exists(VECTORS_PATH),
            "metadata_json_exists": os.path.exists(METADATA_PATH),
        }
        try:
            db_path = diag["db_path"]
            if db_path and os.path.exists(db_path):
                diag["db_exists"] = True
                diag["db_size_bytes"] = os.path.getsize(db_path)
            # Try to query trades count
            if hasattr(self.db, "conn") and self.db.conn:
                cur = self.db.conn.execute("SELECT COUNT(*) FROM trades")
                diag["trades_table_count"] = cur.fetchone()[0]
            elif hasattr(self.db, "get_recent_trades"):
                diag["trades_table_count"] = len(self.db.get_recent_trades(limit=10000))
        except Exception as e:
            log.error(f"[TradeMemory] diagnose_persistence error: {e}")
            diag["error"] = str(e)

        log.info(f"[TradeMemory] persistence diagnostic: {diag}")
        if diag["trades_table_count"] == 0:
            log.warning(
                "[TradeMemory] trades table is EMPTY despite bot running. "
                "Possible causes: (1) on_signal_generated() never called "
                "(check orchestrator wiring); (2) DB write is failing "
                "silently; (3) trades were never BUY/SELL (only NO TRADE)."
            )
        return diag

    def print_stats(self):
        self.db.print_stats()
        self.pattern.print_stats()

        if self._has_model() and self._metadata:
            wins  = sum(1 for m in self._metadata if m["result"] == "WIN")
            total = len(self._metadata)
            bar   = "=" * 44
            log.info(bar)
            log.info(
                f" [MEMORY] VECTOR MEMORY: {total} lessons | "
                f"WinRate: {round(wins/total*100, 1) if total else 0}%"
            )
            log.info(bar)

    # ── Disk I/O ─────────────────────────────────────────────────────────────

    def _load_vector_data(self) -> None:
        if os.path.exists(VECTORS_PATH) and os.path.exists(METADATA_PATH):
            try:
                self._vectors = np.load(VECTORS_PATH)
                with open(METADATA_PATH) as f:
                    self._metadata = json.load(f)
                log.info(f"[TradeMemory] Loaded {len(self._metadata)} past vector lessons")
            except Exception as e:
                log.error(f"[TradeMemory] Vector load failed: {e}")

    def _save_vector_data(self) -> None:
        if self._vectors is not None:
            np.save(VECTORS_PATH, self._vectors)
        with open(METADATA_PATH, "w") as f:
            json.dump(self._metadata, f, indent=2)

    def close(self):
        self.db.close()