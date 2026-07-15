# memory/knowledge_store.py  —  Day 16 | Vector Knowledge Memory

"""
AI-এর Long-Term Memory।

Short-term (SQLite):  "What happened?"
Long-term (ChromaDB): "Have I seen this before?"
"""

import json
import uuid
from pathlib import Path
from datetime import datetime

from utils.logger import get_logger
log = get_logger("knowledge_store")

CHROMA_AVAILABLE = False
_chroma_import_error = ""

try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMA_AVAILABLE = True
except ImportError as e:
    _chroma_import_error = str(e)


def _make_embedding_function():
    """
    Embedding function তৈরি করো।
    - আগে ChromaDB-র built-in ONNX embedding ব্যবহার করো (HuggingFace লাগে না)
    - না পেলে SentenceTransformer try করো (HF_TOKEN ব্যবহার করবে যদি থাকে)
    - সব fail হলে None return করো
    """
    if not CHROMA_AVAILABLE:
        return None

    # Set HF_TOKEN environment variable if available (fixes 401 Unauthorized)
    import os
    hf_token = os.getenv("HF_TOKEN", "")
    if hf_token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        os.environ["HF_HUB_AUTH_TOKEN"] = hf_token

    # Option 1: ChromaDB built-in ONNX embedding (no HuggingFace download needed)
    try:
        ef = embedding_functions.ONNXMiniLM_L6_V2()
        log.info("[KnowledgeStore] Using built-in ONNX embedding (no HuggingFace needed)")
        return ef
    except Exception as e:
        log.warning(f"[KnowledgeStore] ONNX embedding failed: {e}")

    # Option 2: SentenceTransformer (check local cache first, then use HF_TOKEN)
    # Day 81+ — Trigger shared cache so the model downloads exactly once
    # across TradeMemory + KnowledgeStore + MistakeAnalyzer.
    try:
        from memory.sentence_model_cache import get_sentence_model
        shared = get_sentence_model()  # populates HF cache on first call
        # We still create a ChromaDB embedding function via the standard API,
        # but the model files are now cached locally so no re-download.
        from sentence_transformers import SentenceTransformer as _ST  # noqa: F401

        model_name = "all-MiniLM-L6-v2"
        # Check multiple cache locations
        possible_cache_dirs = [
            os.path.join(os.path.expanduser("~"), ".cache", "torch", "sentence_transformers",
                        model_name.replace("/", "_")),
            os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub",
                        f"models--sentence-transformers--{model_name.replace('/', '_')}"),
        ]
        model_path = model_name  # default: download from HF
        for cache_dir in possible_cache_dirs:
            if os.path.isdir(cache_dir):
                model_path = cache_dir
                log.info(f"[KnowledgeStore] Found cached model at: {cache_dir}")
                break

        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_path
        )
        log.info(f"[KnowledgeStore] Using SentenceTransformer embedding: {model_path}")
        return ef
    except Exception as e:
        log.warning(f"[KnowledgeStore] SentenceTransformer embedding failed: {e}")

    # Option 3: ChromaDB default embedding (simplest fallback)
    try:
        ef = embedding_functions.DefaultEmbeddingFunction()
        log.info("[KnowledgeStore] Using ChromaDB DefaultEmbeddingFunction")
        return ef
    except Exception as e:
        log.warning(f"[KnowledgeStore] All embedding options failed: {e}")
        return None


class KnowledgeStore:
    """
    AI Trader-এর vector memory।

    store.add_memory("EURUSD bearish trend, RSI oversold near support")
    store.search_memory("bearish reversal setup at support")
    """

    def __init__(self, path: str = "memory/chroma_db"):
        if not CHROMA_AVAILABLE:
            log.warning(
                f"[KnowledgeStore] ChromaDB not installed — knowledge store disabled.\n"
                f"  To enable: pip install chromadb\n"
                f"  Error: {_chroma_import_error}"
            )
            self._disabled = True
            self.collection = None
            return

        self._disabled = False
        Path(path).mkdir(parents=True, exist_ok=True)

        try:
            self.client = chromadb.PersistentClient(path=path)

            self.ef = _make_embedding_function()

            # Day 72 fix: when getting an existing collection, pass our embedding
            # function explicitly to override whatever was stored previously.
            # This prevents ChromaDB from trying to use the old SentenceTransformer
            # embedding (which needs HuggingFace download → 401 error).
            #
            # Day 81+ hotfix: previously this used `get_collection` first and
            # fell back to `create_collection`. That pattern fails with
            # "Collection already exists" if ChromaDB's internal state lags
            # (e.g. another process created it, or a prior run left it).
            # `get_or_create_collection` is the canonical atomic way — it
            # returns the existing collection if present, otherwise creates
            # it, all in one call.
            #
            # Day 81+ hotfix #2: ChromaDB still throws "Embedding function
            # conflict" when the existing collection was created with a
            # DIFFERENT embedding function (e.g. sentence_transformer from
            # an older run, vs onnx_mini_lm_l6_v2 now). We catch this
            # specific conflict and DELETE the old collection, then recreate
            # with the current embedding function. This is safe because the
            # knowledge store is just a cache of trade-lesson embeddings —
            # it gets re-populated as the bot trades.
            collection_kwargs = {"name": "trading_memory"}
            if self.ef is not None:
                collection_kwargs["embedding_function"] = self.ef
            try:
                self.collection = self.client.get_or_create_collection(**collection_kwargs)
                log.info(f"[KnowledgeStore] Collection ready | Memories: {self.collection.count()}")
            except Exception as conflict_err:
                err_msg = str(conflict_err).lower()
                if ("embedding function" in err_msg and "conflict" in err_msg) \
                   or "already exists" in err_msg:
                    log.warning(
                        f"[KnowledgeStore] Embedding conflict detected — deleting old "
                        f"collection and recreating with current embedding function. "
                        f"(Old memories will be re-learned as the bot trades.)"
                    )
                    try:
                        self.client.delete_collection(name="trading_memory")
                        self.collection = self.client.create_collection(**collection_kwargs)
                        log.info(
                            f"[KnowledgeStore] Collection recreated with new embedding | "
                            f"Memories: 0 (will repopulate as bot trades)"
                        )
                    except Exception as recreate_err:
                        log.error(
                            f"[KnowledgeStore] Recreate after delete also failed: {recreate_err}"
                        )
                        raise
                else:
                    raise

        except Exception as e:
            log.error(f"[KnowledgeStore] Init failed: {e} — knowledge store disabled")
            self._disabled = True
            self.collection = None

    def _is_ready(self) -> bool:
        return not self._disabled and self.collection is not None

    # ── Add Memories ───────────────────────────────────────────

    def add_memory(self, text: str, metadata: dict = None) -> str | None:
        """যেকোনো trading knowledge save করো।"""
        if not self._is_ready():
            return None

        mem_id = str(uuid.uuid4())
        meta = {
            "date": datetime.now().isoformat(),
            "type": "general",
        }
        if metadata:
            meta.update(metadata)

        try:
            self.collection.add(
                documents=[text],
                metadatas=[meta],
                ids=[mem_id],
            )
            return mem_id
        except Exception as e:
            # Log only once per session, then suppress to avoid spam
            if not hasattr(self, '_add_error_logged'):
                log.error(f"[KnowledgeStore] add_memory failed: {e}")
                log.warning("[KnowledgeStore] Further add_memory errors will be suppressed")
                self._add_error_logged = True
            return None

    def add_trade_memory(self, trade: dict, result: str, lesson: str) -> str | None:
        """Trade শেষে automatically memory তৈরি করো।"""
        text = (
            f"{trade.get('pair')} {trade.get('signal')} trade. "
            f"Regime: {trade.get('regime', 'unknown')}. "
            f"RSI: {trade.get('rsi', 'N/A')}. "
            f"Pattern: {trade.get('pattern', 'none')}. "
            f"Confidence: {trade.get('confidence', 0)}%. "
            f"Result: {result}. "
            f"Lesson: {lesson}"
        )
        return self.add_memory(text, metadata={
            "type":       "trade",
            "pair":       trade.get("pair", ""),
            "signal":     trade.get("signal", ""),
            "result":     result,
            "pattern":    trade.get("pattern", ""),
            "regime":     trade.get("regime", ""),
            "confidence": str(trade.get("confidence", 0)),
        })

    def add_rule(self, rule: str) -> str | None:
        """Trading rule permanently store করো।"""
        return self.add_memory(rule, metadata={"type": "rule"})

    def add_lesson(self, lesson: str, pair: str = "") -> str | None:
        """ভুল থেকে শেখা lesson store করো।"""
        return self.add_memory(lesson, metadata={"type": "lesson", "pair": pair})

    # ── Search ────────────────────────────────────────────────

    def search_memory(self, query: str, limit: int = 3) -> list[dict]:
        """Current situation-এর মতো past cases খুঁজে দাও।"""
        if not self._is_ready() or self.collection.count() == 0:
            return []

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(limit, self.collection.count()),
            )

            output = []
            docs      = results.get("documents", [[]])[0]
            metas     = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for doc, meta, dist in zip(docs, metas, distances):
                similarity = round((1 - dist) * 100, 1)
                output.append({
                    "memory":     doc,
                    "metadata":   meta,
                    "similarity": similarity,
                })
            return output

        except Exception as e:
            log.error(f"[KnowledgeStore] search_memory failed: {e}")
            return []

    def search_similar_trades(self, current: dict) -> list[dict]:
        """Current market condition-এর মতো past trades খুঁজো।"""
        query = (
            f"{current.get('pair', '')} "
            f"{current.get('trend', '')} trend "
            f"RSI {current.get('rsi', '')} "
            f"pattern {current.get('pattern', 'none')} "
            f"regime {current.get('regime', '')}"
        )
        return self.search_memory(query, limit=5)

    # ── Bulk Seed ─────────────────────────────────────────────

    def seed_trading_rules(self):
        """Core trading rules একবার store করো। Skip if already seeded."""
        if not self._is_ready():
            log.warning("[KnowledgeStore] Skipping seed — store not ready")
            return

        # Day 72 fix: Skip seeding if collection already has memories.
        # This prevents repeated 401 HuggingFace errors on every restart.
        try:
            existing_count = self.collection.count()
            if existing_count > 0:
                log.info(f"[KnowledgeStore] Already has {existing_count} memories — skipping seed")
                return
        except Exception:
            pass  # If count fails, proceed with seeding

        rules = [
            "Never trade against the higher timeframe trend. If daily is bearish, avoid buying on lower timeframes.",
            "Never enter immediately after high-impact news events. Wait at least 30 minutes.",
            "Only enter when risk-reward ratio is at least 1:2. Never settle for less.",
            "Maximum 1% account risk per trade. Never risk more regardless of confidence.",
            "Wait for candle close confirmation before entering. Never enter mid-candle.",
            "If RSI is oversold in a strong downtrend, it can go lower. Oversold is not a buy signal alone.",
            "Support and resistance zones are more powerful on higher timeframes (4H, Daily).",
            "During high volatility (news events, session open), widen stop loss or avoid trading.",
            "Do not revenge trade after a loss. Stick to the plan.",
            "If confidence is below 65%, do not enter. Wait for a better setup.",
            "London-New York overlap session has highest volume and best setups.",
            "A bullish engulfing at strong support on 4H timeframe is a high probability setup.",
            "Stop loss should always be placed beyond the nearest structure (swing high/low).",
            "Never move stop loss to breakeven too early — give the trade room to breathe.",
            "Take profits at the next resistance level, not greedily targeting all-time highs.",
        ]

        for rule in rules:
            self.add_rule(rule)

        log.info(f"[KnowledgeStore] {len(rules)} trading rules stored in knowledge base")

    # ── Context for LLM ───────────────────────────────────────

    def get_context_for_llm(self, current_condition: dict) -> str:
        """LLM prompt-এ যোগ করার জন্য past experience summary।"""
        if not self._is_ready():
            return "Knowledge store unavailable."

        memories = self.search_similar_trades(current_condition)

        if not memories:
            return "No similar past experiences found."

        lines = ["Past similar situations:"]
        for i, m in enumerate(memories, 1):
            sim = m["similarity"]
            mem = m["memory"]
            lines.append(f"\n{i}. [{sim}% match] {mem}")

        return "\n".join(lines)

    # ── Stats ─────────────────────────────────────────────────

    def stats(self) -> dict:
        if not self._is_ready():
            return {"total_memories": 0, "status": "disabled"}
        return {"total_memories": self.collection.count(), "status": "ready"}

    def print_stats(self):
        s = self.stats()
        log.info(f"[KnowledgeStore] {s['total_memories']} memories stored | status={s['status']}")