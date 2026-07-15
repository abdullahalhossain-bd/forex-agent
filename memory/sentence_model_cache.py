"""
memory/sentence_model_cache.py — Shared SentenceTransformer Cache (Day 81+)
============================================================================

WHY THIS EXISTS:
    TradeMemory, KnowledgeStore, and MistakeAnalyzer each independently
    call `SentenceTransformer("all-MiniLM-L6-v2")` at startup. That
    triggers 3 duplicate HuggingFace download checks (~6s each = ~18s
    wasted) on every boot. This module shares a single model instance
    across all callers, so the download check happens exactly once.

USAGE:
    from memory.sentence_model_cache import get_sentence_model

    model = get_sentence_model()  # returns shared SentenceTransformer or None
    if model is not None:
        embedding = model.encode(["some text"])
"""
from __future__ import annotations

import threading
from typing import Optional

from utils.logger import get_logger

log = get_logger("sentence_model_cache")

_MODEL_NAME = "all-MiniLM-L6-v2"
_shared_model = None
_lock = threading.Lock()
_load_attempted = False


def get_sentence_model():
    """Return a shared SentenceTransformer instance, or None if unavailable.

    The model is loaded exactly once on first call. Subsequent calls
    return the cached instance. If loading fails (e.g. no internet,
    sentence-transformers not installed), returns None forever — callers
    must handle that gracefully.
    """
    global _shared_model, _load_attempted
    if _load_attempted:
        return _shared_model
    with _lock:
        if _load_attempted:
            return _shared_model
        _load_attempted = True
        try:
            from sentence_transformers import SentenceTransformer
            log.info(f"[SentenceModelCache] Loading '{_MODEL_NAME}' (once, shared)...")
            _shared_model = SentenceTransformer(_MODEL_NAME)
            log.info(f"[SentenceModelCache] Model loaded — shared across all callers")
        except ImportError:
            log.warning("[SentenceModelCache] sentence-transformers not installed")
            _shared_model = None
        except Exception as e:
            log.warning(f"[SentenceModelCache] load failed: {e}")
            _shared_model = None
    return _shared_model


def reset_cache() -> None:
    """Force-reload the model on next call. Used in tests only."""
    global _shared_model, _load_attempted
    with _lock:
        _shared_model = None
        _load_attempted = False
