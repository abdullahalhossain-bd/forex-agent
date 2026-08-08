"""
backtest/llm_loss_analyzer.py — Async LLM worker for loss analysis.

Reads losses from the LLM queue (llm_analysis/queue.jsonl), calls an LLM
to analyze each loss, writes structured JSON output to llm_analysis/<symbol>.jsonl.

CRITICAL CONTRACT:
  - This worker runs in a SEPARATE PROCESS from the backtest engine.
  - It NEVER blocks the backtest loop.
  - If the LLM API is unavailable, it retries with exponential backoff,
    then marks the analysis as "failed" — but the trade itself remains
    safely persisted in trades/<symbol>.jsonl.
  - The backtest engine enqueues losses by appending to queue.jsonl;
    this worker consumes them asynchronously.

USAGE:
    # Start the LLM analyzer as a separate process
    python -m backtest.llm_loss_analyzer --run-id 2026-08-08_153022

    # Or run inline (blocks — useful for testing without separate process)
    python -m backtest.llm_loss_analyzer --run-id 2026-08-08_153022 --once

LLM PROVIDERS:
    Uses the same provider cascade as live MasterAnalyst:
      Groq (llama-3.1-8b-instant) → Gemini → OpenRouter
    Configured via environment variables (GROQ_API_KEY, GEMINI_API_KEY, etc.)

OUTPUT SCHEMA (per analysis):
    {
      "trade_id": int,
      "symbol": str,
      "classification": str,  # see CLASSIFICATIONS below
      "primary_reason": str,
      "secondary_reasons": [str],
      "market_condition": str,
      "signal_quality": "good"|"weak"|"bad",
      "execution_quality": "good"|"poor",
      "risk_management_quality": "good"|"poor",
      "avoidable": bool,
      "confidence_assessment": str,
      "explanation": str,
      "lessons": [str],
      "llm_model": str,
      "analyzed_at": str  # ISO timestamp
    }

CLASSIFICATIONS:
    bad_signal, trend_reversal, choppy_market, false_breakout,
    support_resistance_failure, poor_confluence, high_volatility,
    spread_issue, late_entry, bad_RR, execution_issue,
    normal_random_loss, unknown
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backtest.persistence import RunDir, atomic_write_json, append_jsonl, read_jsonl, read_jsonl_count

log = logging.getLogger("llm_loss_analyzer")


# ── Valid classification values ──────────────────────────────────────────

CLASSIFICATIONS = {
    "bad_signal", "trend_reversal", "choppy_market", "false_breakout",
    "support_resistance_failure", "poor_confluence", "high_volatility",
    "spread_issue", "late_entry", "bad_RR", "execution_issue",
    "normal_random_loss", "unknown",
}


# ── LLM provider cascade ─────────────────────────────────────────────────

def _call_groq(prompt: str, timeout: int = 30) -> Optional[str]:
    """Call Groq API. Returns response text or None on failure."""
    api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEYS")
    if not api_key:
        return None
    # Support multiple keys (comma-separated) — pick the first
    if "," in api_key:
        api_key = api_key.split(",")[0].strip()
    try:
        import urllib.request
        url = "https://api.groq.com/openai/v1/chat/completions"
        model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
        body = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a forex trading analyst. Analyze trade losses and return ONLY valid JSON."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 1000,
        }).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST", headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        log.debug(f"Groq call failed: {e}")
        return None


def _call_gemini(prompt: str, timeout: int = 30) -> Optional[str]:
    """Call Gemini API. Returns response text or None on failure."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEYS")
    if not api_key:
        return None
    if "," in api_key:
        api_key = api_key.split(",")[0].strip()
    try:
        import urllib.request
        model = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        body = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "systemInstruction": {"parts": [{"text": "You are a forex trading analyst. Analyze trade losses and return ONLY valid JSON."}]},
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 1000},
        }).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST", headers={
            "Content-Type": "application/json",
        })
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        log.debug(f"Gemini call failed: {e}")
        return None


def _call_openrouter(prompt: str, timeout: int = 30) -> Optional[str]:
    """Call OpenRouter API. Returns response text or None on failure."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return None
    try:
        import urllib.request
        url = "https://openrouter.ai/api/v1/chat/completions"
        model = os.environ.get("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free")
        body = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a forex trading analyst. Analyze trade losses and return ONLY valid JSON."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 1000,
        }).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST", headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        log.debug(f"OpenRouter call failed: {e}")
        return None


def _call_llm(prompt: str, timeout: int = 30) -> tuple[Optional[str], str]:
    """Try each provider in cascade. Returns (response_text, provider_name) or (None, "")."""
    for name, fn in [("groq", _call_groq), ("gemini", _call_gemini), ("openrouter", _call_openrouter)]:
        resp = fn(prompt, timeout=timeout)
        if resp:
            return resp, name
    return None, ""


# ── Prompt builder ───────────────────────────────────────────────────────

def _build_prompt(loss_record: dict) -> str:
    """Build the LLM prompt for a loss analysis.

    Captures the trade's essential context without sending the full DataFrame.
    """
    return f"""Analyze this forex trade loss and return JSON.

TRADE:
- Symbol: {loss_record.get('symbol', '?')}
- Direction: {loss_record.get('direction', '?')}
- Entry: {loss_record.get('entry', '?')}
- Exit: {loss_record.get('exit', '?')}
- Stop Loss: {loss_record.get('sl', '?')}
- Take Profit: {loss_record.get('tp', '?')}
- Lot: {loss_record.get('lot', '?')}
- P&L: ${loss_record.get('pnl_usd', '?')} ({loss_record.get('pnl_pips', '?')} pips)
- Exit Reason: {loss_record.get('exit_reason', '?')}
- Hold Bars: {loss_record.get('hold_bars', '?')}
- Confidence: {loss_record.get('confidence', '?')}
- Entry Time: {loss_record.get('entry_time', '?')}
- Exit Time: {loss_record.get('exit_time', '?')}

Return ONLY valid JSON with this exact schema (no markdown, no commentary):
{{
  "classification": "bad_signal|trend_reversal|choppy_market|false_breakout|support_resistance_failure|poor_confluence|high_volatility|spread_issue|late_entry|bad_RR|execution_issue|normal_random_loss|unknown",
  "primary_reason": "one-sentence main reason",
  "secondary_reasons": ["reason 1", "reason 2"],
  "market_condition": "trending|ranging|volatile|choppy|transition",
  "signal_quality": "good|weak|bad",
  "execution_quality": "good|poor",
  "risk_management_quality": "good|poor",
  "avoidable": true,
  "confidence_assessment": "was confidence too high/low/appropriate",
  "explanation": "2-3 sentence analysis",
  "lessons": ["lesson 1", "lesson 2"]
}}"""


def _parse_llm_response(response: str, trade_id: int, symbol: str, provider: str) -> dict:
    """Parse LLM response into structured JSON. Tolerant of markdown wrappers."""
    # Strip markdown code fences if present
    cleaned = response.strip()
    if cleaned.startswith("```"):
        # Remove first line (```json or ```) and last line (```)
        lines = cleaned.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to extract JSON object from the response
        import re
        match = re.search(r'\{[^{}]*\}', cleaned, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                parsed = {}
        else:
            parsed = {}

    # Normalize classification
    classification = parsed.get("classification", "unknown")
    if classification not in CLASSIFICATIONS:
        classification = "unknown"

    return {
        "trade_id": trade_id,
        "symbol": symbol,
        "classification": classification,
        "primary_reason": str(parsed.get("primary_reason", ""))[:500],
        "secondary_reasons": list(parsed.get("secondary_reasons", []))[:10],
        "market_condition": str(parsed.get("market_condition", "unknown")),
        "signal_quality": str(parsed.get("signal_quality", "unknown")),
        "execution_quality": str(parsed.get("execution_quality", "unknown")),
        "risk_management_quality": str(parsed.get("risk_management_quality", "unknown")),
        "avoidable": bool(parsed.get("avoidable", False)),
        "confidence_assessment": str(parsed.get("confidence_assessment", ""))[:500],
        "explanation": str(parsed.get("explanation", ""))[:2000],
        "lessons": list(parsed.get("lessons", []))[:10],
        "llm_model": provider,
        "analyzed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# ── Worker ───────────────────────────────────────────────────────────────

class LLMAnalyzerWorker:
    """Consumes losses from the queue and writes LLM analyses.

    Designed to run in a separate process, polling the queue file for
    new losses. The backtest engine appends to queue.jsonl; this worker
    reads it, processes each pending loss, and writes results to
    llm_analysis/<symbol>.jsonl.

    Pending losses are tracked by reading the queue and comparing
    trade_ids against already-analyzed trade_ids per symbol.
    """

    def __init__(
        self,
        run_dir: RunDir,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        poll_interval: float = 5.0,
        timeout: int = 30,
    ):
        self.run_dir = run_dir
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.poll_interval = poll_interval
        self.timeout = timeout

    def run_forever(self, stop_when_queue_empty: bool = False):
        """Poll the queue forever (or until empty if stop_when_queue_empty).

        Args:
            stop_when_queue_empty: if True, exit after processing all
                pending losses. Useful for testing. If False (default),
                keep polling forever (production mode).
        """
        log.info(f"[llm_worker] Starting LLM analyzer for run {self.run_dir.run_id}")
        log.info(f"[llm_worker] Queue: {self.run_dir.llm_queue_path}")
        log.info(f"[llm_worker] Poll interval: {self.poll_interval}s, "
                 f"max retries: {self.max_retries}, timeout: {self.timeout}s")

        while True:
            processed = self._process_pending()
            if processed > 0:
                log.info(f"[llm_worker] Processed {processed} losses")
            if stop_when_queue_empty and processed == 0:
                log.info("[llm_worker] Queue empty — exiting")
                break
            time.sleep(self.poll_interval)

    def _process_pending(self) -> int:
        """Process all pending losses. Returns count processed."""
        # Read all queued losses
        queued = list(read_jsonl(self.run_dir.llm_queue_path))
        if not queued:
            return 0

        # Read already-analyzed trade_ids per symbol
        analyzed_per_symbol: dict[str, set] = {}
        for sym_file in self.run_dir.llm_dir.glob("*.jsonl"):
            if sym_file.name == "queue.jsonl":
                continue
            symbol = sym_file.stem
            analyzed_per_symbol[symbol] = {
                rec.get("trade_id") for rec in read_jsonl(sym_file)
                if rec.get("trade_id") is not None
            }

        # Find pending (not yet analyzed)
        pending = []
        for rec in queued:
            symbol = rec.get("symbol")
            tid = rec.get("trade_id")
            if symbol and tid is not None:
                if tid not in analyzed_per_symbol.get(symbol, set()):
                    pending.append(rec)

        if not pending:
            return 0

        log.info(f"[llm_worker] {len(pending)} pending losses to analyze")

        # Process each
        processed = 0
        for loss_record in pending:
            success = self._analyze_one(loss_record)
            if success:
                processed += 1
            else:
                # On failure, log and continue (don't block the queue)
                log.warning(f"[llm_worker] Failed to analyze trade "
                            f"{loss_record.get('trade_id')} ({loss_record.get('symbol')})")

        # Update the loss_summary.json
        self._update_loss_summary()
        return processed

    def _analyze_one(self, loss_record: dict) -> bool:
        """Analyze one loss. Returns True on success, False on failure."""
        trade_id = loss_record.get("trade_id")
        symbol = loss_record.get("symbol", "UNKNOWN")

        # Build prompt
        prompt = _build_prompt(loss_record)

        # Try with retries
        for attempt in range(self.max_retries):
            response, provider = _call_llm(prompt, timeout=self.timeout)
            if response:
                # Parse + persist
                analysis = _parse_llm_response(response, trade_id, symbol, provider)
                try:
                    self.run_dir.append_llm_analysis(symbol, analysis)
                    log.info(f"[llm_worker] Analyzed trade {trade_id} ({symbol}) "
                             f"→ {analysis['classification']} via {provider}")
                    return True
                except Exception as e:
                    log.error(f"[llm_worker] Failed to persist analysis for "
                              f"trade {trade_id}: {e}")
                    return False
            # Backoff before retry
            if attempt < self.max_retries - 1:
                delay = self.retry_delay * (2 ** attempt)
                log.debug(f"[llm_worker] Retry {attempt+1}/{self.max_retries} "
                          f"for trade {trade_id} after {delay}s")
                time.sleep(delay)

        # All retries failed — write a "failed" marker so we don't retry forever
        failed_record = {
            "trade_id": trade_id,
            "symbol": symbol,
            "classification": "unknown",
            "primary_reason": "LLM analysis failed after all retries",
            "secondary_reasons": [],
            "market_condition": "unknown",
            "signal_quality": "unknown",
            "execution_quality": "unknown",
            "risk_management_quality": "unknown",
            "avoidable": False,
            "confidence_assessment": "n/a",
            "explanation": "LLM API unavailable or returned invalid response",
            "lessons": [],
            "llm_model": "none",
            "analyzed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": "failed",
        }
        try:
            self.run_dir.append_llm_analysis(symbol, failed_record)
        except Exception:
            pass
        return False

    def _update_loss_summary(self):
        """Write loss_summary.json with category counts."""
        categories: dict[str, int] = {}
        total_analyzed = 0
        total_failed = 0
        for sym_file in self.run_dir.llm_dir.glob("*.jsonl"):
            if sym_file.name == "queue.jsonl":
                continue
            for rec in read_jsonl(sym_file):
                total_analyzed += 1
                if rec.get("status") == "failed":
                    total_failed += 1
                cat = rec.get("classification", "unknown")
                categories[cat] = categories.get(cat, 0) + 1

        # Count pending from queue
        queued = list(read_jsonl(self.run_dir.llm_queue_path))
        analyzed_tids = set()
        for sym_file in self.run_dir.llm_dir.glob("*.jsonl"):
            if sym_file.name == "queue.jsonl":
                continue
            for rec in read_jsonl(sym_file):
                if rec.get("trade_id") is not None:
                    analyzed_tids.add(rec["trade_id"])
        pending = sum(1 for q in queued if q.get("trade_id") not in analyzed_tids)

        summary = {
            "total_losses_queued": len(queued),
            "analyzed": total_analyzed,
            "pending": pending,
            "failed": total_failed,
            "categories": categories,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        try:
            atomic_write_json(self.run_dir.llm_summary_path, summary)
        except Exception as e:
            log.warning(f"[llm_worker] Failed to write loss summary: {e}")


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM loss analyzer worker")
    parser.add_argument("--run-id", required=True, help="Run ID to analyze")
    parser.add_argument("--once", action="store_true",
                        help="Process pending losses once, then exit (don't poll)")
    parser.add_argument("--poll-interval", type=float, default=5.0,
                        help="Seconds between queue polls (default: 5)")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Max LLM retries per loss (default: 3)")
    parser.add_argument("--timeout", type=int, default=30,
                        help="LLM call timeout in seconds (default: 30)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%H:%M:%S",
    )

    run_dir = RunDir(args.run_id)
    if not run_dir.root.exists():
        log.error(f"Run directory does not exist: {run_dir.root}")
        sys.exit(1)

    worker = LLMAnalyzerWorker(
        run_dir=run_dir,
        max_retries=args.max_retries,
        poll_interval=args.poll_interval,
        timeout=args.timeout,
    )

    if args.once:
        # Process pending and exit
        processed = worker._process_pending()
        log.info(f"Processed {processed} losses (one-shot mode)")
    else:
        # Run forever (until Ctrl+C)
        worker.run_forever(stop_when_queue_empty=False)


if __name__ == "__main__":
    main()
