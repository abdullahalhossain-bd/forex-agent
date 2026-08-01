#!/usr/bin/env python3
"""
scripts/build_book5_knowledge_base.py
======================================

Generates a consolidated JSON knowledge base from the Book 5 (Frank Miller
S&D) rule registry, organized by chapter/concept rather than by page.

Output: /home/z/my-project/download/book5_knowledge_base.json
"""

import json
import sys
import os
from pathlib import Path

# Add forex_ai to path
sys.path.insert(0, "/home/z/my-project/forex_ai")

from analysis.book_rules_index import BOOK_RULES, BookRule


# ════════════════════════════════════════════════════════════════
#  CHAPTER METADATA — Book 5 (Frank Miller S&D)
# ════════════════════════════════════════════════════════════════

CHAPTERS = {
    51: {
        "title": "Support/Resistance Basics & Limitations",
        "pages": "1-20",
        "summary": (
            "Introduces classic S/R theory and its limitations. S/R levels are "
            "prior swing points where price reversed; traditional S/R is "
            "weaker than supply/demand zones because it doesn't reflect "
            "institutional order flow."
        ),
        "key_concepts": ["support", "resistance", "role reversal", "S/R limitations"],
    },
    52: {
        "title": "Supply/Demand Economics & Balance/Imbalance",
        "pages": "21-30",
        "summary": (
            "Defines supply/demand in market terms. Price moves when "
            "supply and demand are imbalanced; price stalls when they "
            "are balanced. Zones of imbalance are where institutional "
            "orders accumulate."
        ),
        "key_concepts": ["supply", "demand", "balance", "imbalance", "institutional orders"],
    },
    53: {
        "title": "Zone Identification (ERC Rule, 3-Step Method)",
        "pages": "31-45",
        "summary": (
            "The 3-step method for identifying zones: (1) find the base, "
            "(2) confirm with Extended-Range Candlesticks (ERCs, body > 50% "
            "of range), (3) classify pattern type (DBR/RBR/RBD/DBD). "
            "Minimum 2 ERCs validate a genuine imbalance move."
        ),
        "key_concepts": ["ERC", "base", "imbalance", "DBR", "RBR", "RBD", "DBD",
                          "reversal", "continuation"],
    },
    54: {
        "title": "Zone Drawing (3 Risk-Based Methods)",
        "pages": "46-55",
        "summary": (
            "Three zone-drawing methods: low-risk (body-only), medium-risk "
            "(body + last wick), high-risk (full wick range). Each defines "
            "distal (stop-loss) and proximal (entry) lines. Includes the "
            "gap-anchoring rule for gap-formed zones."
        ),
        "key_concepts": ["distal line", "proximal line", "low-risk", "medium-risk",
                          "high-risk", "gap-anchoring", "entry", "stop-loss"],
    },
    55: {
        "title": "Fresh/Original Zone Filters",
        "pages": "56-60",
        "summary": (
            "Soft filters (not hard gates) for zone quality. Fresh = never "
            "retested; Original = formed independently, not as a reaction "
            "to a nearby prior zone. Used as weighting features, not "
            "binary pass/fail criteria."
        ),
        "key_concepts": ["freshness", "original zone", "soft filter", "test count"],
    },
    56: {
        "title": "Odd Enhancers Scoring System",
        "pages": "61-78",
        "summary": (
            "The book's central quantitative framework. 4 compulsory "
            "enhancers scored 2+2+3+3=10 max: (1) Strength of Move, "
            "(2) Time at Zone, (3) Fresh Zone, (4) Risk/Reward. Plus 2 "
            "optional enhancers: Original Zone, Overlapping Zones. Tier "
            "A (≥10) = full conviction; Tier B (7-9, no zero) = conditional; "
            "<7 = SKIP. Any enhancer=0 → hard SKIP gate."
        ),
        "key_concepts": ["odd enhancers", "scoring", "compulsory", "optional",
                          "tier A", "tier B", "hard gate", "2+2+3+3=10"],
    },
    57: {
        "title": "Price-Action Confluence (5 Patterns)",
        "pages": "79-88",
        "summary": (
            "Five PA patterns that confirm zones: pin bar, inside bar, "
            "head & shoulders, double top/bottom, engulfing. Used as an "
            "unofficial 7th enhancer (confluence layer) — not formally "
            "added to the numeric scoring but increases confidence when "
            "firing at a zone."
        ),
        "key_concepts": ["pin bar", "inside bar", "head & shoulders", "double top",
                          "double bottom", "engulfing", "confluence"],
    },
    58: {
        "title": "Flip Zones",
        "pages": "89-95",
        "summary": (
            "State-transition rule: when price CONFIRMED-breaks (candle "
            "close beyond distal) a zone, the zone flips type — demand "
            "becomes supply, supply becomes demand. Classic S/R role "
            "reversal ported to the S/D framework."
        ),
        "key_concepts": ["flip zone", "role reversal", "confirmed break", "wick vs close"],
    },
    59: {
        "title": "Reversal/Continuation Trading Rules",
        "pages": "96-105",
        "summary": (
            "3-pattern over-extension rule for distinguishing reversal "
            "from continuation setups. Defines entry/exit tactics for "
            "each pattern type."
        ),
        "key_concepts": ["reversal", "continuation", "over-extension"],
    },
    60: {
        "title": "Gap Trading",
        "pages": "106-115",
        "summary": (
            "Starting vs ending gaps. Gap-anchoring rule for zone "
            "construction when a gap forms the base."
        ),
        "key_concepts": ["starting gap", "ending gap", "gap-anchoring"],
    },
    61: {
        "title": "CCI Indicator Confluence",
        "pages": "116-125",
        "summary": (
            "Complete CCI state machine for entry/add/exit. Entry: "
            "CCI<-100 at demand → long; CCI>+100 at supply → short. "
            "Exit: CCI<+100 (long) or CCI>-100 (short). Add: CCI>0 "
            "(long) or CCI<0 (short). |CCI|<20 = ambiguous → HOLD. "
            "CCI is confluence only — NOT a standalone signal."
        ),
        "key_concepts": ["CCI", "entry rule", "exit rule", "add-to-position",
                          "ambiguous", "confluence stack", "zone failure diagnosis"],
    },
    62: {
        "title": "Multi-Timeframe 'Curve' Methodology",
        "pages": "126-135",
        "summary": (
            "The book's most quantitatively rich methodology. The 'curve' "
            "= price range between nearest demand & supply zone proximal "
            "lines on the higher timeframe. Split into 3 equal sub-zones "
            "(High/Equilibrium/Low). High/Very High → SELL_ONLY; "
            "Low/Very Low → BUY_ONLY; Equilibrium → TREND_FOLLOW_OR_NO_TRADE. "
            "HTF override: 'longer frame always wins' — LTF signals only "
            "actionable if they agree with HTF bias."
        ),
        "key_concepts": ["curve", "MTF", "thirds split", "directional bias",
                          "HTF override", "trading style", "timeframe triplet"],
    },
    63: {
        "title": "Trade Management Walkthrough",
        "pages": "136-152",
        "summary": (
            "Extended worked example applying the full system end-to-end. "
            "Generalized curve-position rule: bias holds until price "
            "crosses into the OPPOSITE extreme zone (persistence/"
            "invalidation condition)."
        ),
        "key_concepts": ["trade walkthrough", "bias persistence", "invalidation"],
    },
    64: {
        "title": "Risk Management",
        "pages": "153-157",
        "summary": (
            "Complete risk system. Risk per trade: 2% experienced, 1% "
            "beginner (until 3x growth). Position sizing: size = "
            "risk_amount / |entry-stop|. Forex: Case A (quote=account "
            "ccy) vs Case B (quote≠account, convert first). Margin call "
            "when loss% × leverage ≥ 100%. Drawdown circuit breaker: "
            "≥20% → halt for month; any drawdown → risk×0.75; restore "
            "only on new equity high."
        ),
        "key_concepts": ["position sizing", "margin call", "drawdown",
                          "circuit breaker", "risk per trade", "compounding"],
    },
}


# ════════════════════════════════════════════════════════════════
#  BUILD KNOWLEDGE BASE
# ════════════════════════════════════════════════════════════════

def rule_to_dict(rule: BookRule) -> dict:
    return {
        "rule_id": rule.rule_id,
        "page": rule.page,
        "category": rule.category,
        "name": rule.name,
        "rule_type": rule.rule_type,
        "implementation_file": rule.implementation_file,
        "implementation_function": rule.implementation_function,
        "description": rule.description,
        "no_trade_condition": rule.no_trade_condition,
    }


def build_knowledge_base() -> dict:
    """Assemble the complete knowledge base from rule registry + chapter metadata."""

    # Group rules by chapter
    rules_by_chapter = {}
    for rule in BOOK_RULES:
        if rule.chapter not in rules_by_chapter:
            rules_by_chapter[rule.chapter] = []
        rules_by_chapter[rule.chapter].append(rule_to_dict(rule))

    # Build chapter entries (Book 5 chapters only: 51-64)
    chapters_output = []
    for ch_num in sorted(CHAPTERS.keys()):
        meta = CHAPTERS[ch_num]
        rules = rules_by_chapter.get(ch_num, [])
        chapters_output.append({
            "chapter_number": ch_num,
            "book": "Book 5 — Supply & Demand (Frank Miller)",
            "title": meta["title"],
            "pages": meta["pages"],
            "summary": meta["summary"],
            "key_concepts": meta["key_concepts"],
            "rule_count": len(rules),
            "rules": rules,
        })

    # Compute aggregate stats
    book5_rules = [r for r in BOOK_RULES if r.chapter >= 51]
    total_rules = len(book5_rules)
    no_trade_count = sum(1 for r in book5_rules if r.no_trade_condition)
    by_category = {}
    for r in book5_rules:
        by_category[r.category] = by_category.get(r.category, 0) + 1

    # List implementation files
    impl_files = sorted(set(r.implementation_file for r in book5_rules))

    knowledge_base = {
        "metadata": {
            "title": "Book 5 (Frank Miller S&D) — Consolidated Knowledge Base",
            "source_book": "Supply & Demand Trading (Frank Miller)",
            "total_pages": "1-163",
            "total_chapters": 14,
            "chapter_numbering_scheme": (
                "Chapters 51-64 reserved for Book 5 to avoid collision with "
                "the Candlestick Bible book (chapters 1-9) in the same registry."
            ),
            "generated_by": "scripts/build_book5_knowledge_base.py",
        },
        "statistics": {
            "total_rules": total_rules,
            "no_trade_conditions": no_trade_count,
            "by_category": by_category,
            "implementation_files_count": len(impl_files),
            "chapters_covered": len(chapters_output),
        },
        "implementation_files": impl_files,
        "chapters": chapters_output,
        "cross_chapter_concepts": {
            "confluence_stack": {
                "description": (
                    "The book's overarching thesis: stack multiple "
                    "confluence layers (zone + trend line + CCI + PA pattern) "
                    "to maximize win probability. No single layer is "
                    "sufficient — each is a confirmation of the others."
                ),
                "layers": [
                    "Supply/Demand zone (Chapter 53-55)",
                    "Zone scoring (Chapter 56 — odd enhancers)",
                    "PA pattern at zone (Chapter 57)",
                    "Flip zone state (Chapter 58)",
                    "CCI confluence (Chapter 61)",
                    "MTF curve bias (Chapter 62)",
                    "Risk management (Chapter 64)",
                ],
            },
            "no_trade_conditions": {
                "description": (
                    "Hard gates that mandate SKIP (no trade). Drawn from "
                    "across all chapters."
                ),
                "conditions": [
                    "Zone score < 7 (Chapter 56)",
                    "Any compulsory enhancer = 0 (Chapter 56)",
                    "Base ≥ 6 candles (Chapter 56, P66)",
                    "Zone retested ≥ 2 times (Chapter 56, P68)",
                    "R:R < 1:1.5 (Chapter 56, P69)",
                    "Weak departure, 0 ERCs (Chapter 56, P65)",
                    "CCI near zero, |CCI| < 20 (Chapter 61, P125)",
                    "HTF bias conflicts with LTF signal (Chapter 62, P135)",
                    "Drawdown ≥ 20% from peak (Chapter 64, P157)",
                    "Margin call triggered (Chapter 64, P154)",
                ],
            },
            "scoring_system": {
                "description": (
                    "The 4 compulsory + 2 optional odd enhancers scoring "
                    "system. Max score = 10 (2+2+3+3)."
                ),
                "compulsory_enhancers": [
                    {"id": 1, "name": "Strength of Move", "max": 2,
                     "tiers": "2 (≥2 ERCs) / 1 (1 ERC) / 0 (0 ERCs)"},
                    {"id": 2, "name": "Time at Zone", "max": 2,
                     "tiers": "2 (≤3 candles) / 1 (4-5) / 0 (≥6)"},
                    {"id": 3, "name": "Fresh Zone", "max": 3,
                     "tiers": "3 (0 retests) / 1.5 (1) / 0 (≥2)"},
                    {"id": 4, "name": "Risk/Reward", "max": 3,
                     "tiers": "3 (R:R≥3) / 1.5 (1.5-2) / 0 (<1.5)"},
                ],
                "optional_enhancers": [
                    {"id": 5, "name": "Original Zone", "type": "boolean"},
                    {"id": 6, "name": "Overlapping Zones (MTF confluence)",
                     "type": "boolean"},
                    {"id": 7, "name": "PA Confluence (unofficial)",
                     "type": "boolean"},
                ],
                "tier_thresholds": {
                    "tier_A": "score ≥ 10 → full conviction (limit order)",
                    "tier_B": "score 7-9 (no zero) → conditional (confirmation entry)",
                    "skip": "score < 7 OR any enhancer = 0",
                },
            },
            "mtf_hierarchy": {
                "description": (
                    "Top-down MTF analysis with HTF override. 'The longer "
                    "frame always wins' (Book P135)."
                ),
                "trading_styles": {
                    "scalper":  {"long": "15m", "medium": "5m", "short": "1m"},
                    "day":      {"long": "1d",  "medium": "4h", "short": "1h"},
                    "swing":    {"long": "1w",  "medium": "1d", "short": "4h"},
                    "position": {"long": "1M",  "medium": "1w", "short": "1d"},
                },
                "curve_split": "subzone_width = (supply_proximal - demand_proximal) / 3",
                "bias_rule": {
                    "very_low": "BUY_ONLY",
                    "low": "BUY_ONLY",
                    "equilibrium": "TREND_FOLLOW_OR_NO_TRADE",
                    "high": "SELL_ONLY",
                    "very_high": "SELL_ONLY",
                },
            },
            "risk_management": {
                "description": (
                    "Complete risk system from Chapter 14."
                ),
                "risk_per_trade": {
                    "experienced": "2%",
                    "beginner": "1% (until 3x account growth)",
                },
                "position_sizing_formula": "size = risk_amount / |entry - stop|",
                "margin_call_trigger": "account_loss% × leverage ≥ 100%",
                "drawdown_circuit_breaker": {
                    "halt_threshold": "20% drawdown → stop trading for month",
                    "risk_reduction_in_drawdown": "25% (2% → 1.5%)",
                    "restore_condition": "new equity high",
                },
                "compounding_formula": "remaining = initial × (1 - risk%)^n_losses",
            },
        },
    }

    return knowledge_base


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════

def main():
    print("=" * 64)
    print("  Building Book 5 Consolidated Knowledge Base")
    print("=" * 64)

    kb = build_knowledge_base()

    output_path = "/home/z/my-project/download/book5_knowledge_base.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2, ensure_ascii=False)

    file_size = os.path.getsize(output_path)
    print(f"\n✓ Knowledge base written to: {output_path}")
    print(f"  File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
    print(f"  Total chapters: {kb['statistics']['chapters_covered']}")
    print(f"  Total rules: {kb['statistics']['total_rules']}")
    print(f"  No-trade conditions: {kb['statistics']['no_trade_conditions']}")
    print(f"  Implementation files: {kb['statistics']['implementation_files_count']}")
    print(f"\n  Chapters covered:")
    for ch in kb["chapters"]:
        print(f"    Ch {ch['chapter_number']:>2} ({ch['pages']:>8}): "
              f"{ch['title']:<45} [{ch['rule_count']} rules]")

    print(f"\n  Cross-chapter concepts:")
    for concept in kb["cross_chapter_concepts"]:
        print(f"    - {concept}")

    print("\n" + "=" * 64)
    print("  Knowledge base build complete.")
    print("=" * 64)


if __name__ == "__main__":
    main()
