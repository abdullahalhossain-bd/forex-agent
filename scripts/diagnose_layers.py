#!/usr/bin/env python3
"""
Decision Layer Diagnostic — Forex AI Trading System
====================================================
Tests every decision layer in the pipeline and reports status.

Usage:
    python scripts/diagnose_layers.py
    python scripts/diagnose_layers.py --pair EURUSD
    python scripts/diagnose_layers.py --verbose
"""
import argparse
import sys
import os
import time
import traceback
from datetime import datetime

# Ensure project root is on path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# Color codes for terminal output
class Color:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'
    GRAY = '\033[90m'

# Windows color fix
if sys.platform == 'win32':
    os.system('color')


def status_icon(ok):
    if ok is True:
        return f"{Color.GREEN}✅ OK{Color.END}"
    elif ok is False:
        return f"{Color.RED}❌ FAIL{Color.END}"
    elif ok is None:
        return f"{Color.YELLOW}⚠️  WARN{Color.END}"
    return str(ok)


results = []


def test_layer(name, layer_func, category):
    """Run a single layer test and record the result."""
    print(f"\n{'─'*60}")
    print(f"  Testing: {Color.BOLD}{name}{Color.END} [{category}]")
    print(f"{'─'*60}")

    t0 = time.time()
    try:
        result = layer_func()
        elapsed = time.time() - t0

        # Handle dict result with 'ok' key
        if isinstance(result, dict):
            ok_val = result.get('ok', False)
            detail = result.get('detail', '')
            if ok_val is True:
                status = True
            elif ok_val is None:
                status = None
            else:
                status = False
        elif result is True:
            status = True
            detail = ''
        elif result is None:
            status = None
            detail = 'skipped'
        else:
            status = False
            detail = str(result)

        icon = status_icon(status)
        print(f"  {icon}  ({elapsed:.2f}s)")
        if detail:
            print(f"  {Color.GRAY}Detail: {detail}{Color.END}")

        results.append({
            'name': name,
            'category': category,
            'status': status,
            'detail': detail,
            'elapsed': elapsed,
        })
        return status
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  {status_icon(False)}  ({elapsed:.2f}s)")
        print(f"  {Color.RED}Error: {e}{Color.END}")
        if args.verbose:
            traceback.print_exc()
        results.append({
            'name': name,
            'category': category,
            'status': False,
            'detail': str(e),
            'elapsed': elapsed,
        })
        return False


# ─── Layer Tests ──────────────────────────────────────────────

def test_imports():
    """Test if all critical modules can be imported."""
    modules = [
        ('config', 'EXECUTION_MODE'),
        ('execution.execution_router', 'ExecutionRouter'),
        ('risk.trade_permission', 'TradePermission'),
        ('ml.model_store', 'ModelStore'),
        ('ml.model_predictor', 'ModelPredictor'),
        ('utils.safe_pickle', 'safe_pickle_load'),
        ('agents.learning_agent', 'LearningAgent'),
        ('agents.decision_agent', 'DecisionAgent'),
    ]
    failed = []
    for mod_name, attr in modules:
        try:
            mod = __import__(mod_name, fromlist=[attr])
            if not hasattr(mod, attr):
                failed.append(f"{mod_name}.{attr} (attr missing)")
        except Exception as e:
            failed.append(f"{mod_name} ({type(e).__name__})")

    # These modules have heavy dependencies — test separately
    heavy_modules = [
        ('core.trader', 'AITrader'),
        ('agents.analysis_agent', 'AnalysisAgent'),
    ]
    heavy_failed = []
    for mod_name, attr in heavy_modules:
        try:
            mod = __import__(mod_name, fromlist=[attr])
            if not hasattr(mod, attr):
                heavy_failed.append(f"{mod_name}.{attr} (attr missing)")
        except Exception as e:
            heavy_failed.append(f"{mod_name} ({type(e).__name__}: {str(e)[:50]})")

    if not failed and not heavy_failed:
        return {'ok': True, 'detail': f'{len(modules)+len(heavy_modules)} modules imported OK'}
    elif not failed and heavy_failed:
        return {'ok': None, 'detail': f'Core OK, heavy deps missing: {"; ".join(heavy_failed[:2])}'}
    return {'ok': False, 'detail': f'Failed: {"; ".join(failed[:3])}'}


def test_config():
    """Test config values."""
    from config import (EXECUTION_MODE, SIMULATION_MODE,
                        MT5_FALLBACK_TO_SIMULATION, MAX_LOT)
    issues = []
    if EXECUTION_MODE not in ('mt5_demo',):
        issues.append(f"EXECUTION_MODE={EXECUTION_MODE}")
    if not isinstance(MT5_FALLBACK_TO_SIMULATION, bool):
        issues.append("MT5_FALLBACK_TO_SIMULATION not bool")
    if MAX_LOT <= 0 or MAX_LOT > 10:
        issues.append(f"MAX_LOT={MAX_LOT} (suspicious)")

    if issues:
        return {'ok': None, 'detail': '; '.join(issues)}
    return {'ok': True, 'detail': f'mode={EXECUTION_MODE}, fallback={MT5_FALLBACK_TO_SIMULATION}, max_lot={MAX_LOT}'}


def test_database():
    """Test database connection."""
    try:
        from memory.database import Database
        db = Database()
        if hasattr(db, '_conn'):
            cursor = db._conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
        return {'ok': True, 'detail': 'DB connection OK'}
    except Exception as e:
        return {'ok': False, 'detail': str(e)}


def test_mt5_connection():
    """Test MT5 connection (or simulation fallback)."""
    try:
        from execution.execution_router import ExecutionRouter
        router = ExecutionRouter(mode='mt5_demo', db=None, paper_trader=None)
        if router._simulation_mode:
            return {'ok': None, 'detail': 'SIMULATION mode (MT5 not connected)'}
        elif router._mt5_conn and getattr(router._mt5_conn, 'connected', False):
            return {'ok': True, 'detail': 'MT5 connected'}
        return {'ok': False, 'detail': 'MT5 not connected, not in simulation'}
    except Exception as e:
        return {'ok': False, 'detail': str(e)}


def test_model_store():
    """Test ML model store loading."""
    try:
        from ml.model_store import ModelStore
        store = ModelStore()
        registry = getattr(store, '_registry', {})
        models = registry.get('models', {})
        count = len(models)
        if count == 0:
            return {'ok': False, 'detail': 'No models in registry'}
        return {'ok': True, 'detail': f'{count} models in registry'}
    except Exception as e:
        return {'ok': False, 'detail': str(e)}


def test_ml_models_loading(args):
    """Test loading each ML model."""
    try:
        from ml.model_store import ModelStore
        store = ModelStore()
        pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCAD', 'AUDUSD', 'XAUUSD']
        loaded = 0
        for pair in pairs:
            model = store.load_model(pair, '15m', 'xgboost')
            if model is not None:
                loaded += 1
        if loaded == 0:
            return {'ok': False, 'detail': '0/6 models loaded'}
        elif loaded < len(pairs):
            return {'ok': None, 'detail': f'{loaded}/{len(pairs)} models loaded'}
        return {'ok': True, 'detail': f'{loaded}/{len(pairs)} models loaded'}
    except Exception as e:
        return {'ok': False, 'detail': str(e)}


def test_model_predictor():
    """Test ModelPredictor.is_ready()."""
    try:
        from ml.model_predictor import ModelPredictor
        p = ModelPredictor()
        if not hasattr(p, 'is_ready'):
            return {'ok': False, 'detail': 'is_ready() method MISSING'}
        ready = p.is_ready()
        if ready:
            return {'ok': True, 'detail': 'is_ready()=True'}
        return {'ok': False, 'detail': 'is_ready()=False — models not loadable'}
    except Exception as e:
        return {'ok': False, 'detail': str(e)}


def test_rl_agent():
    """Test RL agent shape compatibility."""
    try:
        from ml.rl_agent import get_rl_agent
        import numpy as np
        agent = get_rl_agent()
        # Test with 16 features (correct shape)
        state = np.zeros(16, dtype=np.float32)
        result = agent.predict(state, ensemble_signal='BUY', ensemble_confidence=60.0)
        if result is not None:
            return {'ok': True, 'detail': f'RL predict OK (16-features), action={result.action_name}'}
        return {'ok': False, 'detail': 'RL predict returned None'}
    except Exception as e:
        return {'ok': False, 'detail': str(e)}


def test_safe_pickle():
    """Test safe_pickle whitelist allows ML classes."""
    try:
        from utils.safe_pickle import RestrictedUnpickler, ALLOWED_CLASSES, SAFE_MODULE_PREFIXES
        # Check if xgboost is allowed
        xgb_ok = ('xgboost.sklearn.XGBClassifier' in ALLOWED_CLASSES
                  or 'xgboost.' in SAFE_MODULE_PREFIXES)
        if not xgb_ok:
            return {'ok': False, 'detail': 'xgboost not in whitelist'}
        return {'ok': True, 'detail': f'{len(ALLOWED_CLASSES)} classes + {len(SAFE_MODULE_PREFIXES)} prefixes'}
    except Exception as e:
        return {'ok': False, 'detail': str(e)}


def test_trade_permission():
    """Test TradePermission gate logic."""
    try:
        from risk.trade_permission import TradePermission
        tp = TradePermission()
        issues = []
        if tp.MIN_ALIGNED_FACTORS < 1:
            issues.append(f"MIN_ALIGNED_FACTORS={tp.MIN_ALIGNED_FACTORS}")
        if tp.MIN_RR < 1.0:
            issues.append(f"MIN_RR={tp.MIN_RR}")
        if tp.MIN_CONFIDENCE < 30:
            issues.append(f"MIN_CONFIDENCE={tp.MIN_CONFIDENCE}")
        if issues:
            return {'ok': None, 'detail': '; '.join(issues)}
        return {'ok': True, 'detail': f'min_factors={tp.MIN_ALIGNED_FACTORS}, min_rr={tp.MIN_RR}, min_conf={tp.MIN_CONFIDENCE}'}
    except Exception as e:
        return {'ok': False, 'detail': str(e)}


def test_decision_agent():
    """Test DecisionAgent unified voting."""
    try:
        # Read source directly (avoids import chain issues)
        src_path = os.path.join(PROJECT_ROOT, 'agents', 'decision_agent.py')
        with open(src_path, 'r') as f:
            src = f.read()
        if 'unified_signal' in src and 'unified_consensus' in src:
            return {'ok': True, 'detail': 'Unified consensus voting present'}
        return {'ok': False, 'detail': 'Unified consensus voting MISSING'}
    except Exception as e:
        return {'ok': False, 'detail': str(e)}


def test_trader_reject_method():
    """Test AITrader._reject() exists."""
    try:
        src_path = os.path.join(PROJECT_ROOT, 'core', 'trader.py')
        with open(src_path, 'r') as f:
            src = f.read()
        if 'def _reject(' in src:
            return {'ok': True, 'detail': '_reject() method present'}
        return {'ok': False, 'detail': '_reject() method MISSING'}
    except Exception as e:
        return {'ok': False, 'detail': str(e)}


def test_decision_bridge():
    """Test decision_bridge dict.reliability fix."""
    try:
        import inspect
        from analysis.decision_bridge import UnifiedToAdaptiveBridge
        src = inspect.getsource(UnifiedToAdaptiveBridge)
        if 'isinstance(p, dict)' in src or 'isinstance(best_pat, dict)' in src:
            return {'ok': True, 'detail': 'dict/object isinstance check present'}
        return {'ok': False, 'detail': 'isinstance check MISSING'}
    except Exception as e:
        return {'ok': False, 'detail': str(e)}


def test_learning_agent():
    """Test LearningAgent raw_signal field."""
    try:
        import inspect
        from agents.learning_agent import LearningAgent
        src = inspect.getsource(LearningAgent.save_decision)
        if 'raw_signal' in src:
            return {'ok': True, 'detail': 'raw_signal field present'}
        return {'ok': False, 'detail': 'raw_signal field MISSING'}
    except Exception as e:
        return {'ok': False, 'detail': str(e)}


def test_llm_keys():
    """Test LLM API keys are configured."""
    try:
        from core.llm_key_manager import get_llm_key_manager
        km = get_llm_key_manager()
        groq_keys = getattr(km, 'groq_keys', []) or getattr(km, '_groq_keys', [])
        if len(groq_keys) > 0:
            return {'ok': True, 'detail': f'{len(groq_keys)} Groq keys loaded'}
        # Try alternate check
        from config import GROQ_API_KEY
        if GROQ_API_KEY and len(GROQ_API_KEY) > 10:
            return {'ok': True, 'detail': 'GROQ_API_KEY set in config'}
        return {'ok': False, 'detail': 'No Groq keys found'}
    except Exception as e:
        # Fallback: just check env var
        groq_key = os.getenv('GROQ_API_KEY', '')
        if groq_key and len(groq_key) > 10:
            return {'ok': True, 'detail': 'GROQ_API_KEY in env'}
        return {'ok': False, 'detail': str(e)}


def test_telegram():
    """Test Telegram bot token."""
    try:
        from config import TELEGRAM_TOKEN
        if TELEGRAM_TOKEN and len(TELEGRAM_TOKEN) > 20:
            return {'ok': True, 'detail': 'Token present'}
        return {'ok': None, 'detail': 'Telegram not configured (optional)'}
    except Exception:
        token = os.getenv('TELEGRAM_TOKEN', '')
        if token and len(token) > 20:
            return {'ok': True, 'detail': 'Token in env'}
        return {'ok': None, 'detail': 'Telegram not configured (optional)'}


def test_analysis_agent_rl_shape():
    """Test analysis_agent uses [:16] not [:160]."""
    try:
        # Read source file directly (avoids import dependency issues)
        src_path = os.path.join(PROJECT_ROOT, 'agents', 'analysis_agent.py')
        with open(src_path, 'r') as f:
            src = f.read()
        if '[:16]' in src:
            return {'ok': True, 'detail': 'RL shape [:16] (correct)'}
        if '[:160]' in src:
            return {'ok': False, 'detail': 'RL shape [:160] (WRONG — should be [:16])'}
        return {'ok': None, 'detail': 'RL shape not found in source'}
    except Exception as e:
        return {'ok': False, 'detail': str(e)}


def test_execution_router_fallback():
    """Test ExecutionRouter fallback logic."""
    try:
        src_path = os.path.join(PROJECT_ROOT, 'execution', 'execution_router.py')
        with open(src_path, 'r') as f:
            src = f.read()
        if '_init_simulation_mode' in src and '_mt5_fallback_to_sim' in src:
            return {'ok': True, 'detail': 'Fallback logic present'}
        return {'ok': False, 'detail': 'Fallback logic MISSING'}
    except Exception as e:
        return {'ok': False, 'detail': str(e)}


# ─── Main ─────────────────────────────────────────────────────

args = None

def main():
    global args
    parser = argparse.ArgumentParser(description="Decision Layer Diagnostic")
    parser.add_argument('--pair', type=str, default='XAUUSD',
                        help='Pair to test (default: XAUUSD)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show full tracebacks on error')
    args = parser.parse_args()

    print(f"\n{Color.BOLD}{'='*60}{Color.END}")
    print(f"{Color.BOLD}  FOREX AI — DECISION LAYER DIAGNOSTIC{Color.END}")
    print(f"{Color.BOLD}{'='*60}{Color.END}")
    print(f"  Time : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Pair : {args.pair}")
    print(f"  Root : {PROJECT_ROOT}")

    # ── Category 1: Infrastructure ──
    print(f"\n{Color.CYAN}{Color.BOLD}━━━ 1. INFRASTRUCTURE ━━━{Color.END}")
    test_layer("Module Imports", test_imports, "infra")
    test_layer("Config Values", test_config, "infra")
    test_layer("Database Connection", test_database, "infra")
    test_layer("MT5 / Simulation", test_mt5_connection, "infra")
    test_layer("ExecutionRouter Fallback", test_execution_router_fallback, "infra")
    test_layer("LLM API Keys", test_llm_keys, "infra")
    test_layer("Telegram Bot", test_telegram, "infra")

    # ── Category 2: ML / RL ──
    print(f"\n{Color.CYAN}{Color.BOLD}━━━ 2. ML / RL PIPELINE ━━━{Color.END}")
    test_layer("Model Store Registry", test_model_store, "ml")
    test_layer("ML Models Loading", lambda: test_ml_models_loading(args), "ml")
    test_layer("ModelPredictor.is_ready()", test_model_predictor, "ml")
    test_layer("RL Agent (16-features)", test_rl_agent, "ml")
    test_layer("SafePickle Whitelist", test_safe_pickle, "ml")

    # ── Category 3: Decision Pipeline ──
    print(f"\n{Color.CYAN}{Color.BOLD}━━━ 3. DECISION PIPELINE ━━━{Color.END}")
    test_layer("DecisionAgent Voting", test_decision_agent, "decision")
    test_layer("DecisionBridge (dict fix)", test_decision_bridge, "decision")
    test_layer("AnalysisAgent RL Shape", test_analysis_agent_rl_shape, "decision")
    test_layer("TradePermission Gates", test_trade_permission, "decision")
    test_layer("AITrader._reject()", test_trader_reject_method, "decision")
    test_layer("LearningAgent raw_signal", test_learning_agent, "decision")

    # ── Summary ──
    print(f"\n{Color.BOLD}{'='*60}{Color.END}")
    print(f"{Color.BOLD}  SUMMARY{Color.END}")
    print(f"{Color.BOLD}{'='*60}{Color.END}")

    ok_count = sum(1 for r in results if r['status'] is True)
    warn_count = sum(1 for r in results if r['status'] is None)
    fail_count = sum(1 for r in results if r['status'] is False)
    total = len(results)

    print(f"  {Color.GREEN}✅ OK:    {ok_count}/{total}{Color.END}")
    print(f"  {Color.YELLOW}⚠️  WARN:  {warn_count}/{total} (optional/env-specific){Color.END}")
    print(f"  {Color.RED}❌ FAIL:  {fail_count}/{total}{Color.END}")
    print()

    # Show failures
    failures = [r for r in results if r['status'] is False]
    if failures:
        print(f"{Color.RED}{Color.BOLD}  FAILED LAYERS:{Color.END}")
        for f in failures:
            print(f"    {Color.RED}✗ {f['name']}{Color.END}")
            print(f"      {Color.GRAY}{f['detail']}{Color.END}")
        print()

    # Show warnings (env-specific, not code bugs)
    warnings = [r for r in results if r['status'] is None]
    if warnings:
        print(f"{Color.YELLOW}{Color.BOLD}  WARNINGS (env-specific, not code bugs):{Color.END}")
        for w in warnings:
            print(f"    {Color.YELLOW}⚠ {w['name']}{Color.END}")
            print(f"      {Color.GRAY}{w['detail']}{Color.END}")
        print()

    # Overall
    print(f"{'='*60}")
    # Only count real failures (not warnings)
    real_fails = [r for r in results if r['status'] is False]
    if len(real_fails) == 0:
        print(f"  {Color.GREEN}{Color.BOLD}🎉 ALL CRITICAL LAYERS OK!{Color.END}")
        print(f"  {Color.GREEN}Ready to run: python main.py{Color.END}")
    elif len(real_fails) <= 2:
        print(f"  {Color.YELLOW}{Color.BOLD}⚠️  {len(real_fails)} layer(s) need attention{Color.END}")
        print(f"  {Color.YELLOW}Bot may run but with reduced functionality{Color.END}")
    else:
        print(f"  {Color.RED}{Color.BOLD}❌ {len(real_fails)} layers FAILED — fix before running{Color.END}")
    print(f"{'='*60}\n")

    return 0 if len(real_fails) == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
