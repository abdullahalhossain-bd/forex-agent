"""
verify_new_modules.py — Verify all newly added modules load and work
"""
import sys
sys.path.insert(0, '/home/z/my-project/forex_ai')

print("=" * 60)
print("  VERIFICATION: All new Day 82-89 modules")
print("=" * 60)

modules = []

# 1. Strategy Selector
try:
    from strategy.selector import StrategySelector, STRATEGY_TREND_FOLLOW, STRATEGY_SMC_PULLBACK
    sel = StrategySelector()
    choice = sel.select({
        "regime": "TRENDING", "direction": "BULLISH", "strength": "STRONG",
        "volatility": "NORMAL", "strategy": {"risk_mult": 1.0, "type": "TREND_FOLLOW"}
    })
    print(f"OK  strategy.selector         | strategy={choice['strategy']} confidence={choice['confidence']}%")
    modules.append(("strategy.selector", "OK"))
except Exception as e:
    print(f"FAIL strategy.selector         | {e}")
    modules.append(("strategy.selector", f"FAIL: {e}"))

# 2. Divergence
try:
    from analysis.divergence import DivergenceEngine
    e = DivergenceEngine()
    print(f"OK  analysis.divergence       | loaded")
    modules.append(("analysis.divergence", "OK"))
except Exception as e:
    print(f"FAIL analysis.divergence       | {e}")
    modules.append(("analysis.divergence", f"FAIL: {e}"))

# 3. Ichimoku
try:
    from analysis.ichimoku import IchimokuEngine
    e = IchimokuEngine()
    print(f"OK  analysis.ichimoku         | loaded")
    modules.append(("analysis.ichimoku", "OK"))
except Exception as e:
    print(f"FAIL analysis.ichimoku         | {e}")
    modules.append(("analysis.ichimoku", f"FAIL: {e}"))

# 4. Volatility
try:
    from analysis.volatility import VolatilityEngine
    e = VolatilityEngine()
    print(f"OK  analysis.volatility       | loaded")
    modules.append(("analysis.volatility", "OK"))
except Exception as e:
    print(f"FAIL analysis.volatility       | {e}")
    modules.append(("analysis.volatility", f"FAIL: {e}"))

# 5. Volume Profile
try:
    from analysis.volume_profile import VolumeProfileEngine
    e = VolumeProfileEngine()
    print(f"OK  analysis.volume_profile   | loaded")
    modules.append(("analysis.volume_profile", "OK"))
except Exception as e:
    print(f"FAIL analysis.volume_profile   | {e}")
    modules.append(("analysis.volume_profile", f"FAIL: {e}"))

# 6. SMC Advanced
try:
    from analysis.smc_advanced import SMCAdvancedEngine
    e = SMCAdvancedEngine()
    print(f"OK  analysis.smc_advanced     | loaded")
    modules.append(("analysis.smc_advanced", "OK"))
except Exception as e:
    print(f"FAIL analysis.smc_advanced     | {e}")
    modules.append(("analysis.smc_advanced", f"FAIL: {e}"))

# 7. Structure MTF
try:
    from analysis.structure_mtf import MTFStructureEngine
    e = MTFStructureEngine()
    print(f"OK  analysis.structure_mtf    | loaded")
    modules.append(("analysis.structure_mtf", "OK"))
except Exception as e:
    print(f"FAIL analysis.structure_mtf    | {e}")
    modules.append(("analysis.structure_mtf", f"FAIL: {e}"))

# 8. Expectancy
try:
    from risk.expectancy import ExpectancyCalculator
    calc = ExpectancyCalculator()
    r = calc.calculate_from_pnls([100, -50, 200, -30, 150, -40])
    print(f"OK  risk.expectancy           | E={r['expectancy']:.2f} quality={r['system_quality']}")
    modules.append(("risk.expectancy", "OK"))
except Exception as e:
    print(f"FAIL risk.expectancy           | {e}")
    modules.append(("risk.expectancy", f"FAIL: {e}"))

# 9. Patch analytics
try:
    from risk.expectancy import patch_analytics_expectancy
    patch_analytics_expectancy()
    from analytics.analytics import PerformanceAnalyzer
    print(f"OK  analytics patch           | PerformanceAnalyzer patched")
    modules.append(("analytics patch", "OK"))
except Exception as e:
    print(f"FAIL analytics patch           | {e}")
    modules.append(("analytics patch", f"FAIL: {e}"))

# Summary
print()
print("=" * 60)
ok = sum(1 for _, s in modules if s == "OK")
print(f"  RESULT: {ok}/{len(modules)} modules OK")
print("=" * 60)
for name, status in modules:
    icon = "[OK]" if status == "OK" else "[FAIL]"
    print(f"  {icon:<7} {name:<28} {status}")
