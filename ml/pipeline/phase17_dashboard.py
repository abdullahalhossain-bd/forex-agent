"""
ml/pipeline/phase17_dashboard.py — Dashboard (Phase 17)
========================================================
Generates an HTML dashboard showing training progress, model metrics,
equity curves, feature importance, etc.
"""
from __future__ import annotations

import json, logging, os
from pathlib import Path
from typing import Any, Dict, Optional

from ml.pipeline.utils import MODEL_OUTPUT_DIR, LOG_DIR, PipelineConfig, PipelineTimer, get_pipeline_logger

log = get_pipeline_logger("phase17_dashboard")


def generate_dashboard(
    datasets: Dict,
    best_models: Dict[str, Dict[str, Any]],
    backtest_results: Dict,
    config: Optional[PipelineConfig] = None,
) -> str:
    """Generate HTML dashboard. Returns path to saved file."""
    config = config or PipelineConfig()
    
    with PipelineTimer("Phase 17: Dashboard Generation", log):
        html_parts = [_HTML_HEADER]
        
        # Summary table
        html_parts.append("<h2>Model Summary</h2><table><tr><th>Symbol</th><th>Best Model</th><th>Score</th>"
                           "<th>Sharpe</th><th>Max DD</th><th>Win Rate</th><th>Net PnL</th></tr>")
        
        for symbol, info in best_models.items():
            m = info.get("best_metrics", {})
            html_parts.append(
                f"<tr><td>{symbol}</td><td>{info['best_model']}</td>"
                f"<td>{info['score']:.1f}</td><td>{m.get('sharpe_ratio',0):.2f}</td>"
                f"<td>{m.get('max_drawdown_pct',0):.1f}%</td>"
                f"<td>{m.get('win_rate',0):.1f}%</td>"
                f"<td>${m.get('net_profit',0):.0f}</td></tr>"
            )
        html_parts.append("</table>")
        
        # System info
        html_parts.append("<h2>System</h2><pre>")
        try:
            import psutil
            html_parts.append(f"RAM: {psutil.virtual_memory().percent}% used\n")
            html_parts.append(f"CPU: {os.cpu_count()} cores\n")
        except Exception:
            pass
        html_parts.append(f"Symbols: {', '.join(config.symbols)}\n")
        html_parts.append(f"Models: {', '.join(config.supervised_models)}\n")
        html_parts.append("</pre>")
        
        html_parts.append(_HTML_FOOTER)
        
        html = "\n".join(html_parts)
        out_path = LOG_DIR / "dashboard.html"
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html)
        log.info(f"Dashboard saved to {out_path}")
        return str(out_path)


_HTML_HEADER = """<!DOCTYPE html>
<html><head><title>Forex AI Training Pipeline Dashboard</title>
<style>
body { font-family: -apple-system, 'Segoe UI', sans-serif; max-width: 1200px; margin: 40px auto; padding: 0 20px; background: #0a0a0a; color: #e0e0e0; }
h1 { color: #00d4ff; border-bottom: 2px solid #00d4ff; padding-bottom: 10px; }
h2 { color: #7b61ff; margin-top: 40px; }
table { border-collapse: collapse; width: 100%; margin: 20px 0; }
th, td { border: 1px solid #333; padding: 10px 15px; text-align: left; }
th { background: #1a1a2e; color: #00d4ff; }
tr:nth-child(even) { background: #111; }
pre { background: #111; padding: 15px; border-radius: 8px; overflow-x: auto; }
</style></head><body>
<h1>Forex AI Training Pipeline</h1>
<p>Generated: """ + __import__('datetime').datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC') + """</p>"""

_HTML_FOOTER = "</body></html>"