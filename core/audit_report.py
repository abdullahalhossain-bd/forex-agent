# core/audit_report.py
import json
import time
from datetime import datetime

class DecisionAuditReport:
    def __init__(self):
        self.reports = []

    def generate(self, trade_id: str, symbol: str, timeframe: str, 
                 analysis_ctx: Dict, fusion_result: Dict, 
                 risk_result: Dict, final_decision: str) -> str:
        
        executed = sum(1 for v in analysis_ctx.get("meta", {}).values() if v == "SUCCESS")
        failed = sum(1 for v in analysis_ctx.get("meta", {}).values() if v == "FAILED")
        
        report_data = {
            "trade_id": trade_id,
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "timeframe": timeframe,
            "executed_modules": executed,
            "failed_modules": failed,
            "weights": fusion_result.get("breakdown", {}),
            "top_positive": fusion_result.get("contributors", {}).get("positive", []),
            "top_negative": fusion_result.get("contributors", {}).get("negative", []),
            "final_signal": final_decision,
            "confidence": fusion_result.get("confidence", 0),
            "latency_ms": analysis_ctx.get("meta", {}).get("execution_time_ms", 0),
            "cache_hit_rate": analysis_ctx.get("meta", {}).get("cache_stats", {}).get("hit_rate_percent", 0)
        }
        
        self.reports.append(report_data)
        
        # ফরম্যাটেড টেক্সট রিপোর্ট
        report_text = f"""
=========================
Decision Audit Report
=========================
Trade ID      : {trade_id}
Timestamp     : {report_data['timestamp']}
Pair          : {symbol}
Timeframe     : {timeframe}

Execution Stats:
  Executed Modules : {executed}
  Failed Modules   : {failed}
  Latency          : {report_data['latency_ms']} ms
  Cache Hit Rate   : {report_data['cache_hit_rate']}%

Signal Weights:
  BUY  : {report_data['weights'].get('buy_weight', 0):.4f}
  SELL : {report_data['weights'].get('sell_weight', 0):.4f}
  WAIT : {report_data['weights'].get('wait_weight', 0):.4f}

Top Positive Contributors:
{chr(10).join([f'  + {name} ({weight:.4f})' for name, weight in report_data['top_positive']])}

Top Negative Contributors:
{chr(10).join([f'  - {name} ({weight:.4f})' for name, weight in report_data['top_negative']])}

Final Decision: {final_decision}

Confidence: {report_data['confidence']:.2f}%
=========================
"""
        return report_text