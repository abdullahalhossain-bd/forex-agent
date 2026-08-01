"""
ml/pipeline/phase16_logging.py — Pipeline Logging (Phase 16)
============================================================
Comprehensive logging for all pipeline phases: download, features,
training, validation, backtest, model selection, errors, GPU, RAM, time.
"""
from __future__ import annotations

import json, logging, os, psutil, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from ml.pipeline.utils import LOG_DIR, PipelineConfig, PipelineTimer, get_pipeline_logger

log = get_pipeline_logger("phase16_logging")

# Global pipeline run log
_run_log: Dict[str, Any] = {"phases": {}, "system": {}, "started_at": None, "completed_at": None}


def start_run_log(config: PipelineConfig) -> None:
    _run_log["started_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _run_log["config"] = config.to_dict()
    _run_log["system"] = {
        "python_version": os.sys.version.split()[0],
        "platform": os.sys.platform,
        "cpu_count": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "pid": os.getpid(),
    }


def log_phase(phase_name: str, status: str, metrics: Optional[Dict] = None, error: Optional[str] = None) -> None:
    entry = {"status": status, "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if metrics:
        entry["metrics"] = metrics
    if error:
        entry["error"] = error
    _run_log["phases"][phase_name] = entry


def get_system_stats() -> Dict[str, float]:
    """Get current RAM and (if available) GPU usage."""
    stats = {
        "ram_used_gb": round(psutil.virtual_memory().used / 1e9, 2),
        "ram_pct": psutil.virtual_memory().percent,
    }
    try:
        import torch
        if torch.cuda.is_available():
            stats["gpu_name"] = torch.cuda.get_device_name(0)
            stats["gpu_mem_used_gb"] = round(torch.cuda.memory_allocated() / 1e9, 2)
            stats["gpu_mem_total_gb"] = round(torch.cuda.get_device_properties(0).total_mem / 1e9, 2)
    except ImportError:
        pass
    return stats


def save_run_log() -> Path:
    _run_log["completed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LOG_DIR / f"pipeline_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(_run_log, indent=2, default=str))
    log.info(f"Run log saved to {out_path}")
    return out_path