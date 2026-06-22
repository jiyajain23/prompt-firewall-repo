"""
Audit Logger — writes JSONL decision records.
Raw prompt text is NEVER stored — only sha256 hash.
"""

from __future__ import annotations

import datetime
import json
import os
import pathlib
from typing import Dict


class AuditLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = pathlib.Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log(self, request_id: str, result: Dict) -> None:
        entry = {
            "ts":           datetime.datetime.utcnow().isoformat(),
            "request_id":   request_id,
            "prompt_hash":  result.get("prompt_hash", ""),
            "verdict":      result.get("verdict", ""),
            "is_adversarial": result.get("is_adversarial", False),
            "ensemble_score": result.get("ensemble_score") or result.get("final_score"),
            "xgb_score":    result.get("xgb_score"),
            "transformer_score": result.get("transformer_score"),
            "signals":      result.get("signals", []),
            "top_families": result.get("top_families", []),
            "latency_ms":   result.get("latency_ms"),
            "stage":        result.get("stage"),         # session only
            "session_id":   result.get("session_id"),    # session only
        }
        log_file = self.log_dir / f"{datetime.date.today()}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
