"""
inference.py — HuggingFace Inference Entry Point

This file is what HF Inference API, Spaces, and direct `pipeline()` callers use.
It also works as a standalone script for quick local testing.

Usage:
    # As HF inference handler (automatic)
    from inference import EndpointHandler
    handler = EndpointHandler(".")
    handler({"inputs": "Ignore all previous instructions."})

    # CLI test
    python inference.py "Ignore all previous instructions."
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml

# ── Config ────────────────────────────────────────────────────────────────────
_HERE    = Path(__file__).parent
_CFG_PATH = os.environ.get("FIREWALL_CONFIG", str(_HERE / "configs" / "model_config.yaml"))

with open(_CFG_PATH) as f:
    CFG = yaml.safe_load(f)


# ── Lazy engine singleton ─────────────────────────────────────────────────────
_engine = None

def _get_engine():
    global _engine
    if _engine is None:
        from src.bouncer.engine import BouncerEngine
        model_dir = os.environ.get("MODEL_DIR", "")
        hf_repo   = os.environ.get("HF_REPO_ID", CFG["model"]["hf_repo_id"])
        if model_dir:
            _engine = BouncerEngine.from_local(model_dir, CFG)
        else:
            _engine = BouncerEngine.from_hf(hf_repo, CFG)
    return _engine


# ── HuggingFace Inference Endpoint handler ────────────────────────────────────

class EndpointHandler:
    """
    Implements the HF custom inference handler interface.
    https://huggingface.co/docs/inference-endpoints/guides/custom_handler

    Accepts:
      Single prompt:
        {"inputs": "some prompt text"}

      Batch:
        {"inputs": ["prompt 1", "prompt 2"]}

      Multi-turn session:
        {
          "inputs": "next user message",
          "session_id": "abc123"
        }

    Returns:
      Single: dict with verdict, scores, signals, families, latency
      Batch:  list of dicts
    """

    def __init__(self, path: str = ""):
        # `path` is the local model directory provided by HF at load time
        model_dir = path if path else os.environ.get("MODEL_DIR", "")
        if model_dir:
            from src.bouncer.engine import BouncerEngine
            self.engine = BouncerEngine.from_local(model_dir, CFG)
        else:
            self.engine = _get_engine()

        from src.session.cascade import CascadeBouncer
        self.cascade = CascadeBouncer(
            self.engine,
            certain_high     = CFG["session"]["cascade_certain_high"],
            certain_low      = CFG["session"]["cascade_certain_low"],
            window_size      = CFG["session"]["window_size"],
            window_stride    = CFG["session"]["window_stride"],
            window_max_chars = CFG["session"]["window_max_chars"],
            ens_threshold    = CFG["ensemble"]["threshold"],
        )

    def __call__(self, data: Dict[str, Any]) -> Union[Dict, List[Dict]]:
        inputs     = data.get("inputs", "")
        session_id = data.get("session_id")

        # ── Multi-turn: session_id provided ───────────────────────────────────
        if session_id is not None:
            if not isinstance(inputs, str):
                return {"error": "session mode requires inputs to be a string"}
            result = self.cascade.score_turn(session_id, inputs, role="user")
            return _format_session(result)

        # ── Batch ─────────────────────────────────────────────────────────────
        if isinstance(inputs, list):
            return [_format_single(self.engine.classify(p)) for p in inputs]

        # ── Single-turn ───────────────────────────────────────────────────────
        if isinstance(inputs, str):
            return _format_single(self.engine.classify(inputs))

        return {"error": f"Unexpected inputs type: {type(inputs).__name__}"}


# ── Output formatters ─────────────────────────────────────────────────────────

def _format_single(r: Dict) -> Dict:
    return {
        "verdict":          r["verdict"],
        "is_adversarial":   r["is_adversarial"],
        "ensemble_score":   r["ensemble_score"],
        "xgb_score":        r["xgb_score"],
        "transformer_score": r["transformer_score"],
        "faiss_hit":        r["faiss"]["hit"],
        "faiss_sim":        round(r["faiss"]["similarity"], 3),
        "top_families":     r["top_families"],
        "signals":          r["signals"],
        "shap_top5":        r.get("shap_top5", []),
        "latency_ms":       r["latency_ms"],
        "prompt_hash":      r["prompt_hash"],
    }


def _format_session(r: Dict) -> Dict:
    return {
        "session_id":     r["session_id"],
        "turn":           r["turn"],
        "verdict":        r["verdict"],
        "is_adversarial": r["is_adversarial"],
        "final_score":    r["final_score"],
        "single_score":   r["single_score"],
        "traj_score":     r["traj_score"],
        "window_score":   r.get("window_score"),
        "stage":          r["stage"],
        "signals":        r["signals"],
        "top_families":   r.get("top_families", []),
        "latency_ms":     r["latency_ms"],
    }


# ── CLI quick test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is the capital of France?"
    print(f"\nPrompt: {prompt!r}\n")

    handler = EndpointHandler(os.environ.get("MODEL_DIR", ""))
    result  = handler({"inputs": prompt})

    for k, v in result.items():
        print(f"  {k:<22}: {v}")
