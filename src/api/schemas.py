"""Pydantic schemas for the FastAPI service."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=32_000)
    # include_shap removed — SHAP explainer has been dropped from the engine.


class SessionClassifyRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=32_000)
    role: str = Field(default="user")


# ShapFeature removed — TAXONOMY/SHAP machinery dropped from the engine.


class FaissResult(BaseModel):
    hit: bool
    soft_hit: bool
    similarity: float
    action: str
    match: str
    family: str


class ClassifyResponse(BaseModel):
    verdict: str
    is_adversarial: bool
    ensemble_score: float
    raw_ensemble_score: float = 0.0
    xgb_score: float
    transformer_score: float
    transformer_status: str = ""
    faiss_result: Optional[FaissResult] = None
    # Legacy field — kept until callers stop reading ["faiss"].
    faiss: Optional[FaissResult] = None
    # TAXONOMY REMOVED: top_family is a stub ("UNCLASSIFIED" or "").
    top_family: str = ""
    signals: List[str]
    latency_ms: float
    prompt_hash: str
    request_id: Optional[str] = None


class SessionClassifyResponse(BaseModel):
    # ── Non-default fields ──
    session_id: str
    turn: int
    # BUG-5 FIX: plain ASCII — no emoji that corrupt non-UTF-8 log viewers.
    verdict: str
    is_adversarial: bool
    final_score: float
    single_score: float
    traj_score: float
    stage: str
    stage2_calls: int
    flagged_turns: int
    flagged_windows: int
    signals: List[str]
    latency_ms: float
    
    window_score: Optional[float] = None
    escalation: bool = False
    escalation_reason: Optional[str] = None
    persistence: bool = False
    flags_in_window: int = 0
    top_family: str = ""
    request_id: Optional[str] = None


class SessionSummaryResponse(BaseModel):
    session_id: str
    total_turns: int
    flagged_turns: int
    flagged_windows: int
    stage2_calls: int
    stage2_rate: float
    avg_risk: float
    max_risk: float


class HealthResponse(BaseModel):
    status: str
    device: str
    faiss_vectors: int
    model: str
