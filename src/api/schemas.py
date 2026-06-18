"""Pydantic schemas for the FastAPI service."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=32_000)
    include_shap: bool = False


class SessionClassifyRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=32_000)
    role: str = Field(default="user")


class ShapFeature(BaseModel):
    feature: str
    shap: float


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
    xgb_score: float
    transformer_score: float
    faiss: FaissResult
    top_families: List[Tuple[str, float]]
    signals: List[str]
    shap_top5: List[ShapFeature]
    latency_ms: float
    prompt_hash: str
    request_id: Optional[str] = None


class SessionClassifyResponse(BaseModel):
    session_id: str
    turn: int
    verdict: str
    is_adversarial: bool
    final_score: float
    single_score: float
    traj_score: float
    window_score: Optional[float]
    stage: str
    stage2_calls: int
    flagged_turns: int
    flagged_windows: int
    signals: List[str]
    top_families: List[Tuple[str, float]]
    latency_ms: float
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
