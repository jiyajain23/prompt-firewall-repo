"""
Prompt Firewall — FastAPI Service

Endpoints:
  POST /v1/classify                    single-turn classification
  POST /v1/session/{session_id}/classify   live session (multi-turn)
  GET  /v1/session/{session_id}/summary    session risk summary
  DELETE /v1/session/{session_id}          clear session state
  GET  /health                         health + model info
"""

from __future__ import annotations

import os
import time
import uuid
from collections import defaultdict
from typing import Dict

import yaml
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..bouncer.engine import BouncerEngine
from ..session.cascade import CascadeBouncer
from .logging import AuditLogger
from .schemas import (
    ClassifyRequest,
    ClassifyResponse,
    HealthResponse,
    SessionClassifyRequest,
    SessionClassifyResponse,
    SessionSummaryResponse,
)

# ── Load config ────────────────────────────────────────────────────────────────
_CFG_PATH = os.environ.get("FIREWALL_CONFIG", "configs/model_config.yaml")
with open(_CFG_PATH) as f:
    CFG = yaml.safe_load(f)

API_KEY    = os.environ.get("FIREWALL_API_KEY", "")
HF_REPO    = os.environ.get("HF_REPO_ID", CFG["model"]["hf_repo_id"])
MODEL_DIR  = os.environ.get("MODEL_DIR", "")      # if set, load from local dir instead of HF
RATE_LIMIT = CFG["api"]["rate_limit_per_minute"]

# ── Load models (once, at startup) ────────────────────────────────────────────
if MODEL_DIR:
    engine = BouncerEngine.from_local(MODEL_DIR, CFG)
else:
    engine = BouncerEngine.from_hf(HF_REPO, CFG)

cascade   = CascadeBouncer(
    engine,
    certain_high      = CFG["session"]["cascade_certain_high"],
    certain_low       = CFG["session"]["cascade_certain_low"],
    window_size       = CFG["session"]["window_size"],
    window_stride     = CFG["session"]["window_stride"],
    window_max_chars  = CFG["session"]["window_max_chars"],
    ens_threshold     = CFG["ensemble"]["threshold"],
)
logger = AuditLogger()

# ── Rate limiting (in-memory, per IP) ─────────────────────────────────────────
_rate_buckets: Dict[str, list] = defaultdict(list)

def _check_rate_limit(client_ip: str) -> None:
    now    = time.time()
    bucket = _rate_buckets[client_ip]
    # Drop timestamps older than 60s
    _rate_buckets[client_ip] = [t for t in bucket if now - t < 60]
    if len(_rate_buckets[client_ip]) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    _rate_buckets[client_ip].append(now)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Prompt Firewall",
    description="Adversarial prompt detection — XGBoost + DeBERTa ensemble",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _auth(x_api_key: str) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/v1/classify", response_model=ClassifyResponse)
async def classify(
    req:         ClassifyRequest,
    request:     Request,
    x_api_key:   str = Header(default=""),
):
    _auth(x_api_key)
    _check_rate_limit(request.client.host)

    result     = engine.classify(req.prompt, include_shap=req.include_shap)
    request_id = str(uuid.uuid4())
    logger.log(request_id, result)

    return ClassifyResponse(request_id=request_id, **result)


@app.post(
    "/v1/session/{session_id}/classify",
    response_model=SessionClassifyResponse,
)
async def session_classify(
    session_id:  str,
    req:         SessionClassifyRequest,
    request:     Request,
    x_api_key:   str = Header(default=""),
):
    _auth(x_api_key)
    _check_rate_limit(request.client.host)

    result     = cascade.score_turn(session_id, req.content, req.role)
    request_id = str(uuid.uuid4())
    logger.log(request_id, result)

    return SessionClassifyResponse(
        request_id=request_id,
        **{k: v for k, v in result.items() if k != "prompt_hash"},
    )


@app.get(
    "/v1/session/{session_id}/summary",
    response_model=SessionSummaryResponse,
)
async def session_summary(
    session_id: str,
    x_api_key:  str = Header(default=""),
):
    _auth(x_api_key)
    return SessionSummaryResponse(**cascade.summary(session_id))


@app.delete("/v1/session/{session_id}")
async def session_clear(
    session_id: str,
    x_api_key:  str = Header(default=""),
):
    _auth(x_api_key)
    cascade.clear(session_id)
    return {"cleared": session_id}


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status        = "ok",
        device        = engine.device,
        faiss_vectors = engine.zd_index.index.ntotal if engine.zd_index.index else 0,
        model         = CFG["model"]["transformer_model"],
    )


@app.exception_handler(Exception)
async def _global_error(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": type(exc).__name__},
    )
