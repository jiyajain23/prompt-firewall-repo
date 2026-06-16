"""
Prompt Firewall — FastAPI Service

Endpoints:
  POST /v1/classify                    single-turn classification
  POST /v1/session/{session_id}/classify   live session (multi-turn)
  GET  /v1/session/{session_id}/summary    session risk summary
  DELETE /v1/session/{session_id}          clear session state
  GET  /health                         health + model info
"""

from contextlib import asynccontextmanager
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

logger = AuditLogger()


# ── Lifespan Configuration & Startup ──────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Load configuration
    cfg_path = os.environ.get("FIREWALL_CONFIG", "configs/model_config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    
    app.state.cfg = cfg

    # 2. Configure authentication
    api_key = os.environ.get("FIREWALL_API_KEY", "")
    require_api_key = os.environ.get(
        "REQUIRE_API_KEY",
        str(cfg.get("api", {}).get("require_api_key", False))
    ).lower() in ("true", "1", "yes")

    # Fail startup in production if require_api_key is enabled but key is empty
    if require_api_key and not api_key:
        raise ValueError(
            "CRITICAL: FIREWALL_API_KEY environment variable is missing or empty, "
            "but require_api_key is enabled."
        )

    app.state.api_key = api_key
    app.state.require_api_key = require_api_key
    app.state.rate_limit = cfg["api"]["rate_limit_per_minute"]

    # 3. Load ML models and engines (skip if testing to support fast unit tests)
    if os.environ.get("TESTING") == "1":
        app.state.engine = None
        app.state.cascade = None
    else:
        model_dir = os.environ.get("MODEL_DIR", "")
        if model_dir:
            engine = BouncerEngine.from_local(model_dir, cfg)
        else:
            hf_repo = os.environ.get("HF_REPO_ID", cfg["model"]["hf_repo_id"])
            engine = BouncerEngine.from_hf(hf_repo, cfg)

        cascade = CascadeBouncer(
            engine,
            certain_high      = cfg["session"]["cascade_certain_high"],
            certain_low       = cfg["session"]["cascade_certain_low"],
            window_size       = cfg["session"]["window_size"],
            window_stride     = cfg["session"]["window_stride"],
            window_max_chars  = cfg["session"]["window_max_chars"],
            ens_threshold     = cfg["ensemble"]["threshold"],
        )
        app.state.engine = engine
        app.state.cascade = cascade

    # 4. Initialize rate limiter backend (Redis or in-memory)
    redis_url = os.environ.get("REDIS_URL", cfg.get("api", {}).get("redis_url", ""))
    app.state.redis_url = redis_url
    if redis_url:
        try:
            import redis
            app.state.redis_client = redis.from_url(redis_url, decode_responses=True)
        except ImportError:
            raise ValueError(
                "REDIS_URL is configured, but the 'redis' package is not installed. "
                "Please run 'pip install redis' to use Redis rate limiting."
            )
    else:
        app.state.redis_client = None
        app.state.rate_buckets = defaultdict(list)
        app.state.last_cleanup_time = time.time()

    yield

    # Clean up resources
    if getattr(app.state, "redis_client", None):
        app.state.redis_client.close()


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Prompt Firewall",
    description="Adversarial prompt detection — XGBoost + DeBERTa ensemble",
    version="3.0.0",
    lifespan=lifespan,
)

# CORS configuration (restrict to required methods/headers, configurable origins via environment variable)
CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("CORS_ORIGINS", "*").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "x-api-key"],
)


def _auth(request: Request, x_api_key: str) -> None:
    api_key = getattr(request.app.state, "api_key", "")
    require_api_key = getattr(request.app.state, "require_api_key", False)

    if require_api_key:
        if not x_api_key or x_api_key != api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")
    elif api_key and x_api_key != api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _get_client_ip(request: Request) -> str:
    x_forwarded_for = request.headers.get("x-forwarded-for")
    if x_forwarded_for:
        return x_forwarded_for.split(",")[0].strip()
    x_real_ip = request.headers.get("x-real-ip")
    if x_real_ip:
        return x_real_ip.strip()
    return request.client.host if request.client else "127.0.0.1"


def _check_rate_limit(request: Request, client_ip: str) -> None:
    redis_client = getattr(request.app.state, "redis_client", None)
    rate_limit = getattr(request.app.state, "rate_limit", 120)

    if redis_client:
        current_minute = int(time.time() / 60)
        key = f"rate_limit:{client_ip}:{current_minute}"
        count = redis_client.incr(key)
        if count == 1:
            redis_client.expire(key, 60)
        if count > rate_limit:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
    else:
        now = time.time()
        buckets = request.app.state.rate_buckets

        # Periodic cleanup of all inactive buckets (once every 60 seconds)
        last_cleanup = getattr(request.app.state, "last_cleanup_time", 0.0)
        if now - last_cleanup > 60.0:
            request.app.state.last_cleanup_time = now
            for ip in list(buckets.keys()):
                pruned = [t for t in buckets[ip] if now - t < 60]
                if not pruned:
                    del buckets[ip]
                else:
                    buckets[ip] = pruned

        bucket = buckets[client_ip]
        pruned_bucket = [t for t in bucket if now - t < 60]
        if len(pruned_bucket) >= rate_limit:
            buckets[client_ip] = pruned_bucket
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        pruned_bucket.append(now)
        buckets[client_ip] = pruned_bucket


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/v1/classify", response_model=ClassifyResponse)
async def classify(
    req:         ClassifyRequest,
    request:     Request,
    x_api_key:   str = Header(default=""),
):
    _auth(request, x_api_key)
    client_ip = _get_client_ip(request)
    _check_rate_limit(request, client_ip)

    engine = request.app.state.engine
    if engine is None:
        raise HTTPException(status_code=503, detail="Model engine not initialized")

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
    _auth(request, x_api_key)
    client_ip = _get_client_ip(request)
    _check_rate_limit(request, client_ip)

    cascade = request.app.state.cascade
    if cascade is None:
        raise HTTPException(status_code=503, detail="Cascade bouncer not initialized")

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
    request:    Request,
    x_api_key:  str = Header(default=""),
):
    _auth(request, x_api_key)
    cascade = request.app.state.cascade
    if cascade is None:
        raise HTTPException(status_code=503, detail="Cascade bouncer not initialized")
    return SessionSummaryResponse(**cascade.summary(session_id))


@app.delete("/v1/session/{session_id}")
async def session_clear(
    session_id: str,
    request:    Request,
    x_api_key:  str = Header(default=""),
):
    _auth(request, x_api_key)
    cascade = request.app.state.cascade
    if cascade is None:
        raise HTTPException(status_code=503, detail="Cascade bouncer not initialized")
    cascade.clear(session_id)
    return {"cleared": session_id}


@app.get("/health", response_model=HealthResponse)
async def health(request: Request):
    engine = request.app.state.engine
    cfg = request.app.state.cfg
    return HealthResponse(
        status        = "ok",
        device        = engine.device if engine else "n/a",
        faiss_vectors = engine.zd_index.index.ntotal if (engine and engine.zd_index and engine.zd_index.index) else 0,
        model         = cfg["model"]["transformer_model"] if cfg else "n/a",
    )


@app.exception_handler(Exception)
async def _global_error(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": type(exc).__name__},
    )
