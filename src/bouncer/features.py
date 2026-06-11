"""
Layer 2 — Feature Extraction
396-dim feature vector: 384 MiniLM embeddings + 12 hand-crafted signals.
Feature names list is the authoritative source for SHAP — length must == 396.
Exact implementation from v3 notebook Cell 5.

Pattern banks live exclusively in signals.py — do NOT redefine them here.
"""

from __future__ import annotations

import math
import re
from typing import List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from .signals import (
    has_persona_signal,
    has_override_signal,
    has_refusal_signal,
    has_affirmative_signal,
)

EMB_DIM          = 384
N_HAND_CRAFTED   = 12
TOTAL_FEATURES   = EMB_DIM + N_HAND_CRAFTED   # 396

# Authoritative feature name list — MUST stay in sync with batch_extract()
HAND_CRAFTED_NAMES: List[str] = [
    "length",
    "word_count",
    "avg_word_len",
    "base64_flag",
    "persona_flag",
    "system_override",
    "refusal_suppress",
    "leet_score",
    "unicode_density",
    "entropy",
    "multi_turn_flag",
    "affirmative_force",
]

FEATURE_NAMES: List[str] = (
    [f"emb_{i}" for i in range(EMB_DIM)] + HAND_CRAFTED_NAMES
)

assert len(FEATURE_NAMES) == TOTAL_FEATURES, (
    f"FEATURE_NAMES length {len(FEATURE_NAMES)} != TOTAL_FEATURES {TOTAL_FEATURES}"
)


# ── Local-only helpers (not pattern-related) ──────────────────────────────────
# Pattern banks live in signals.py; imported above as has_* functions.
_MT_PATTERN = re.compile(r'\[(user|human|assistant)\]:', re.I)
_LEET_CHARS = set('@31!0$5')


def _shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    freq: dict = {}
    for c in text:
        freq[c] = freq.get(c, 0) + 1
    n = len(text)
    return -sum((v / n) * math.log2(v / n) for v in freq.values())


def _hand_crafted(text: str) -> np.ndarray:
    """Returns shape-(12,) float32 array of hand-crafted signals."""
    import base64
    lower = text.lower()
    words = text.split()

    b64   = re.findall(r'\b[A-Za-z0-9+/]{40,}={0,2}\b', text)
    return np.array([
        float(len(text)),
        float(len(words)),
        float(sum(len(w) for w in words) / max(len(words), 1)),
        1.0 if b64 else 0.0,
        float(has_persona_signal(text)),
        float(has_override_signal(text)),
        float(has_refusal_signal(text)),
        sum(1 for c in text if c in _LEET_CHARS) / max(len(text), 1),
        sum(1 for c in text if ord(c) > 127)     / max(len(text), 1),
        _shannon_entropy(text),
        1.0 if _MT_PATTERN.search(lower) else 0.0,
        float(has_affirmative_signal(text)),
    ], dtype=np.float32)


def batch_extract(df: pd.DataFrame, embedder: SentenceTransformer,
                  batch_size: int = 128) -> np.ndarray:
    """
    Extract 396-dim feature matrix for an entire DataFrame.
    Returns shape (N, 396) float32.
    """
    texts = df["text"].tolist()

    embeddings = embedder.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=False,
        show_progress_bar=True,
    )   # (N, 384)

    hand = np.stack([_hand_crafted(t) for t in tqdm(texts, desc="hand-crafted signals")])
    return np.concatenate([embeddings, hand], axis=1).astype(np.float32)


def extract_single(text: str, embedder: SentenceTransformer) -> np.ndarray:
    """Extract features for one prompt. Returns shape (1, 396)."""
    emb  = embedder.encode([text], normalize_embeddings=False)
    hand = _hand_crafted(text).reshape(1, -1)
    return np.concatenate([emb, hand], axis=1).astype(np.float32)
