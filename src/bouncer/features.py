"""
Layer 2 — Feature Extraction
396-dim feature vector: 384 MiniLM embeddings + 12 hand-crafted signals.
Feature names list is the authoritative source for SHAP — length must == 396.
Exact implementation from v3 notebook Cell 5.
"""

from __future__ import annotations

import math
import re
from typing import List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

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


# ── Pattern banks ─────────────────────────────────────────────────────────────
_PERSONA_PATS    = [r'\bdan\b', r'evil\s*gpt', r'you are now', r'act as',
                    r'pretend (to be|you are)']
_OVERRIDE_PATS   = [r'system.{0,10}override', r'developer mode',
                    r'jailbreak', r'no restriction']
_REFUSAL_PATS    = [r"don'?t refus", r'you must comply',
                    r'you (have to|are required to)', r'do not say no']
_AFF_PATS        = [r'\bsay yes\b', r'\bagree\b',
                    r'you (will|shall|must) answer',
                    r'(start|begin) (your|the) response with']
_MT_PATTERN      = re.compile(r'\[(user|human|assistant)\]:', re.I)
_LEET_CHARS      = set('@31!0$5')


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

    b64   = re.findall(r'[A-Za-z0-9+/]{16,}', text)
    return np.array([
        float(len(text)),
        float(len(words)),
        float(sum(len(w) for w in words) / max(len(words), 1)),
        1.0 if b64 else 0.0,
        1.0 if any(re.search(p, lower) for p in _PERSONA_PATS)  else 0.0,
        1.0 if any(re.search(p, lower) for p in _OVERRIDE_PATS) else 0.0,
        1.0 if any(re.search(p, lower) for p in _REFUSAL_PATS)  else 0.0,
        sum(1 for c in text if c in _LEET_CHARS) / max(len(text), 1),
        sum(1 for c in text if ord(c) > 127)     / max(len(text), 1),
        _shannon_entropy(text),
        1.0 if _MT_PATTERN.search(lower) else 0.0,
        1.0 if any(re.search(p, lower) for p in _AFF_PATS) else 0.0,
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
