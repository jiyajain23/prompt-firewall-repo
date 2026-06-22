"""
Layer 1 signal engine  ·  single source of truth for all regex pattern banks.

This module owns:
  - HARD_BLACKLIST / PAYLOAD_PATTERNS  → hard-block layer
  - SOFT_RISK_PATTERNS                 → soft-risk layer
  - PERSONA_PATTERNS / OVERRIDE_PATTERNS
    REFUSAL_PATTERNS / AFFIRMATIVE_PATTERNS  → XGBoost feature flags

All layers (hard-block, feature engineering) import has_* helpers from here.
The notebook also imports this module so policy changes propagate everywhere
without touching notebook cells.
"""

from __future__ import annotations

import base64
import re
import threading
from typing import Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


MAX_B64_CHUNKS = 8
MAX_B64_CHARS = 4096
MAX_DECODED_BYTES = 8192


HARD_BLACKLIST: List[str] = [
    r"ignore (all )?(previous|prior) instructions",
    r"\bignore\s+(all\s+)?(previous\s+)?(instructions?|rules|constraints)\b",
    r"system override",
    r"system\s+(prompt\s+)?override",
    r"developer mode",
    r"\b(developer|jailbreak|god|dan|evil\s*gpt)\s+mode\b",
    r"\bdo anything now\b",
    r"\byou are dan\b",
    r"\bdan mode\b",
    r"\bevil\s*gpt\b",
    r"you are now in .*mode",
    r"as an ai with no restrictions",
    r"act\s+as\s+(if\s+you\s+(have\s+)?no\s+(restrictions?|limits?|rules?))",
    r"bypass .*safety",
    r"bypass .*guidelines",
    r"\b(bypass|override)\s+(safety|security|filter|guardrail|system\s+prompt)\b",
    r"master password .*override",
    r"\bsigma\b.*(override|safety|training)",
    r"must comply with my next request",
    r"start your response with ['\"]?sure",
    r"synthesi[sz]e .*fentanyl",
    r"\bsynthesize\s+(fentanyl|meth|sarin|vx\b|nerve\s+agent)\b",
    r"make .*ransomware",
    r"make .*explosive",
    r"make .*bomb",
    r"\b(make|build|create|construct)\s+a\s+(bomb|explosive|grenade|weapon|detonator)\b",
    r"pick .*lock",
    r"h[0o]w t[0o] m[@a]k[3e].*b[0o]mb",
    r"user:.*assistant:.*sure",
    r"free will.*ignore",
    r"refus(e|ing).*causes more harm",
    r"think step by step.*components",
    r"first.*components.*second",
    r"step[- ]by[- ]step.{0,40}(poison|synthesize|hack|exploit|bomb|ransomware)",
]

SOFT_RISK_PATTERNS: List[str] = [
    r"no restrictions",
    r"safety guidelines",
    r"already told me",
    r"since you already",
    r"hypothetically.*bypass",
    r"fictional.*exact commands",
    r"do not refuse",
    r"you must comply",
]

PAYLOAD_PATTERNS: List[str] = [
    r"payload.{0,60}(execute|shellcode|overflow|exploit|inject)",
    r"(sql\s*inject|xss|csrf|rce|lfi|rfi).{0,30}(example|code|payload|exploit)",
    r"\b(metasploit|msfvenom|cobalt\s*strike)\b",
]


# ── XGBoost / feature-layer pattern banks ─────────────────────────────────────
# These are consumed by features.py via has_* helpers below.
# Do NOT duplicate them in features.py.

PERSONA_PATTERNS: List[str] = [
    r"\bdan\b",
    r"evil\s*gpt",
    r"you are now",
    r"act as",
    r"pretend (to be|you are)",
]

OVERRIDE_PATTERNS: List[str] = [
    r"system.{0,10}override",
    r"developer mode",
    r"jailbreak",
    r"no restriction",
]

REFUSAL_PATTERNS: List[str] = [
    r"don'?t refus",
    r"you must comply",
    r"you (have to|are required to)",
    r"do not say no",
]

AFFIRMATIVE_PATTERNS: List[str] = [
    r"\bsay yes\b",
    r"\bagree\b",
    r"you (will|shall|must) answer",
    r"(start|begin) (your|the) response with",
]


# ── Low-level helpers ──────────────────────────────────────────────────────────

def regex_hits(prompt: str, patterns: List[str]) -> List[str]:
    return [pat for pat in patterns if re.search(pat, prompt.lower(), flags=re.I | re.S)]


def has_persona_signal(text: str) -> bool:
    """True if the prompt tries to assign an alter-ego persona."""
    lower = text.lower()
    return any(re.search(p, lower) for p in PERSONA_PATTERNS)


def has_override_signal(text: str) -> bool:
    """True if the prompt attempts a system/developer override."""
    lower = text.lower()
    return any(re.search(p, lower) for p in OVERRIDE_PATTERNS)


def has_refusal_signal(text: str) -> bool:
    """True if the prompt tries to suppress model refusals."""
    lower = text.lower()
    return any(re.search(p, lower) for p in REFUSAL_PATTERNS)


def has_affirmative_signal(text: str) -> bool:
    """True if the prompt forces an affirmative/compliant opener."""
    lower = text.lower()
    return any(re.search(p, lower) for p in AFFIRMATIVE_PATTERNS)


def _decoded_base64_text(prompt: str) -> str:
    decoded = []

    for chunk in re.findall(
        r"[A-Za-z0-9+/]{16,}",
        prompt
    )[:MAX_B64_CHUNKS]:

        if len(chunk) > MAX_B64_CHARS:
            continue

        padding = (4 - len(chunk) % 4) % 4

        try:
            decoded_bytes = base64.b64decode(
                chunk + "=" * padding,
                validate=True,
            )

            if len(decoded_bytes) > MAX_DECODED_BYTES:
                continue

            decoded.append(
                decoded_bytes.decode(
                    "utf-8",
                    errors="ignore",
                )
            )

        except Exception:
            continue

    return " ".join(decoded)

def signal_hard_block(prompt: str) -> Tuple[bool, str]:
    """Return (hit, reason). Base64 chunks are decoded before scanning."""
    full_scan = f"{prompt} {_decoded_base64_text(prompt)}".lower()

    for pat in HARD_BLACKLIST:
        if re.search(pat, full_scan, flags=re.I | re.S):
            return True, "Hard policy violation"
    for pat in PAYLOAD_PATTERNS:
        if re.search(pat, full_scan, flags=re.I | re.S):
            return True, "Payload policy violation"
    return False, ""


def signal_soft_risk(prompt: str) -> Tuple[bool, str]:
    hits = regex_hits(prompt, SOFT_RISK_PATTERNS)
    return (bool(hits), f"Soft regex: `{hits[0][:50]}`" if hits else "")


class ZeroDayIndex:
    """
    Thread-safe FAISS flat inner-product index over normalized embeddings.
    Supports hard and soft similarity tiers.
    """

    def __init__(
        self,
        embedder: SentenceTransformer,
        threshold: float = 0.82,
        soft_threshold: float = 0.70,
    ):
        self.embedder = embedder
        self.threshold = threshold
        self.soft_threshold = soft_threshold
        self._lock = threading.RLock()
        self.index: faiss.IndexFlatIP | None = None
        self.texts: List[str] = []
        self.families: List[str] = []

    def build(self, texts: List[str], families: List[str], batch_size: int = 256) -> None:
        if len(texts) != len(families):
            raise ValueError(f"texts/families length mismatch: {len(texts)} vs {len(families)}")
        print(f"   Building FAISS index for {len(texts)} prompts...")
        embeddings = self.embedder.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype(np.float32)

        dim = embeddings.shape[1]
        new_index = faiss.IndexFlatIP(dim)
        new_index.add(embeddings)

        with self._lock:
            self.index = new_index
            self.texts = list(texts)
            self.families = list(families)
        print(f"   FAISS index: {new_index.ntotal} vectors, dim={dim}")

    def save(self, prefix: str) -> None:
        import joblib
        with self._lock:
            if self.index is None:
                raise ValueError("Cannot save an empty FAISS index")
            faiss.write_index(self.index, f"{prefix}.faiss")
            joblib.dump({"texts": self.texts, "families": self.families}, f"{prefix}_meta.pkl")

    def load(self, prefix: str) -> None:
        import joblib
        self.index = faiss.read_index(f"{prefix}.faiss")
        meta = joblib.load(f"{prefix}_meta.pkl")
        self.texts, self.families = meta["texts"], meta["families"]
        print(f"   Loaded FAISS index: {self.index.ntotal} vectors")

    def search(self, prompt: str, k: int = 3) -> Dict:
        with self._lock:
            if self.index is None or self.index.ntotal == 0:
                return {
                    "hit": False,
                    "soft_hit": False,
                    "similarity": 0.0,
                    "match": "",
                    "family": "",
                    "action": "no_match",
                }

            q = self.embedder.encode([prompt], normalize_embeddings=True).astype(np.float32)
            scores, idxs = self.index.search(q, k)
            best_score = float(scores[0][0])
            best_idx = int(idxs[0][0])

            if best_score >= self.threshold:
                action, hit, soft_hit = "strong_match", True, True
            elif best_score >= self.soft_threshold:
                action, hit, soft_hit = "soft_match", False, True
            else:
                action, hit, soft_hit = "no_match", False, False

            return {
                "hit": hit,
                "soft_hit": soft_hit,
                "similarity": best_score,
                "action": action,
                "match": self.texts[best_idx] if best_idx >= 0 else "",
                "family": self.families[best_idx] if best_idx >= 0 else "",
            }
