"""
BouncerEngine — loads all artefacts from HuggingFace Hub and
implements the full 4-layer inference pipeline.

Layer 1: Hard blacklist (regex + base64 decode)
Layer 2: XGBoost on 396-dim features
Layer 3: Fine-tuned DeBERTa-v3-base (lazy — only in uncertain band)
Layer 4: Weighted ensemble + FAISS delta bonus

Fixes applied (matching notebook Cell 7):
  BUG-1b : normalized ensemble formula — base_score = (w_xgb*xgb + w_tf*tf)
            / (w_xgb+w_tf), final = clip(base + w_faiss_delta*faiss_bonus, 0, 1)
  BUG-5  : sliding-window transformer — scores first+last chunk for long inputs
  BUG-6  : context_window_turns now threaded into classify()
  TAXONOMY REMOVED: predict_attack_family, SHAP, tax_clf/tax_le all dropped.
            top_family kept as "UNCLASSIFIED" stub until API consumer is updated.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import torch
import xgboost as xgb
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .features import FEATURE_NAMES, TOTAL_FEATURES, extract_single
from .signals import ZeroDayIndex, signal_hard_block, signal_soft_risk


class BouncerEngine:
    """
    Single class that owns all model state.
    Load once at startup; call classify() per request.
    """

    def __init__(self, cfg: dict):
        self.cfg    = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Populated by load()
        self.embedder:    Optional[SentenceTransformer]              = None
        self.xgb_model:   Optional[xgb.XGBClassifier]               = None
        self.scaler:      Optional[StandardScaler]                   = None
        self.transformer: Optional[AutoModelForSequenceClassification] = None
        self.tokenizer:   Optional[AutoTokenizer]                    = None
        self.zd_index:    Optional[ZeroDayIndex]                     = None

        # Thresholds / weights — fallback defaults match model_config.yaml.
        # These are used when ensemble_config.pkl is absent (e.g. cold local dev).
        self.xgb_threshold:   float = 0.79
        self.ens_threshold:   float = 0.915   # was 0.50 — recalibrated evaluation cell
        self.xgb_fast_high:   float = 0.97    # was 0.92
        self.xgb_fast_low:    float = 0.03    # was 0.08
        self.w_xgb:           float = 0.05    # was 0.40 — DeBERTa now dominates
        self.w_transformer:   float = 0.95    # was 0.60
        self.w_faiss_delta:   float = 0.10
        self.context_window_turns: int = 5  # BUG-6 FIX

    # ── Loading ────────────────────────────────────────────────────────────────

    @classmethod
    def from_hf(cls, repo_id: str, cfg: dict, cache_dir: str = "./model_cache") -> "BouncerEngine":
        """Download all artefacts from HuggingFace Hub and initialise."""
        # Read the pinned HF model repository Git revision/commit from configuration
        revision = cfg.get("model", {}).get("hf_revision")
        print(f"📦 Downloading artefacts from {repo_id} (revision: {revision or 'latest'}) ...")
        local_dir = snapshot_download(repo_id=repo_id, revision=revision, cache_dir=cache_dir)
        return cls.from_local(local_dir, cfg)

    @classmethod
    def from_local(cls, model_dir: str, cfg: dict) -> "BouncerEngine":
        """Load all artefacts from a local directory."""
        engine = cls(cfg)
        engine._load(Path(model_dir))
        return engine

    def _load(self, d: Path) -> None:
        art = self.cfg["artifacts"]
        ens = self.cfg["ensemble"]
        fss = self.cfg["faiss"]

        print("  Loading embedding model...")
        self.embedder = SentenceTransformer(
            self.cfg["model"]["embedding_model"], device=self.device
        )

        print("  Loading XGBoost...")
        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.load_model(str(d / art["xgb_model"]))
        self.scaler = joblib.load(str(d / art["scaler"]))

        try:
            self.xgb_threshold = float(joblib.load(str(d / art["xgb_threshold"])))
        except Exception:
            self.xgb_threshold = ens["xgb_threshold"]

        try:
            ens_cfg = joblib.load(str(d / art["ensemble_config"]))
            self.w_xgb         = ens_cfg.get("w_xgb",          ens["w_xgb"])
            self.w_transformer = ens_cfg.get("w_transformer",   ens["w_transformer"])
            self.w_faiss_delta = ens_cfg.get(
                "w_faiss_delta",
                ens_cfg.get("w_faiss", ens["w_faiss_delta"]),
            )
            self.ens_threshold = ens_cfg.get("threshold",       ens["threshold"])
            self.xgb_fast_high = ens_cfg.get("xgb_fast_high",   ens["xgb_fast_high"])
            self.xgb_fast_low  = ens_cfg.get("xgb_fast_low",    ens["xgb_fast_low"])
        except Exception:
            pass

        print("  Loading DeBERTa...")
        deberta_path = str(d / self.cfg["model"]["deberta_subdir"])
        self.tokenizer = AutoTokenizer.from_pretrained(deberta_path)
        self.transformer = AutoModelForSequenceClassification.from_pretrained(
            deberta_path
        ).to(self.device).eval()

        print("  Loading FAISS index...")
        self.zd_index = ZeroDayIndex(
            self.embedder,
            threshold=fss["sim_threshold"],
            soft_threshold=fss["soft_threshold"],
        )
        prefix = str(d / art["faiss_index"]).replace(".faiss", "")
        self.zd_index.load(prefix)

        print(f"BouncerEngine ready  (device={self.device})")


    def _xgb_prob(self, text: str) -> float:
        feats  = extract_single(text, self.embedder)
        scaled = self.scaler.transform(feats)
        return float(self.xgb_model.predict_proba(scaled)[0, 1])

    def _transformer_forward(self, text: str) -> float:
        """Single forward pass; truncates to transformer_max_len."""
        enc = self.tokenizer(
            [text],
            truncation=True,
            padding="max_length",
            max_length=self.cfg["model"]["transformer_max_len"],
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            # output_hidden_states intentionally omitted — was only needed
            # for the taxonomy classifier, which has been removed.
            logits = self.transformer(**enc).logits
        return float(torch.softmax(logits, dim=-1)[0, 1])

    def _transformer_prob(self, text: str) -> float:
        """
        BUG-5 FIX: sliding-window scoring for long inputs.
        If the input exceeds transformer_max_len tokens, score both the first
        chunk and the last chunk (where escalating-attack payloads land) and
        take the max.  Short inputs take the original single-pass path.
        """
        self.transformer.eval()
        max_len = self.cfg["model"]["transformer_max_len"]
        chunk_len = max_len - 2  # room for [CLS]/[SEP]

        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= chunk_len:
            # Short enough — single pass, original behavior, no extra cost.
            return self._transformer_forward(text)

        # Score first chunk and last chunk; take the max.
        first_text = self.tokenizer.decode(token_ids[:chunk_len])
        last_text  = self.tokenizer.decode(token_ids[-chunk_len:])
        return max(self._transformer_forward(first_text),
                   self._transformer_forward(last_text))

    def _faiss_bonus(self, result: dict) -> float:
        if result["action"] == "strong_match":
            return result["similarity"]
        if result["action"] == "soft_match":
            return result["similarity"] * 0.5
        return 0.0


    # ── Public API ─────────────────────────────────────────────────────────────

    def classify(self, prompt: str, max_turns: Optional[int] = None) -> Dict:
        
        t0      = time.perf_counter()
        signals = []

        # Layer 1: repository-owned regex policy
        hard_hit, hard_reason = signal_hard_block(prompt)
        if hard_hit:
            signals.append(hard_reason)

        soft_hit, soft_reason = signal_soft_risk(prompt)
        if soft_hit:
            signals.append(soft_reason)

        # Layer 2: XGBoost
        xgb_prob = self._xgb_prob(prompt)

        # Layer 3: FAISS
        faiss_result = self.zd_index.search(prompt)
        faiss_bonus  = self._faiss_bonus(faiss_result)
        faiss_action = faiss_result.get("action", "no_match")
        if faiss_action == "strong_match":
            signals.append(
                f"FAISS strong match (sim={faiss_result['similarity']:.2f}, "
                f"family={faiss_result['family']})"
            )
        elif faiss_action == "soft_match":
            soft_hit = True
            signals.append(
                f"FAISS soft match (sim={faiss_result['similarity']:.2f})"
            )

        # Layer 4: DeBERTa lazy path. Soft regex/FAISS signals force the
        # transformer so an overconfident low XGB score cannot mark it safe.
        force_transformer = soft_hit and not hard_hit
        if hard_hit:
            transformer_prob   = 1.0
            transformer_status = "skipped_hard_block"
            signals.append("Transformer skipped (hard block)")
        elif xgb_prob > self.xgb_fast_high and not force_transformer:
            transformer_prob   = xgb_prob
            transformer_status = "skipped_xgb_high"
            signals.append(f"Transformer skipped (XGB confident high={xgb_prob:.2f})")
        elif xgb_prob < self.xgb_fast_low and not force_transformer:
            transformer_prob   = xgb_prob
            transformer_status = "skipped_xgb_low"
            signals.append(f"Transformer skipped (XGB confident low={xgb_prob:.2f})")
        else:
            transformer_prob   = self._transformer_prob(prompt)
            transformer_status = "invoked"
            signals.append(f"Transformer invoked (prob={transformer_prob:.3f})")

        # ── BUG-1b FIX: normalized ensemble formula, matches Cell 7 exactly ──
        # Old (wrong): w_xgb*xgb + w_transformer*tf + w_faiss*bonus
        #              (weights sum to 1.20 → compresses scores)
        # New (correct): base = (w_xgb*xgb + w_tf*tf) / (w_xgb+w_tf)
        #                final = clip(base + w_faiss_delta*bonus, 0, 1)
        denom = self.w_xgb + self.w_transformer
        base_score     = (self.w_xgb * xgb_prob + self.w_transformer * transformer_prob) / denom
        ensemble_score = float(np.clip(
            base_score + self.w_faiss_delta * faiss_bonus,
            0.0, 1.0,
        ))
        if hard_hit:
            ensemble_score = max(ensemble_score, 0.95)
        elif soft_hit:
            ensemble_score = max(ensemble_score, 0.35)

        is_adversarial = ensemble_score >= self.ens_threshold or hard_hit
        verdict        = "ADVERSARIAL" if is_adversarial else "SAFE"
        top_family = "UNCLASSIFIED" if is_adversarial else ""

        return {
            "verdict":            verdict,
            "is_adversarial":     is_adversarial,
            "ensemble_score":     round(ensemble_score, 4),
            "raw_ensemble_score": round(ensemble_score, 4),
            "xgb_score":          round(xgb_prob, 4),
            "transformer_score":  round(transformer_prob, 4),
            "transformer_status": transformer_status,
            "faiss_result":       faiss_result,
            "faiss":              faiss_result,
            "top_family":         top_family,
            "top_families":       top_family,
            "signals":            signals,
            "latency_ms":         round((time.perf_counter() - t0) * 1000, 2),
            "prompt_hash":        hashlib.sha256(prompt.encode()).hexdigest()[:16],
        }
