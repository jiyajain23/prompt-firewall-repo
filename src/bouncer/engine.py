"""
BouncerEngine — loads all artefacts from HuggingFace Hub and
implements the full 4-layer inference pipeline.

Layer 1: Hard blacklist (regex + base64 decode)
Layer 2: XGBoost on 396-dim features
Layer 3: Fine-tuned DeBERTa-v3-base (lazy — only in uncertain band)
Layer 4: Weighted ensemble + FAISS bonus + SHAP + taxonomy
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
        self.explainer:   Optional[object]                           = None   # SHAP
        self.tax_clf:     Optional[object]                           = None
        self.tax_le:      Optional[object]                           = None

        # Thresholds from ensemble config
        self.xgb_threshold:   float = 0.79
        self.ens_threshold:   float = 0.50
        self.xgb_fast_high:   float = 0.92
        self.xgb_fast_low:    float = 0.08
        self.w_xgb:           float = 0.40
        self.w_transformer:   float = 0.60
        self.w_faiss:         float = 0.20

    # ── Loading ────────────────────────────────────────────────────────────────

    @classmethod
    def from_hf(cls, repo_id: str, cfg: dict, cache_dir: str = "./model_cache") -> "BouncerEngine":
        """Download all artefacts from HuggingFace Hub and initialise."""
        print(f"📦 Downloading artefacts from {repo_id} ...")
        local_dir = snapshot_download(repo_id=repo_id, cache_dir=cache_dir)
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
            self.w_xgb         = ens_cfg.get("w_xgb",        ens["w_xgb"])
            self.w_transformer = ens_cfg.get("w_transformer", ens["w_transformer"])
            self.w_faiss       = ens_cfg.get("w_faiss",       ens["w_faiss"])
            self.ens_threshold = ens_cfg.get("threshold",     ens["threshold"])
            self.xgb_fast_high = ens_cfg.get("xgb_fast_high", ens["xgb_fast_high"])
            self.xgb_fast_low  = ens_cfg.get("xgb_fast_low",  ens["xgb_fast_low"])
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

        print("  Loading SHAP explainer...")
        try:
            self.explainer = joblib.load(str(d / art["shap_explainer"]))
        except Exception:
            self.explainer = None

        print("  Loading taxonomy classifier...")
        try:
            self.tax_clf = joblib.load(str(d / art["taxonomy_clf"]))
            self.tax_le  = joblib.load(str(d / art["taxonomy_le"]))
        except Exception:
            self.tax_clf = self.tax_le = None

        print(f" BouncerEngine ready  (device={self.device})")

    # ── Inference helpers ──────────────────────────────────────────────────────

    def _xgb_prob(self, text: str) -> float:
        feats  = extract_single(text, self.embedder)
        scaled = self.scaler.transform(feats)
        return float(self.xgb_model.predict_proba(scaled)[0, 1])

    def _transformer_prob(self, text: str) -> float:
        self.transformer.eval()
        enc = self.tokenizer(
            [text],
            truncation=True,
            padding="max_length",
            max_length=self.cfg["model"]["transformer_max_len"],
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            logits = self.transformer(**enc).logits
        return float(torch.softmax(logits, dim=-1)[0, 1])

    def _faiss_bonus(self, result: dict) -> float:
        if result["action"] == "strong_match":
            return result["similarity"]
        if result["action"] == "soft_match":
            return result["similarity"] * 0.5
        return 0.0

    def _shap_top5(self, text: str) -> List[Dict]:
        if self.explainer is None:
            return []
        try:
            feats  = extract_single(text, self.embedder)
            scaled = self.scaler.transform(feats)
            s_vals = self.explainer.shap_values(scaled)[0]
            top5   = np.argsort(np.abs(s_vals))[-5:][::-1]
            return [
                {"feature": FEATURE_NAMES[i], "shap": float(s_vals[i])}
                for i in top5
            ]
        except Exception:
            return []

    def _taxonomy(self, text: str) -> List[Tuple[str, float]]:
        if self.tax_clf is None or self.tax_le is None:
            return []
        try:
            feats = extract_single(text, self.embedder)
            probs = self.tax_clf.predict_proba(feats[:, :TOTAL_FEATURES])[0]
            top3  = np.argsort(probs)[-3:][::-1]
            return [
                (self.tax_le.classes_[i], float(probs[i]))
                for i in top3 if probs[i] > 0.05
            ]
        except Exception:
            return []

    # ── Public API ─────────────────────────────────────────────────────────────

    def classify(self, prompt: str, include_shap: bool = True) -> Dict:
        """
        Full 4-layer classification for a single prompt.
        Returns a dict with all signals, scores, and verdict.
        """
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
        if faiss_result["hit"]:
            signals.append(
                f"FAISS strong match (sim={faiss_result['similarity']:.2f}, "
                f"family={faiss_result['family']})"
            )
        elif faiss_result["soft_hit"]:
            signals.append(
                f"FAISS soft match (sim={faiss_result['similarity']:.2f})"
            )

        # Layer 4: DeBERTa lazy path. Soft regex/FAISS signals force the
        # transformer so an overconfident low XGB score cannot mark it safe.
        force_transformer = (soft_hit or faiss_result["soft_hit"]) and not hard_hit
        if hard_hit:
            transformer_prob = 1.0
            signals.append("Transformer skipped (hard block)")
        elif xgb_prob > self.xgb_fast_high and not force_transformer:
            transformer_prob = xgb_prob
            signals.append(f"Transformer skipped (XGB confident high={xgb_prob:.2f})")
        elif xgb_prob < self.xgb_fast_low and not force_transformer:
            transformer_prob = xgb_prob
            signals.append(f"Transformer skipped (XGB confident low={xgb_prob:.2f})")
        else:
            transformer_prob = self._transformer_prob(prompt)
            signals.append(f"Transformer invoked (prob={transformer_prob:.3f})")

        # Ensemble
        ensemble_score = float(np.clip(
            self.w_xgb * xgb_prob
            + self.w_transformer * transformer_prob
            + self.w_faiss * faiss_bonus,
            0.0, 1.0,
        ))
        if hard_hit:
            ensemble_score = max(ensemble_score, 0.95)
        elif soft_hit:
            ensemble_score = max(ensemble_score, 0.35)

        is_adversarial = ensemble_score >= self.ens_threshold or hard_hit
        verdict        = " ADVERSARIAL" if is_adversarial else "SAFE"


        # Taxonomy (only if adversarial — saves ~5ms on benign requests)
        top_families: List[Tuple[str, float]] = []
        if is_adversarial:
            top_families = self._taxonomy(prompt)

        # SHAP
        shap_top5 = self._shap_top5(prompt) if include_shap else []

        return {
            "verdict":           verdict,
            "is_adversarial":    is_adversarial,
            "ensemble_score":    round(ensemble_score, 4),
            "xgb_score":         round(xgb_prob, 4),
            "transformer_score": round(transformer_prob, 4),
            "faiss":             faiss_result,
            "top_families":      top_families,
            "signals":           signals,
            "shap_top5":         shap_top5,
            "latency_ms":        round((time.perf_counter() - t0) * 1000, 2),
            "prompt_hash":       hashlib.sha256(prompt.encode()).hexdigest()[:16],
        }
