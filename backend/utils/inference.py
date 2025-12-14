# backend/utils/inference.py
import os
import joblib
import numpy as np
from typing import List
import logging

logger = logging.getLogger("backend.inference")

# --------------------------------------------------
# Optional transformer imports (lazy)
# --------------------------------------------------
try:
    import torch
    from transformers import (
        DistilBertTokenizerFast,
        DistilBertForSequenceClassification,
    )
    TRANSFORMERS_AVAILABLE = True
except Exception as e:
    TRANSFORMERS_AVAILABLE = False
    torch = None
    DistilBertTokenizerFast = None
    DistilBertForSequenceClassification = None
    logger.exception("Transformers not available: %s", e)


class InferenceEngine:
    def __init__(
        self,
        model_dir: str,
        baseline_path: str | None = None,
    ):
        if not model_dir:
            raise ValueError("model_dir must be provided")

        self.model_dir = model_dir
        self.baseline_path = baseline_path

        # 🔒 FORCE CPU (Render has no GPU)
        self.device = "cpu"

        # --------------------------------------------------
        # Load transformer
        # --------------------------------------------------
        self.tokenizer = None
        self.model = None
        self.scaler = None

        if TRANSFORMERS_AVAILABLE:
            try:
                logger.info("Loading transformer from: %s", model_dir)

                self.tokenizer = DistilBertTokenizerFast.from_pretrained(
                    model_dir,
                    use_fast=True,
                )

                self.model = DistilBertForSequenceClassification.from_pretrained(
                    model_dir
                )

                self.model.to(self.device)
                self.model.eval()

                logger.info("✅ Transformer loaded successfully (CPU)")

            except Exception:
                logger.exception("❌ Failed to load transformer model")
                self.tokenizer = None
                self.model = None

        # --------------------------------------------------
        # Load Platt scaler (optional)
        # --------------------------------------------------
        try:
            scaler_path = os.path.join(model_dir, "platt_scaler.joblib")
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info("✅ Platt scaler loaded")
        except Exception:
            logger.exception("Platt scaler load failed")

        # --------------------------------------------------
        # Baseline model (optional fallback)
        # --------------------------------------------------
        self.baseline_artifact = None
        self.tfidf = None
        self.baseline_clf = None

        if baseline_path and os.path.exists(baseline_path):
            try:
                self.baseline_artifact = joblib.load(baseline_path)
                self.tfidf = self.baseline_artifact.get("tfidf")
                self.baseline_clf = self.baseline_artifact.get("clf")
                logger.info("✅ Baseline model loaded")
            except Exception:
                logger.exception("Baseline load failed")

    # --------------------------------------------------
    # Transformer prediction
    # --------------------------------------------------
    def predict_transformer(self, texts: List[str]):
        if not TRANSFORMERS_AVAILABLE or self.model is None or self.tokenizer is None:
            raise RuntimeError("Transformer not available")

        enc = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=192,
            return_tensors="pt",
        )

        with torch.no_grad():
            logits = self.model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
            ).logits

        probs = torch.softmax(logits, dim=1).cpu().numpy()

        # Model labels are LABEL_0 / LABEL_1
        # Convention used:
        # LABEL_0 → REAL
        # LABEL_1 → FAKE
        fake_idx = 1

        prob_fake = probs[:, fake_idx]

        # Optional Platt calibration
        if self.scaler is not None:
            try:
                prob_fake = self.scaler.predict_proba(
                    prob_fake.reshape(-1, 1)
                )[:, 1]
            except Exception:
                logger.exception("Platt scaler failed")

        labels = ["FAKE" if p >= 0.5 else "REAL" for p in prob_fake]
        return labels, prob_fake

    # --------------------------------------------------
    # Baseline fallback
    # --------------------------------------------------
    def predict_baseline(self, texts: List[str]):
        if self.baseline_clf is None or self.tfidf is None:
            raise RuntimeError("Baseline model not available")

        X = self.tfidf.transform(texts)
        prob_fake = self.baseline_clf.predict_proba(X)[:, 1]
        labels = ["FAKE" if p >= 0.5 else "REAL" for p in prob_fake]
        return labels, prob_fake


# --------------------------------------------------
# Singleton engine
# --------------------------------------------------
_engine = None


def load_engine(model_dir: str | None = None, baseline_path: str | None = None):
    global _engine

    model_dir = model_dir or os.getenv("MODEL_DIR")
    baseline_path = baseline_path or os.getenv("BASELINE_PATH")

    if not model_dir:
        raise RuntimeError("MODEL_DIR environment variable not set")

    if _engine is None:
        _engine = InferenceEngine(
            model_dir=model_dir,
            baseline_path=baseline_path,
        )

    return _engine


# --------------------------------------------------
# Public API
# --------------------------------------------------
def predict_texts(texts: List[str]):
    engine = load_engine()

    try:
        labels, prob_fake = engine.predict_transformer(texts)
    except Exception:
        logger.exception("Transformer failed, trying baseline")
        labels, prob_fake = engine.predict_baseline(texts)

    results = []
    for i, text in enumerate(texts):
        pf = float(prob_fake[i])
        pr = float(1.0 - pf)

        results.append(
            {
                "text": text,
                "label": labels[i],
                "prob_fake": pf,
                "prob_real": pr,
                "confidence": max(pf, pr),
            }
        )

    return results
