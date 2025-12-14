# backend/utils/inference.py
import os
import joblib
import numpy as np
import logging
import requests
from typing import List

logger = logging.getLogger("backend.inference")

# --------------------------------------------------
# Environment (HF)
# --------------------------------------------------
HF_MODEL_ID = os.getenv("HF_MODEL_ID")          # e.g. "username/fake-news-distilbert"
HF_API_TOKEN = os.getenv("HF_API_TOKEN")        # hf_xxx
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}" if HF_MODEL_ID else None

USE_HF_API = bool(HF_MODEL_ID and HF_API_TOKEN)

# --------------------------------------------------
# Optional local transformer imports (dev only)
# --------------------------------------------------
try:
    import torch
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False
    torch = None
    DistilBertTokenizerFast = None
    DistilBertForSequenceClassification = None


class InferenceEngine:
    def __init__(
        self,
        model_dir: str | None = None,
        baseline_path: str | None = None,
        device: str | None = None,
    ):
        self.model_dir = model_dir
        self.baseline_path = baseline_path
        self.device = device or (
            "cuda" if TRANSFORMERS_AVAILABLE and torch and torch.cuda.is_available() else "cpu"
        )

        # --------------------------------------------------
        # Local transformer (DEV ONLY)
        # --------------------------------------------------
        self.tokenizer = None
        self.model = None
        self.scaler = None

        if not USE_HF_API and model_dir and TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
                self.model = DistilBertForSequenceClassification.from_pretrained(model_dir)
                self.model.to(self.device)
                self.model.eval()
                logger.info("Local transformer loaded on %s", self.device)
            except Exception:
                logger.exception("Failed to load local transformer")

        # --------------------------------------------------
        # Platt scaler (optional)
        # --------------------------------------------------
        if model_dir:
            try:
                scaler_path = os.path.join(model_dir, "platt_scaler.joblib")
                if os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)
                    logger.info("Platt scaler loaded")
            except Exception:
                logger.exception("Failed to load Platt scaler")

        # --------------------------------------------------
        # Baseline model (optional)
        # --------------------------------------------------
        self.baseline_artifact = None
        self.tfidf = None
        self.baseline_clf = None

        if baseline_path and os.path.exists(baseline_path):
            try:
                self.baseline_artifact = joblib.load(baseline_path)
                self.tfidf = self.baseline_artifact.get("tfidf")
                self.baseline_clf = self.baseline_artifact.get("clf")
                logger.info("Baseline model loaded")
            except Exception:
                logger.exception("Failed to load baseline model")

    # --------------------------------------------------
    # Hugging Face Inference API (PROD)
    # --------------------------------------------------
    def predict_hf_api(self, texts: List[str]):
        headers = {
            "Authorization": f"Bearer {HF_API_TOKEN}",
            "Content-Type": "application/json",
        }

        results_labels = []
        results_probs = []

        for text in texts:
            payload = {"inputs": text}
            resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            outputs = resp.json()

            # HF returns list of {label, score}
            # Example:
            # [{"label":"LABEL_1","score":0.67},{"label":"LABEL_0","score":0.33}]
            scores = {o["label"]: o["score"] for o in outputs}

            prob_fake = scores.get("FAKE") or scores.get("LABEL_1") or 0.0
            prob_fake = float(prob_fake)

            label = "FAKE" if prob_fake >= 0.5 else "REAL"

            results_labels.append(label)
            results_probs.append(prob_fake)

        return results_labels, np.array(results_probs)

    # --------------------------------------------------
    # Local transformer prediction (DEV)
    # --------------------------------------------------
    def predict_local_transformer(self, texts: List[str]):
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Local transformer not available")

        enc = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=192,
            return_tensors="pt",
        )

        with torch.no_grad():
            logits = self.model(
                input_ids=enc["input_ids"].to(self.device),
                attention_mask=enc["attention_mask"].to(self.device),
            ).logits

        probs_all = torch.softmax(logits, dim=1).cpu().numpy()

        id2label = {int(k): v for k, v in self.model.config.id2label.items()}

        # LABEL_1 = FAKE (training assumption)
        fake_idx = next(
            (idx for idx, name in id2label.items() if name.upper() in ("FAKE", "LABEL_1")),
            None,
        )
        if fake_idx is None:
            raise RuntimeError("FAKE label not found")

        prob_fake = probs_all[:, fake_idx]

        if self.scaler is not None:
            try:
                prob_fake = self.scaler.predict_proba(prob_fake.reshape(-1, 1))[:, 1]
            except Exception:
                logger.exception("Platt scaler failed")

        labels = ["FAKE" if p >= 0.5 else "REAL" for p in prob_fake]
        return labels, prob_fake

    # --------------------------------------------------
    # Baseline fallback
    # --------------------------------------------------
    def predict_baseline(self, texts: List[str]):
        if self.baseline_artifact is None:
            raise RuntimeError("Baseline not available")

        X = self.tfidf.transform(texts)
        prob_fake = self.baseline_clf.predict_proba(X)[:, 1]
        labels = ["FAKE" if p >= 0.5 else "REAL" for p in prob_fake]
        return labels, prob_fake


# --------------------------------------------------
# Singleton engine
# --------------------------------------------------
_engine = None


def load_engine():
    global _engine
    if _engine is None:
        _engine = InferenceEngine(
            model_dir=os.getenv("MODEL_DIR"),
            baseline_path=os.getenv("BASELINE_PATH"),
        )
    return _engine


# --------------------------------------------------
# Public API
# --------------------------------------------------
def predict_texts(texts: List[str]):
    engine = load_engine()

    try:
        if USE_HF_API:
            labels, prob_fake = engine.predict_hf_api(texts)
        else:
            labels, prob_fake = engine.predict_local_transformer(texts)
    except Exception:
        logger.exception("Primary model failed; using baseline")
        labels, prob_fake = engine.predict_baseline(texts)

    results = []
    for i, text in enumerate(texts):
        pf = float(prob_fake[i])
        pr = 1.0 - pf
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
