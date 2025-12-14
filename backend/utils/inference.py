# backend/utils/inference.py
import os
import requests
import logging
from typing import List

logger = logging.getLogger("backend.inference")

HF_MODEL_ID = os.getenv("MODEL_DIR")  # e.g. ShubhamAC/fake-news-distilbert
HF_TOKEN = os.getenv("HF_API_TOKEN")

HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}


def _call_hf_api(texts: List[str]):
    payload = {"inputs": texts}

    r = requests.post(
        HF_API_URL,
        headers=HEADERS,
        json=payload,
        timeout=60,
    )

    r.raise_for_status()
    return r.json()


def predict_texts(texts: List[str]):
    if not HF_TOKEN:
        raise RuntimeError("HF_API_TOKEN not set")

    response = _call_hf_api(texts)

    results = []

    for text, output in zip(texts, response):
        # HF returns list of label/score dicts
        # Example:
        # [{'label': 'LABEL_1', 'score': 0.82}, {'label': 'LABEL_0', 'score': 0.18}]

        prob_fake = 0.0
        for item in output:
            if item["label"] in ("LABEL_1", "FAKE"):
                prob_fake = item["score"]
                break

        prob_real = 1.0 - prob_fake
        label = "FAKE" if prob_fake >= 0.5 else "REAL"

        results.append(
            {
                "text": text,
                "label": label,
                "prob_fake": float(prob_fake),
                "prob_real": float(prob_real),
                "confidence": max(prob_fake, prob_real),
            }
        )

    return results
