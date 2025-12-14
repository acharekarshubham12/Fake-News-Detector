# backend/utils/inference.py
import os
import requests
import logging
from typing import List

logger = logging.getLogger("backend.inference")

# --------------------------------------------------
# Hugging Face configuration
# --------------------------------------------------
HF_MODEL_ID = os.getenv("MODEL_DIR")  # e.g. ShubhamAC/fake-news-distilbert
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

if not HF_MODEL_ID:
    raise RuntimeError("MODEL_DIR environment variable not set")

HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"

HEADERS = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json",
}


# --------------------------------------------------
# Call Hugging Face Inference API
# --------------------------------------------------
def _call_hf_api(texts: List[str]):
    if not HF_API_TOKEN:
        raise RuntimeError("HF_API_TOKEN not set")

    payload = {"inputs": texts}

    response = requests.post(
        HF_API_URL,
        headers=HEADERS,
        json=payload,
        timeout=60,  # cold-start safe
    )

    response.raise_for_status()
    return response.json()


# --------------------------------------------------
# Public API used by routers
# --------------------------------------------------
def predict_texts(texts: List[str]):
    """
    Always returns:
    [
      {
        "text": str,
        "label": "FAKE" | "REAL",
        "prob_fake": float,
        "prob_real": float,
        "confidence": float
      }
    ]
    """

    hf_response = _call_hf_api(texts)

    # HF may return dict on error; guard against that
    if isinstance(hf_response, dict):
        logger.error("HF API returned error: %s", hf_response)
        raise RuntimeError(f"Hugging Face API error: {hf_response}")

    results = []

    for text, output in zip(texts, hf_response):
        # output example:
        # [
        #   {"label": "LABEL_1", "score": 0.82},
        #   {"label": "LABEL_0", "score": 0.18}
        # ]

        prob_fake = 0.0

        for item in output:
            if item.get("label") in ("LABEL_1", "FAKE"):
                prob_fake = float(item.get("score", 0.0))
                break

        prob_real = 1.0 - prob_fake
        label = "FAKE" if prob_fake >= 0.5 else "REAL"
        confidence = max(prob_fake, prob_real)

        results.append(
            {
                "text": text,
                "label": label,
                "prob_fake": prob_fake,
                "prob_real": prob_real,
                "confidence": confidence,
            }
        )

    return results

