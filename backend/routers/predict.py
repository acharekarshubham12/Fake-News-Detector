# backend/routers/predict.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import os
import json
import time
import logging

# ✅ FIXED IMPORTS (package-safe)
from backend.utils.inference import predict_texts
from backend.utils.explain import generate_lime_explanation

logger = logging.getLogger("backend.predict")
router = APIRouter(prefix="/predict", tags=["predict"])

# --------------------------------------------------
# Stats file path
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATS_DIR = os.path.join(BASE_DIR, "stats")
os.makedirs(STATS_DIR, exist_ok=True)
PRED_STATS_FILE = os.getenv("PRED_STATS", os.path.join(STATS_DIR, "predictions.json"))

# --------------------------------------------------
# Request / Response models
# --------------------------------------------------
class PredictRequest(BaseModel):
    texts: List[str]

class PredictResponse(BaseModel):
    text: str
    label: str
    prob_fake: float
    prob_real: float
    confidence: float
    lime_url: str | None = None

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def _read_stats():
    try:
        with open(PRED_STATS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {
            "total_predictions": 0,
            "counts": {"FAKE": 0, "REAL": 0, "UNKNOWN": 0},
            "last_predictions": [],
        }

def _write_stats(data):
    try:
        with open(PRED_STATS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception:
        logger.exception("Failed to write stats file")

def _record_predictions(records: List[dict]):
    st = _read_stats()
    st["total_predictions"] = st.get("total_predictions", 0) + len(records)

    counts = st.get("counts", {"FAKE": 0, "REAL": 0, "UNKNOWN": 0})
    last = st.get("last_predictions", [])

    for r in records:
        lbl = r.get("label", "UNKNOWN")
        counts[lbl] = counts.get(lbl, 0) + 1
        last.insert(0, r)

    st["counts"] = counts
    st["last_predictions"] = last[:50]
    _write_stats(st)

# --------------------------------------------------
# Endpoint
# --------------------------------------------------
@router.post("/", response_model=List[PredictResponse])
def predict_endpoint(payload: PredictRequest):
    if not payload.texts:
        raise HTTPException(status_code=400, detail="No texts provided")

    preds = predict_texts(payload.texts)

    results = []
    records_to_save = []

    for p in preds:
        text = p.get("text", "")
        prob_fake = float(p.get("prob_fake", 0.0))
        prob_real = max(0.0, 1.0 - prob_fake)

        # Label from probability (consistent & safe)
        label = "FAKE" if prob_fake >= 0.5 else "REAL"

        confidence = max(prob_fake, prob_real)

        lime_url = None
        try:
            lime_url = generate_lime_explanation(text, label)
        except Exception:
            logger.exception("LIME generation failed")

        res = {
            "text": text,
            "label": label,
            "prob_fake": prob_fake,
            "prob_real": prob_real,
            "confidence": confidence,
            "lime_url": lime_url,
        }
        results.append(res)

        records_to_save.append({
            "text": text,
            "label": label,
            "prob_fake": prob_fake,
            "prob_real": prob_real,
            "confidence": confidence,
            "lime_url": lime_url,
            "timestamp": int(time.time()),
        })

    try:
        _record_predictions(records_to_save)
    except Exception:
        logger.exception("Failed to persist prediction records")

    return results
