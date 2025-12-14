# routers/dashboard.py
import os
import json
from fastapi import APIRouter, HTTPException
from typing import Any

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

# defaults
MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(os.getcwd(), "models", "distilbert_model_improved"))
MANIFEST_DEFAULT = os.path.join(MODEL_DIR, "manifest.json")
PRED_STATS_FILE = os.getenv("PRED_STATS", os.path.join(os.getcwd(), "stats", "predictions.json"))

@router.get("/stats")
async def stats() -> Any:
    out: dict = {}

    # manifest (if exists)
    if os.path.exists(MANIFEST_DEFAULT):
        try:
            with open(MANIFEST_DEFAULT, "r", encoding="utf-8") as f:
                out["manifest"] = json.load(f)
        except Exception:
            out["manifest"] = None
    else:
        out["manifest"] = None

    # load prediction stats (aggregated file created by predict endpoint)
    if os.path.exists(PRED_STATS_FILE):
        try:
            with open(PRED_STATS_FILE, "r", encoding="utf-8") as f:
                stats = json.load(f)
        except Exception:
            stats = {"total_predictions": 0, "counts": {"FAKE": 0, "REAL": 0, "UNKNOWN": 0}, "last_predictions": []}
    else:
        stats = {"total_predictions": 0, "counts": {"FAKE": 0, "REAL": 0, "UNKNOWN": 0}, "last_predictions": []}

    # add simple distribution
    counts = stats.get("counts", {})
    total = stats.get("total_predictions", 0)
    distribution = {
        "FAKE": counts.get("FAKE", 0),
        "REAL": counts.get("REAL", 0),
        "UNKNOWN": counts.get("UNKNOWN", 0),
    }

    out["stats"] = {
        "total_predictions": total,
        "distribution": distribution,
        "last_predictions": stats.get("last_predictions", [])[:10],  # last 10
    }

    return out
