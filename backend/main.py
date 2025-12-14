# backend/main.py
import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backend.main")

app = FastAPI(title="FakeNews Detection API", version="1.0")

# --------------------------------------------------
# CORS
# --------------------------------------------------
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*")
if ALLOW_ORIGINS == "*":
    allow_origins = ["*"]
else:
    allow_origins = [o.strip() for o in ALLOW_ORIGINS.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Static files (for LIME explanations)
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
EXPLAIN_DIR = os.path.join(STATIC_DIR, "explanations")

os.makedirs(EXPLAIN_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
logger.info("Mounted explanations directory at /static")

# --------------------------------------------------
# Routers
# --------------------------------------------------
try:
    # ✅ FIXED IMPORTS (package-safe)
    from backend.routers.predict import router as predict_router
    from backend.routers.dashboard import router as dashboard_router

    app.include_router(predict_router)    # prefix="/predict"
    app.include_router(dashboard_router)  # prefix="/dashboard"

    logger.info("Routers loaded successfully.")

except Exception as e:
    logger.exception("Failed to import routers; using fallback endpoints.")

    from fastapi import APIRouter

    fallback = APIRouter()

    @fallback.post("/predict")
    async def predict_fallback(payload: dict):
        return {"error": "predict router not available", "payload": payload}

    @fallback.get("/dashboard/stats")
    async def dashboard_fallback():
        return {"error": "dashboard router not available"}

    app.include_router(fallback)

# --------------------------------------------------
# Health
# --------------------------------------------------
@app.get("/health", tags=["health"])
async def health():
    return {
        "status": "ok",
        "model_dir": os.getenv("MODEL_DIR", "not-set"),
        "baseline_path": os.getenv("BASELINE_PATH", "not-set"),
    }

# --------------------------------------------------
# Startup
# --------------------------------------------------
@app.on_event("startup")
async def startup_checks():
    logger.info(
        f"Starting FakeNews API. "
        f"MODEL_DIR={os.getenv('MODEL_DIR')}, "
        f"BASELINE_PATH={os.getenv('BASELINE_PATH')}"
    )
    # Optional: warm-load models here if needed

# --------------------------------------------------
# Local run (optional)
# --------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
    )
