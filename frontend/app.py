# frontend/app.py
import os
import json
from typing import List, Dict, Any
import requests
import streamlit as st
from datetime import datetime
import pandas as pd

# --------------------------------------------------
# Config
# --------------------------------------------------
DEFAULT_BACKEND = "https://fake-news-backend-vyaz.onrender.com"
BACKEND_URL = os.getenv("STREAMLIT_API_URL", DEFAULT_BACKEND)

TIMEOUT = 30  # backend cold-start safe

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Fake News Detector — Demo",
    page_icon="📰",
    layout="wide",
)

# --------------------------------------------------
# Styles
# --------------------------------------------------
st.markdown(
    """
    <style>
    .stButton>button {
        background-color:#0b6cff;
        color:white;
        border-radius:10px;
        padding:8px 18px;
        font-weight:600;
    }
    .card {
        background:#ffffff;
        padding:16px;
        border-radius:12px;
        box-shadow:0 6px 22px rgba(11,108,255,0.08);
        margin-bottom:16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Backend helpers
# --------------------------------------------------
def fetch_health():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=10)
        r.raise_for_status()
        return True, r.json()
    except Exception:
        return False, None


def call_predict(texts: List[str], generate_lime: bool):
    payload = {"texts": texts}
    params = {
        "prefer": "Auto (fast)",
        "explain": str(generate_lime).lower(),
    }

    r = requests.post(
        f"{BACKEND_URL}/predict",
        json=payload,
        params=params,
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.title("📰 Fake News Detector")
    st.caption("Streamlit frontend for FastAPI backend")

    ok, health = fetch_health()
    if ok:
        st.success("Backend connected")
        st.write("Model:", health.get("model_dir"))
    else:
        st.error("Backend unreachable (cold start?)")

    st.markdown("---")
    generate_lime = st.checkbox(
        "Generate LIME explanations (slow)",
        value=False,
    )

    st.markdown("---")
    st.markdown(f"[API Docs]({BACKEND_URL}/docs)")


# --------------------------------------------------
# Main UI
# --------------------------------------------------
st.header("Fake News Detection")

text_input = st.text_area(
    "Enter one or more texts (one per line):",
    height=220,
    value=(
        "Scientists Confirm the Moon Is Slowly Turning Into a Giant Diamond\n"
        "RBI Raises Growth Forecast for India to 7.2% for FY2025\n"
        "Government to Ban All Smartphones by 2026"
    ),
)

run = st.button("Predict")

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if run:
    texts = [t.strip() for t in text_input.splitlines() if t.strip()]

    if not texts:
        st.warning("Please enter some text.")
    else:
        with st.spinner("Calling backend (may take ~30s on first request)..."):
            try:
                results = call_predict(texts, generate_lime)
            except Exception as e:
                st.error(f"Backend request failed: {e}")
                st.stop()

        st.success("Predictions received")

        # --------------------------------------------------
        # Display results
        # --------------------------------------------------
        for i, r in enumerate(results):
            st.markdown(f"### Example {i+1}")
            st.write(r["text"])

            prob_fake = float(r["prob_fake"])
            prob_real = float(r["prob_real"])
            confidence = float(r["confidence"])
            label = r["label"]

            st.markdown('<div class="card">', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)

            color = "#ef4444" if label == "FAKE" else "#0b6cff"
            c1.markdown(
                f"<h3 style='color:{color}'>{label}</h3>",
                unsafe_allow_html=True,
            )
            c2.metric("Prob Fake", f"{prob_fake:.3f}")
            c3.metric("Prob Real", f"{prob_real:.3f}")
            c4.metric("Confidence", f"{confidence:.3f}")
            st.markdown("</div>", unsafe_allow_html=True)

            if r.get("lime_url"):
                url = (
                    r["lime_url"]
                    if r["lime_url"].startswith("http")
                    else BACKEND_URL + r["lime_url"]
                )
                st.markdown(f"[🔍 LIME Explanation]({url})")

            st.markdown("---")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.caption("Built with FastAPI + Hugging Face + Streamlit")

