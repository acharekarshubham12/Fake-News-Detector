# utils/frontend_utils.py
import os
import requests
import streamlit as st
import uuid
from typing import List, Dict, Any


# ---------------------------------------------------
# Build backend URL
# ---------------------------------------------------
def build_backend_url() -> str:
    return os.getenv("STREAMLIT_API_URL", "http://localhost:8000").rstrip("/")


# ---------------------------------------------------
# Backend health check
# ---------------------------------------------------
def fetch_health(base_url: str):
    try:
        r = requests.get(f"{base_url}/health", timeout=4)
        r.raise_for_status()
        return True, r.json()
    except Exception:
        return False, {}


# ---------------------------------------------------
# Call backend /predict
# ---------------------------------------------------
def call_predict(base_url: str, texts: List[str], prefer="Auto (fast)", ask_lime=True):
    payload = {"texts": texts}

    params = {"prefer": prefer}
    if ask_lime:
        params["explain"] = "true"

    headers = {"Content-Type": "application/json"}

    r = requests.post(
        f"{base_url}/predict/",
        json=payload,
        params=params,
        headers=headers,
        timeout=120
    )
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------
# Embed LIME iframe (if user wants it)
# ---------------------------------------------------
def embed_lime_iframe(url: str, height: int = 450):
    iframe_html = f"""
    <iframe src="{url}" width="100%" height="{height}" style="border:1px solid #ccc;"></iframe>
    """
    st.components.v1.html(iframe_html, height=height + 20, scrolling=True)


# ---------------------------------------------------
# Example sets (ONLY your requested Mixed Batch)
# ---------------------------------------------------
example_sets = {
    "Mixed batch": [
        "Scientists Confirm the Moon Is Slowly Turning Into a Giant Diamond",
        "The decades-long campaign to forcefully stop the flow of drugs into the US is back despite being judged by many policymakers to be a failure",
        "RBI Raises Growth Forecast for India to 7.2% for FY2025",
        "Government to Ban All Smartphones by 2026",
    ]
}


# ---------------------------------------------------
# JSON Download Button
# ---------------------------------------------------
def download_button(data: Any, filename: str):
    import base64
    import json

    json_str = json.dumps(data, indent=2)
    b64 = base64.b64encode(json_str.encode()).decode()

    href = f"""
    <a href="data:application/json;base64,{b64}" download="{filename}">
        📥 Download results (JSON)
    </a>
    """
    st.markdown(href, unsafe_allow_html=True)
