# frontend/app.py
import os
import json
from typing import List, Dict, Any
import requests
import streamlit as st
from datetime import datetime
import pandas as pd

from utils.frontend_utils import (
    build_backend_url,
    call_predict,
    fetch_health,
    example_sets,
    download_button,
)

# ---------------- Page config & theme tweak ----------------
st.set_page_config(
    page_title="Fake News Detector — Demo",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# small CSS polish for card-like look & dark-ish header
st.write(
    """
    <style>
    .stButton>button { background-color:#0b6cff; color: white; border-radius:10px; padding:8px 18px; }
    .stAlert { border-radius: 8px; }
    .card { background: linear-gradient(180deg, #ffffff 0%, #f7fbff 100%); padding:16px; border-radius:12px;
            box-shadow: 0 6px 22px rgba(11,108,255,0.08); margin-bottom: 18px; }
    .metric-label { color:#94a3b8; font-weight:700; }
    </style>
    """,
    unsafe_allow_html=True,
)

BACKEND_URL = build_backend_url()

# ---------------- Helpers ----------------
def append_recent_predictions(results: List[Dict[str, Any]]):
    if "recent_preds" not in st.session_state:
        st.session_state["recent_preds"] = []
    for r in results:
        rec = {
            "ts": datetime.utcnow().isoformat(),
            "text": r.get("text", "")[:1000],
            "label": r.get("label", "UNKNOWN"),
            "prob_fake": float(r.get("prob_fake", 0.0)),
            "prob_real": float(r.get("prob_real", 1.0 - r.get("prob_fake", 0.0))),
            "confidence": float(r.get("confidence", 0.0)),
            "lime_url": r.get("lime_url", None),
        }
        st.session_state["recent_preds"].insert(0, rec)
    st.session_state["recent_preds"] = st.session_state["recent_preds"][:200]

def get_dashboard_stats():
    try:
        r = requests.get(f"{BACKEND_URL}/dashboard/stats", timeout=5)
        r.raise_for_status()
        return True, r.json()
    except Exception:
        return False, {}

# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("Fake News Detector")
    st.markdown("Streamlit demo for your FastAPI backend")
    st.markdown("---")
    st.subheader("Backend")
    ok, health = fetch_health(BACKEND_URL)
    if ok:
        st.success("Backend reachable")
        st.write(f"Model dir: `{health.get('model_dir','not-set')}`")
        st.write(f"Baseline: `{health.get('baseline_path','not-set')}`")
    else:
        st.error("Backend unreachable")
        st.write("Start backend and set STREAMLIT_API_URL if needed.")
    st.markdown("---")
    st.subheader("Options")
    generate_lime = st.checkbox("Generate LIME explanations (may be slow)", value=True)
    st.markdown("---")
    st.write("Examples")
    if st.button("Load Mixed batch examples"):
        st.session_state["text_input"] = "\n".join(example_sets["Mixed batch"])
    st.markdown("---")
    st.write("Quick links")
    st.markdown(f"- [API docs]({BACKEND_URL}/docs)")

# ---------------- Tabs: Predict / Dashboard ----------------
tab_predict, tab_dashboard = st.tabs(["Predict", "Dashboard"])

# ---------------- Predict Tab ----------------
with tab_predict:
    st.header("Predict")
    st.caption("Enter one or more texts (one per line).")

    text_input = st.text_area(
        "Input texts (one per line):",
        value=st.session_state.get("text_input", "\n".join(example_sets["Mixed batch"])),
        height=220,
    )

    col_run, col_clear = st.columns([1, 1])
    with col_run:
        run_predict = st.button("Predict")
    with col_clear:
        if st.button("Clear"):
            st.session_state["text_input"] = ""
            st.experimental_rerun()

    model_mode = st.selectbox("Preferred Model (hint)", ["Auto (fast)", "Baseline only (fast)", "Transformer (accurate, slower)"])

    if run_predict:
        texts = [t.strip() for t in text_input.splitlines() if t.strip()]
        if not texts:
            st.warning("Please enter some text.")
        else:
            with st.spinner("Calling backend for predictions..."):
                try:
                    results = call_predict(BACKEND_URL, texts, prefer=model_mode, ask_lime=generate_lime)
                except Exception as e:
                    st.error(f"Backend request failed: {e}")
                    results = None

            if results is None:
                st.stop()

            append_recent_predictions(results)
            st.success("Predictions received")
            download_button(results, f"predictions_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
            st.markdown("")

            # Display results with simple card style and highlight stronger label
            for i, r in enumerate(results):
                st.markdown(f"### Example {i+1}")
                st.write(r.get("text", ""))

                # highlight the winning class visually
                prob_fake = float(r.get("prob_fake", 0.0))
                prob_real = float(r.get("prob_real", 1.0 - prob_fake)) if r.get("prob_real") is None else float(r.get("prob_real"))
                confidence = float(r.get("confidence", max(prob_fake, prob_real)))

                win_label = "REAL" if prob_real >= prob_fake else "FAKE"

                # Card container
                st.markdown('<div class="card">', unsafe_allow_html=True)
                c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
                # Label with simple emphasis
                label_display = r.get("label", win_label)
                if label_display == "REAL":
                    c1.markdown(f"<div style='color:#0b6cff; font-weight:700; font-size:20px'>REAL</div>", unsafe_allow_html=True)
                elif label_display == "FAKE":
                    c1.markdown(f"<div style='color:#ef4444; font-weight:700; font-size:20px'>FAKE</div>", unsafe_allow_html=True)
                else:
                    c1.markdown(f"<div style='color:#64748b; font-weight:700; font-size:20px'>{label_display}</div>", unsafe_allow_html=True)

                c2.metric("Prob (fake)", f"{prob_fake:.3f}")
                c3.metric("Prob (real)", f"{prob_real:.3f}")
                c4.metric("Confidence", f"{confidence:.3f}")
                st.markdown("</div>", unsafe_allow_html=True)

                lime_url = r.get("lime_url")
                if lime_url:
                    full_url = lime_url if lime_url.startswith("http") else BACKEND_URL.rstrip("/") + lime_url
                    st.markdown(f"**LIME explanation:** [{full_url}]({full_url})")
                else:
                    if generate_lime:
                        st.info("LIME explanation not yet available (backend may still be generating it).")

                st.markdown("---")

# ---------------- Dashboard Tab ----------------
with tab_dashboard:
    st.header("Dashboard")
    st.caption("Session-level summary and backend stats")

    ok_stats, stats_json = get_dashboard_stats()
    left, right = st.columns([2, 1])

    with left:
        st.subheader("Backend manifest & stats")
        if ok_stats:
            manifest = stats_json.get("manifest")
            stats = stats_json.get("stats", {})
            if manifest:
                st.write("Manifest (snippet):")
                st.code(json.dumps(manifest, indent=2)[:1500] + ("..." if len(json.dumps(manifest)) > 1500 else ""))
            else:
                st.info("No manifest found in model dir.")
            st.write("Stats:")
            st.write(stats)
        else:
            st.warning("Could not fetch stats from backend.")

    with right:
        st.subheader("Session summary")
        recent = st.session_state.get("recent_preds", [])
        st.metric("Predictions (this session)", len(recent))
        if recent:
            df = pd.DataFrame(recent)
            fake_count = int((df["label"] == "FAKE").sum())
            real_count = int((df["label"] == "REAL").sum())
            st.metric("FAKE", fake_count)
            st.metric("REAL", real_count)
        else:
            st.metric("FAKE", 0)
            st.metric("REAL", 0)

    st.markdown("---")
    st.subheader("Recent predictions (most recent first)")
    recent = st.session_state.get("recent_preds", [])
    if recent:
        for entry in recent[:10]:
            with st.expander(f"{entry['ts']} — {entry['label']} ({entry['confidence']:.3f})"):
                st.write(entry["text"])
                st.write(f"Prob fake: {entry['prob_fake']:.3f} — Prob real: {entry['prob_real']:.3f}")
                if entry.get("lime_url"):
                    url = entry["lime_url"] if entry["lime_url"].startswith("http") else BACKEND_URL.rstrip("/") + entry["lime_url"]
                    st.write("LIME:", url)
    else:
        st.info("No predictions in this session yet. Run some predictions from the Predict tab.")

# ---------------- Footer ----------------
st.markdown("---")
st.caption("Built with Streamlit • Connects to your FastAPI backend")
