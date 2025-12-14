# backend/utils/explain.py
import os
import hashlib
import logging
from typing import Optional, List
import numpy as np

logger = logging.getLogger("backend.explain")

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPL_DIR = os.path.join(BASE_DIR, "static", "explanations")
os.makedirs(EXPL_DIR, exist_ok=True)


def _predict_proba_for_lime(texts: List[str]):
    """
    Lightweight predict_proba wrapper used by LIME.
    Must return array (n_samples, 2):
    [prob_real, prob_fake]
    """
    # ✅ FIXED IMPORT
    from backend.utils.inference import load_engine

    engine = load_engine()

    try:
        labels, probs_fake = engine.predict_transformer(texts)
        probs_fake = np.array(probs_fake, dtype=float)
        probs_real = 1.0 - probs_fake
        return np.vstack((probs_real, probs_fake)).T

    except Exception:
        # Fallback to baseline
        try:
            labels, probs_fake = engine.predict_baseline(texts)
            probs_fake = np.array(probs_fake, dtype=float)
            probs_real = 1.0 - probs_fake
            return np.vstack((probs_real, probs_fake)).T
        except Exception:
            # Last-resort uniform probabilities
            n = len(texts)
            return np.tile(np.array([0.5, 0.5]), (n, 1))


def generate_lime_explanation(text: str, label: str) -> Optional[str]:
    """
    Synchronous fast LIME explanation.
    Generates an HTML file and returns public path (/static/explanations/...)
    """
    try:
        from lime.lime_text import LimeTextExplainer

        uid = hashlib.md5(text.encode()).hexdigest()[:10]
        filename = f"lime_{uid}.html"
        filepath = os.path.join(EXPL_DIR, filename)

        explainer = LimeTextExplainer(class_names=["REAL", "FAKE"])

        def predict_proba_fn(xs):
            return _predict_proba_for_lime(list(xs))

        exp = explainer.explain_instance(
            text_instance=text,
            classifier_fn=predict_proba_fn,
            num_features=6,
            num_samples=200,
            top_labels=2,
        )

        html = exp.as_html()
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)

        public_path = f"/static/explanations/{filename}"
        logger.info("Wrote LIME explanation %s", filepath)
        return public_path

    except Exception as e:
        logger.exception("generate_lime_explanation failed: %s", e)
        return None
