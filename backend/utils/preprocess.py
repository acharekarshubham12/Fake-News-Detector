# utils/preprocess.py
import re


def clean_text(text: str) -> str:
    text = text or ""
    text = re.sub(r"\s+", " ", text).strip()
    # remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)
    # normalize quotes
    text = text.replace('\u201c', '"').replace('\u201d', '"').replace("“", '"').replace("”", '"')
    return text

