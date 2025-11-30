# streamlit_app.py
"""
Single-file Streamlit app that:
- Downloads model + vectorizer (or vocab) from a Hugging Face repo.
- Loads model and either:
    * loads a pickle'd vectorizer (preferred), or
    * rebuilds a TfidfVectorizer from saved vocabulary (vocab.pkl).
- Provides a clean UI for checking and saving test results.

Edit the HF_REPO value below if your Hugging Face repo is different.
"""

import os
import math
import pickle
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from huggingface_hub import hf_hub_download, RevisionNotFoundError

# -------------------------
# Configuration (edit if needed)
# -------------------------
HF_REPO = "naniganadi3/fakenews-model"  # <--- set to your HF repo if different
MODEL_FILE = "model.pkl"
VECTOR_FILE = "vectorizer.pkl"
VOCAB_FILE = "vocab.pkl"

TEST_LOG = "test_results.csv"
MIN_LEN = 10

# -------------------------
# Page config + small CSS
# -------------------------
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {padding-top: 1.0rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Helper: download a file via HuggingFace Hub
# -------------------------
def hf_download_if_missing(repo_id: str, filename: str):
    """
    Download `filename` from the HF repo `repo_id` into the current working dir,
    and return the local path. If file already exists locally, skip downloading.
    """
    if os.path.exists(filename):
        return filename
    try:
        local_path = hf_hub_download(repo_id=repo_id, filename=filename)
        # hf_hub_download often writes to cache; copy to working dir for clarity.
        # But we can just return local_path (works with direct open).
        return local_path
    except RevisionNotFoundError:
        # file not found in repo
        return None
    except Exception as e:
        # other download error
        st.warning(f"Warning: failed to download {filename} from Hugging Face: {e}")
        return None

# -------------------------
# Load model + vectorizer (cached)
# -------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts_from_hf(repo_id=HF_REPO):
    """
    Try to obtain (in this order):
      1) model.pkl AND vectorizer.pkl -> load both (ideal)
      2) model.pkl AND vocab.pkl -> load model + rebuild TfidfVectorizer from vocab (fallback)
    Returns: (model, vectorizer, info_dict)
    info_dict contains helpful info for debugging (which path used)
    """
    info = {"loaded_model": False, "loaded_vectorizer": False, "reconstructed_from_vocab": False, "errors": []}
    model = None
    vectorizer = None

    # 1) download model
    model_local = hf_download_if_missing(repo_id, MODEL_FILE)
    if model_local is None:
        info["errors"].append(f"{MODEL_FILE} not found in HF repo {repo_id}.")
        return None, None, info

    # load model
    try:
        with open(model_local, "rb") as f:
            model = pickle.load(f)
        info["loaded_model"] = True
        info["model_path"] = model_local
    except Exception as e:
        info["errors"].append(f"Failed to load model from {model_local}: {e}")
        return None, None, info

    # 2) try to download & load full vectorizer.pkl
    vect_local = hf_download_if_missing(repo_id, VECTOR_FILE)
    if vect_local:
        try:
            with open(vect_local, "rb") as f:
                vectorizer = pickle.load(f)
            info["loaded_vectorizer"] = True
            info["vectorizer_path"] = vect_local
            return model, vectorizer, info
        except Exception as e:
            info["errors"].append(f"Failed to load vectorizer from {vect_local}: {e}")
            # fall through to try vocab approach

    # 3) try to download vocab.pkl and rebuild vectorizer
    vocab_local = hf_download_if_missing(repo_id, VOCAB_FILE)
    if vocab_local:
        try:
            with open(vocab_local, "rb") as f:
                vocab = pickle.load(f)
            # vocab should be a dict mapping token->index
            # Rebuild TfidfVectorizer with vocabulary. Use use_idf=False to avoid NotFittedError if idf_ missing.
            vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7, vocabulary=vocab, use_idf=False, norm="l2")
            info["reconstructed_from_vocab"] = True
            info["vocab_path"] = vocab_local
            info["loaded_vectorizer"] = True
            return model, vectorizer, info
        except Exception as e:
            info["errors"].append(f"Failed to rebuild vectorizer from {vocab_local}: {e}")
            return model, None, info

    # No vectorizer or vocab found
    info["errors"].append("No vectorizer.pkl or vocab.pkl found in Hugging Face repo.")
    return model, None, info

model, vectorizer, load_info = load_artifacts_from_hf()

# -------------------------
# UI: Sidebar options
# -------------------------
with st.sidebar:
    st.header("Options")
    source = st.selectbox("Source", ["BBC", "CNN", "The Onion", "Other"])
    ground_truth = st.radio("Ground truth", ["Unknown", "Real", "Fake"])
    auto_save = st.checkbox("Auto save after check", value=False)
    st.markdown("---")
    # show load status
    if model is None:
        st.error("Model not loaded.")
    else:
        st.success(f"Model loaded ({type(model).__name__})")
    if vectorizer is None:
        st.error("Vectorizer not loaded.")
    else:
        st.info(f"Vectorizer ready. Reconstructed from vocab: {load_info.get('reconstructed_from_vocab', False)}")
    if load_info.get("errors"):
        with st.expander("Load details / errors"):
            for e in load_info["errors"]:
                st.write("- ", e)

# -------------------------
# Helpers: prediction, confidence, save
# -------------------------
def compute_confidence(m, X):
    try:
        if hasattr(m, "predict_proba"):
            p = m.predict_proba(X)[0]
            return float(max(p))
        elif hasattr(m, "decision_function"):
            val = m.decision_function(X)
            score = float(val[0]) if (hasattr(val, "__len__") and len(val.shape) == 1) else float(val[0][0])
            return 1.0 / (1.0 + math.exp(-score))
    except Exception:
        return None

def predict_text(text: str):
    if model is None or vectorizer is None:
        raise RuntimeError("Model or vectorizer not loaded.")
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    conf = compute_confidence(model, X)
    label = "REAL" if int(pred) == 1 else "FAKE"
    return label, conf

def save_test(text, source, gt, pred, conf):
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "source": source,
        "ground_truth": gt,
        "prediction": pred,
        "confidence": conf if conf is not None else "",
        "snippet": text.strip()[:300].replace("\n", " ")
    }
    df = pd.DataFrame([row])
    if os.path.exists(TEST_LOG):
        df.to_csv(TEST_LOG, mode="a", header=False, index=False)
    else:
        df.to_csv(TEST_LOG, index=False)

# -------------------------
# Main UI
# -------------------------
st.markdown("<div style='display:flex;align-items:center'><h1 style='margin:0 0.5rem 0 0'>📰 Fake News Detector</h1></div>", unsafe_allow_html=True)
st.write("Paste an article (headline + 1–2 paragraphs recommended) and click **Check**.")

text = st.text_area("News text", height=260, key="news_text")

col1, col2 = st.columns([1, 6])
with col1:
    if st.button("Check", key="check_btn"):
        if model is None or vectorizer is None:
            st.error("Model or vectorizer not found. Check the sidebar details.")
        else:
            if not isinstance(text, str) or not text.strip():
                st.error("Please enter some text.")
            elif len(text.strip()) < MIN_LEN:
                st.error(f"Please enter more text (min {MIN_LEN} characters).")
            else:
                try:
                    pred_label, conf_val = predict_text(text)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    pred_label, conf_val = None, None

                pct = int(round(conf_val * 100)) if conf_val is not None else None

                if pred_label == "FAKE":
                    if pct is not None:
                        st.error(f"🚨 {pct}% likely to be FAKE")
                    else:
                        st.error("🚨 Predicted: FAKE")
                elif pred_label == "REAL":
                    if pct is not None:
                        st.success(f"✅ {pct}% likely to be REAL")
                    else:
                        st.success("✅ Predicted: REAL")

                # auto save option
                if auto_save and pred_label is not None:
                    save_test(text, source, ground_truth, pred_label, conf_val)
                    st.info("Saved test result.")

with col2:
    st.write("")
    st.caption("Use the sidebar to set Source and Ground truth, or enable Auto save.")

# manual save
if st.button("Save test result", key="save_btn"):
    if not text or not text.strip():
        st.error("Nothing to save.")
    else:
        if model is not None and vectorizer is not None:
            try:
                pred_label, conf_val = predict_text(text)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                pred_label, conf_val = "", ""
        else:
            pred_label, conf_val = "", ""
        save_test(text, source, ground_truth, pred_label, conf_val)
        st.success("Saved test to disk.")

st.markdown("---")
st.caption("© Fake News Detector — UI optimized for clean presentation")

