import os
import pickle
import math
import pandas as pd
import streamlit as st
from datetime import datetime

# -------------------------
# Streamlit page config
# -------------------------
st.set_page_config(page_title="Fake News Detector", layout="centered")

# -------------------------
# Small CSS polish
# -------------------------
st.markdown(
    """
    <style>
    /* remove streamlit menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    /* tighten spacing */
    .block-container {padding-top: 1.2rem;}
    .stTextArea>div>div>textarea {min-height: 260px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Paths & settings
# -------------------------
MODEL_PATH = "model.pkl"
VECT_PATH = "vectorizer.pkl"
TEST_LOG = "test_results.csv"
MIN_LEN = 10

# -------------------------
# Load model + vectorizer
# -------------------------
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECT_PATH):
        return None, None
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECT_PATH, "rb") as f:
        vect = pickle.load(f)
    return model, vect

model, vect = load_artifacts()

# -------------------------
# Helpers
# -------------------------
def compute_confidence(m, X):
    """
    Returns the confidence score (probability) of the PREDICTED class.
    """
    try:
        if hasattr(m, "predict_proba"):
            p = m.predict_proba(X)[0]
            return float(max(p))
        elif hasattr(m, "decision_function"):
            val = m.decision_function(X)
            # Handle shape inconsistencies in different sklearn versions
            score = float(val[0]) if (hasattr(val, "__len__") and len(val.shape) == 1) else float(val[0][0])
            
            # Sigmoid function gives probability of the Positive Class (Class 1 = REAL)
            prob_real = 1.0 / (1.0 + math.exp(-score))
            
            # If prob_real > 0.5, predicted is REAL, conf is prob_real
            # If prob_real < 0.5, predicted is FAKE, conf is 1 - prob_real
            return max(prob_real, 1.0 - prob_real)
    except Exception:
        return None

def predict(text):
    X = vect.transform([text])
    pred = model.predict(X)[0]
    conf = compute_confidence(model, X)
    
    # 0 = FAKE, 1 = REAL
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
# Sidebar controls
# -------------------------
with st.sidebar:
    st.header("Options")
    source = st.selectbox("Source", ["BBC", "CNN", "The Onion", "Other"])
    ground_truth = st.radio("Ground truth", ["Unknown", "Real", "Fake"])
    auto_save = st.checkbox("Auto save after check", value=False)
    st.markdown("---")
    st.caption("Model loaded: " + (type(model).__name__ if model else "Not found"))
    
    if st.button("Load latest saved tests"):
        if os.path.exists(TEST_LOG):
            try:
                st.success(f"Found {len(pd.read_csv(TEST_LOG))} saved tests")
            except Exception:
                st.warning("Could not read log file.")
        else:
            st.info("No saved tests found")

# -------------------------
# Main UI
# -------------------------
st.markdown("<div style='display:flex;align-items:center'><h1 style='margin:0 0.5rem 0 0'>📰 Fake News Detector</h1></div>", unsafe_allow_html=True)
st.write("Paste an article (headline + 1–2 paragraphs recommended) and click **Check**.")

text = st.text_area("News text", height=260, key="news_text")

col1, col2 = st.columns([1, 6])
with col1:
    if st.button("Check", key="check_btn"):
        if model is None or vect is None:
            st.error("Model or vectorizer not found. Train / serialize model first.")
        else:
            if not isinstance(text, str) or not text.strip():
                st.error("Please enter some text.")
            elif len(text.strip()) < MIN_LEN:
                st.error(f"Please enter more text (min {MIN_LEN} characters).")
            else:
                pred_label, conf_val = predict(text)
                pct = int(round(conf_val * 100)) if conf_val is not None else None

                if pred_label == "FAKE":
                    if pct is not None:
                        st.error(f"🚨 {pct}% likely to be FAKE")
                    else:
                        st.error("🚨 Predicted: FAKE")
                else:
                    if pct is not None:
                        st.success(f"✅ {pct}% likely to be REAL")
                    else:
                        st.success("✅ Predicted: REAL")

                if auto_save:
                    save_test(text, source, ground_truth, pred_label, conf_val)
                    st.info("Saved test result.")

with col2:
    st.write("")
    st.caption("Use the sidebar to set Source and Ground truth, or enable Auto save.")

# Manual save button
if st.button("Save test result", key="save_btn"):
    if not text or not text.strip():
        st.error("Nothing to save.")
    else:
        if model is not None and vect is not None:
            pred_label, conf_val = predict(text)
        else:
            pred_label, conf_val = "", ""
        save_test(text, source, ground_truth, pred_label, conf_val)
        st.success("Saved test to disk.")

st.markdown("---")
st.caption("© Fake News Detector — UI optimized for clean presentation")
