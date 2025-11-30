# streamlit_app.py
import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from download_artifacts import download_artifacts

# ensure files are present (downloads if missing)
download_artifacts()

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load saved vocabulary
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

# Recreate vectorizer WITHOUT IDF (prevents NotFittedError)
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7,
    vocabulary=vocab,
    use_idf=False,
    norm="l2"
)

st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector")

text = st.text_area("Paste news text (headline + 1–2 paragraphs):", height=260)

if st.button("Check"):
    if not text.strip():
        st.error("Please enter some text.")
    else:
        X = vectorizer.transform([text])
        pred = model.predict(X)[0]
        label = "REAL" if int(pred) == 1 else "FAKE"

        # optional confidence calculation (may be unavailable)
        conf = None
        try:
            if hasattr(model, "predict_proba"):
                conf = max(model.predict_proba(X)[0])
            elif hasattr(model, "decision_function"):
                import math
                score = float(model.decision_function(X)[0])
                conf = 1.0 / (1.0 + math.exp(-score))
        except Exception:
            conf = None

        if label == "FAKE":
            if conf is not None:
                st.error(f"🚨 FAKE ({int(conf*100)}% confidence)")
            else:
                st.error("🚨 FAKE")
        else:
            if conf is not None:
                st.success(f"✅ REAL ({int(conf*100)}% confidence)")
            else:
                st.success("✅ REAL")
