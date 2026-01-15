# Fake-News-Detection
import streamlit as st
import joblib
import os
from pathlib import Path

# 1. Define the paths correctly
# This looks for the files in the same folder as app.py
current_dir = Path(__file__).parent
VECTORIZER_PATH = current_dir / "vectorizer.jb"
MODEL_PATH = current_dir / "lr_model.jb"

@st.cache_resource
def load_models():
    # Now VECTORIZER_PATH is defined and can be checked
    if not VECTORIZER_PATH.exists() or not MODEL_PATH.exists():
        st.error(f"Files not found! Looking for: {VECTORIZER_PATH}")
        return None, None
    
    try:
        vectorizer = joblib.load(VECTORIZER_PATH)
        model = joblib.load(MODEL_PATH)
        return vectorizer, model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# 2. Load the models
vectorizer, model = load_models()

# 3. UI logic
if vectorizer is None or model is None:
    st.stop() # Stops the app here if files are missing so you don't get more errors

st.title("Fake News Detector")
# ... rest of your code
