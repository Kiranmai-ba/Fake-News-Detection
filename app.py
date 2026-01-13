# Fake-News-Detection
import streamlit as st
import joblib
from pathlib import Path

# 1. Get the path
BASE_DIR = Path(__file__).resolve().parent
VECTORIZER_PATH = BASE_DIR / "vectorizer.jb"
MODEL_PATH = BASE_DIR / "model.jb" 

@st.cache_resource
def load_models():
    if not VECTORIZER_PATH.exists():
        return None, None
    
    try:
        vectorizer = joblib.load(VECTORIZER_PATH)
        model = joblib.load(MODEL_PATH)
        return vectorizer, model
    except:
        return None, None

# 2. Load
vectorizer, model = load_models()

# 3. UI
st.title("Fake News Detector")
if vectorizer is None:
    st.error("Error: 'vectorizer.jb' not found in GitHub. Please upload the file.")
else:
    user_input = st.text_area("Enter a News Article below:")
    if st.button("Check News"):
        if user_input:
            data = vectorizer.transform([user_input])
            prediction = model.predict(data)
            st.success(f"Result: {prediction[0]}")
