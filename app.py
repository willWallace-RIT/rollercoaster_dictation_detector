import streamlit as st
import joblib
import os

st.title("🎢 Roller Coaster Text Dictation Detector")
st.write("Analyze whether a transcribed paragraph was likely dictated while riding a roller coaster!")

# Load model safely
model_path = "models/coaster_detector.pkl"
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.warning("Model not found! Please run `python src/train.py` first.")
    model = None

user_input = st.text_area("Paste the transcribed paragraph here:", "Oh wow we are going upside down right now AAAH!")

if st.button("Assess Text"):
    if model and user_input:
        prediction = model.predict([user_input])[0]
        probability = model.predict_proba([user_input])[0][prediction]
        
        if prediction == 1:
            st.error(st.markdown(f"🚨 **Likely dictated on a Roller Coaster!** (Confidence: {probability:.2f})"))
            st.info("Indicators: High emotional intensity, abrupt phrasing, or speech fragmentation markers.")
        else:
            st.success(st.markdown(f"✅ **Likely Normal/Calm Dictation** (Confidence: {probability:.2f})"))
    else:
        st.error("Please ensure the model is trained and input text is provided.")
