import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import os

def train_model():
    # Load dataset
    df = pd.read_csv("data/raw/samples.csv")
    X, y = df["text"], df["is_roller_coaster"]
    
    # Build a simple machine learning pipeline
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english")),
        ("clf", LogisticRegression())
    ])
    
    print("Training model...")
    model.fit(X, y)
    
    # Save model artifact
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/coaster_detector.pkl")
    print("Model trained and saved to models/coaster_detector.pkl!")

if __name__ == "__main__":
    train_model()
