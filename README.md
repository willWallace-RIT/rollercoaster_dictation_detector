Roller Coaster Text Dictation Detector (roller-coaster-dictation-detector)
An end-to-end machine learning project and interactive web application designed to analyze text paragraphs and assess whether they were dictated during a high-speed roller coaster ride or under normal, calm conditions.
🎢 Project Overview
Speech-to-text transcripts generated under extreme physical motion (like drops, loops, and sharp turns on a roller coaster) often display unique linguistic patterns—such as fragmented sentences, abrupt exclamation markers, emotional volatility, and erratic punctuation.
This repository provides:
 * A foundational dataset template with seed data.
 * A machine learning pipeline utilizing TF-IDF feature extraction and Logistic Regression.
 * A Streamlit web application for real-time inference and probability scoring.
📂 Repository Structure
roller-coaster-dictation-detector/
│
├── data/
│   ├── raw/                # Raw training data (CSV format)
│   └── processed/          # Cleaned/engineered datasets
│
├── models/                 # Saved model artifacts (.pkl)
│
├── src/
│   ├── __init__.py
│   ├── collect.py          # Script to initialize/format the dataset
│   └── train.py            # Feature extraction and model training script
│
├── app.py                  # Streamlit web application for inference
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation

🚀 Getting Started
1. Installation
Clone the repository and install the required Python dependencies:
pip install -r requirements.txt

2. Generate the Dataset
Run the data collection script to create your initial raw dataset with seed examples:
python src/collect.py

(Tip: You can open data/raw/samples.csv and add more custom examples to improve your classifier's accuracy!)
3. Train the Algorithm
Execute the training script to vectorize the text data and fit the classification model:
python src/train.py

This will compile the model and save the artifact to models/coaster_detector.pkl.
4. Launch the Web App
Test your trained algorithm interactively using Streamlit:
streamlit run app.py

🛠️ Tech Stack
 * Python 3.x
 * Scikit-Learn: For text vectorization (TF-IDF) and classification modeling.
 * Pandas / NumPy: For data manipulation and processing.
 * Streamlit: For building the interactive front-end application.
 * Joblib: For model serialization.
💡 Future Improvements
 * Incorporate advanced transformer-based embeddings (e.g., Hugging Face transformers) for deeper semantic context.
 * Add specialized regex or NLP features parsing punctuation density, capital letter ratios, and word repetition (e.g., "ahhhhh").
