import streamlit as st
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os

# Set page config FIRST
st.set_page_config(page_title="Deepfake Detector", layout="wide")

# Function to download Kaggle dataset
def download_kaggle_dataset():
    try:
        api = KaggleApi()
        api.authenticate()  # Requires kaggle.json in C:\Users\ADMIN\.kaggle\
        dataset_path = "dataset"
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
            api.dataset_download_files('ashukr/deepfake-detection', path=dataset_path, unzip=True)
            st.success("Dataset downloaded successfully!")
        return dataset_path
    except IOError as e:
        st.error(f"Kaggle Authentication Failed: {str(e)}")
        st.info("Fix: Place 'kaggle.json' in C:\\Users\\jhunj\\.kaggle\\. Steps: Kaggle > Settings > API > Create New Token.")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# Placeholder data loading (replace with actual dataset structure)
def load_data(dataset_path):
    # Mock data since actual dataset structure is unknown
    sample_data = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'label': np.random.choice([0, 1], 100)  # 0 = real, 1 = deepfake
    })
    return sample_data

# Train a simple model
@st.cache_resource
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def main():
    st.title("Deepfake Detection Predictor")
    st.write("Detect deepfakes using the ashukr/deepfake-detection dataset from Kaggle.")

    # Instructions for Kaggle setup
    st.info("Ensure 'kaggle.json' is in C:\\Users\\jhunj\\.kaggle\\. Download from Kaggle Settings > API > Create New Token.")

    # Download dataset
    with st.spinner("Downloading dataset..."):
        dataset_path = download_kaggle_dataset()

    if dataset_path:
        # Load data
        data = load_data(dataset_path)
        X = data[['feature1', 'feature2']]
        y = data['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = train_model(X_train, y_train)

        # User input for prediction
        st.subheader("Test a Sample")
        feature1 = st.slider("Feature 1:", 0.0, 1.0, 0.5)
        feature2 = st.slider("Feature 2:", 0.0, 1.0, 0.5)

        if st.button("Predict"):
            input_data = np.array([[feature1, feature2]])
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]
            result = "Deepfake" if prediction == 1 else "Real"
            st.success(f"Prediction: **{result}** (Deepfake Probability: {probability:.2f})")

        # Show sample data
        st.subheader("Sample Data Preview")
        st.dataframe(data.head())

if __name__ == "__main__":
    main()