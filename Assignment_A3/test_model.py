import pytest
import numpy as np
import pandas as pd
import pickle
import mlflow
import os
import sys
from Utils.utils import load_latest_model


# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_test.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "encoder.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
BRAND_MEANS_PATH = os.path.join(BASE_DIR, "brand_means.pkl")

print("Encoder path:", ENCODER_PATH)
print("Scaler path:", SCALER_PATH)
print("File size (bytes):", os.path.getsize(ENCODER_PATH))

# Load encoder and scaler
with open(ENCODER_PATH, "rb") as f:
    encoder = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# Load brand means (if still needed)
with open(BRAND_MEANS_PATH, "rb") as f:
    brand_means = pickle.load(f)

# Load model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Define sample input as DataFrame
columns = ["year", "km_driven", "seller_type", "transmission", "engine", "max_power", "brand"]
sample_input = pd.DataFrame([[2015, 50000, "Individual", "Automatic", 1000, 80, "Audi"]], columns=columns)

# Map brand to encoded value
sample_input["brand_encoded"] = sample_input["brand"].map(brand_means)
sample_input["brand_encoded"] = sample_input["brand_encoded"].fillna(np.mean(list(brand_means.values())))
sample_input = sample_input.drop(columns=["brand"])  # Drop original brand column

def test_load_model():
    ml_model = load_latest_model()
    assert ml_model

def test_model_input():
    """Test if the model takes expected input format."""
    try:
        # One-hot encode categorical features
        categorical_cols = ['seller_type', 'transmission']
        encoded_categorical = encoder.transform(sample_input[categorical_cols])

        # Scale numerical features
        numerical_cols = ['km_driven', 'engine', 'max_power', 'year']
        scaled_numerical = scaler.transform(sample_input[numerical_cols])

        # Combine the features
        features = np.hstack([scaled_numerical, encoded_categorical])

        # Model prediction
        model.predict(features, is_test=True)
    except Exception as e:
        pytest.fail(f"Model failed to take expected input: {e}")


feature_names = ['seller_type_Individual', 'seller_type_Dealer', 'seller_type_Trustmark_Dealer', 'transmission_Automatic', 'transmission_Manual', 'year', 'km_driven', 'engine', 'max_power', 'brand_encoded']
def test_model_output():
    """Test if model output shape is correct."""
    coef, bias = model._coeff_and_biases(feature_names)
    assert coef.shape == (10, 4) and bias.shape == (4,), \
        f"Output shape is incorrect. Got coef shape {coef.shape}, bias shape {bias.shape}"
    print("Model output shape test passed!")
