import pytest
import numpy as np
import pandas as pd
import cloudpickle  
import mlflow
import os
import sys
from Utils.utils import load_latest_model


# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_test.pkl")
PREPROCESS_PATH = os.path.join(BASE_DIR, "preprocess_test.pkl")
BRAND_MEANS_PATH = os.path.join(BASE_DIR, "brand_means.pkl")

print("Preprocessor path:", PREPROCESS_PATH)
print("File size (bytes):", os.path.getsize(PREPROCESS_PATH))

# Load with cloudpickle instead of joblib 
with open(PREPROCESS_PATH, "rb") as f:
    preprocessor = cloudpickle.load(f)
#
with open(BRAND_MEANS_PATH, "rb") as f:
    brand_means = cloudpickle.load(f)
#

with open(MODEL_PATH, "rb") as f:
     model = cloudpickle.load(f)

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
        preprocess_input = preprocessor.transform(sample_input)
        model.predict(preprocess_input, is_test=True)
    except Exception as e:
        pytest.fail(f"Model failed to take expected input: {e}")


feature_names = ['seller_type_Individual', 'seller_type_Dealer','seller_type_Trustmark_Dealer','transmission_Automatic', 'transmission_Manual','year','km_driven','engine', 'max_power', 'brand_encoded']
def test_model_output():
    """Test if model output shape is correct."""
    coef, bias = model._coeff_and_biases(feature_names)
    assert coef.shape == (10, 4) and bias.shape == (4,), \
        f"Output shape is incorrect. Got coef shape {coef.shape}, bias shape {bias.shape}"
    print("Model output shape test passed!")