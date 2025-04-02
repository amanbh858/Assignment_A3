from flask import Flask, render_template, request
import mlflow.pyfunc
import pandas as pd
import numpy as np
import os
import pickle
from dotenv import load_dotenv
import logging

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MODEL_NAME = "st125713-a3-model"

if not MLFLOW_TRACKING_URI:
    logger.error("MLFLOW_TRACKING_URI is not set in environment variables.")
    raise RuntimeError("Missing MLflow tracking URI.")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Load MLflow Model
try:
    logger.info(f"Loading MLflow model: {MODEL_NAME}")
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Staging")
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise RuntimeError("Model loading failed")

# Load preprocessing artifacts (encoder & scaler)
try:
    with open("encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    logger.info("Preprocessing artifacts loaded successfully")
except Exception as e:
    logger.error(f"Failed to load preprocessing artifacts: {str(e)}")
    raise RuntimeError("Preprocessing artifacts loading failed")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        fuel = request.form['fuel']
        owner = request.form['owner']
        brand = request.form['brand']
        km_driven = float(request.form['km_driven'])
        seats = int(request.form['seats'])
        year = int(request.form['year'])
        engine = int(request.form['engine'])

        # Prepare input data
        input_data = pd.DataFrame({
            'fuel': [fuel],
            'owner': [owner],
            'brand': [brand],
            'km_driven': [km_driven],
            'engine': [engine],
            'seats': [seats],
            'year': [year]
        })

        # Preprocess data
        # 1. One-hot encode categorical features
        categorical_cols = ['fuel', 'owner', 'brand']
        encoded_categorical = encoder.transform(input_data[categorical_cols])
        
        # 2. Scale numerical features
        numerical_cols = ['km_driven', 'engine']
        scaled_numerical = scaler.transform(input_data[numerical_cols])
        
        # 3. Get other features that don't need scaling
        other_features = input_data[['seats', 'year']].values

        # Combine all features
        features = np.hstack([
            scaled_numerical,
            encoded_categorical,
            other_features
        ])

        # Convert to DataFrame with correct column names if needed
        # (This depends on how your model was trained)
        # For MLflow models, you might need to ensure the input matches the signature
        
        # Reshape to match expected input shape (-1, 40)
        # Pad with zeros if necessary (this is a hacky solution)
        if features.shape[1] < 40:
            padding = np.zeros((features.shape[0], 40 - features.shape[1]))
            features = np.hstack([features, padding])
        
        logger.info(f"Final input shape: {features.shape}")

        # Make prediction
        prediction = int(model.predict(features)[0])

        # Map categories to meaningful labels
        category_labels = {0: "Budget", 1: "Mid-Range", 2: "Premium"}

        return render_template(
            'result.html',
            prediction=f"Predicted Price Category: {category_labels.get(prediction, 'Unknown')}"
        )

    except ValueError as e:
        logger.error(f"Input validation error: {str(e)}")
        return render_template('index.html', error=f"Invalid input: {str(e)}")
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return render_template('index.html', error="An error occurred. Please try again.")

if __name__ == '__main__':
    logger.info("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)