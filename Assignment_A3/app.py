from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
from LogisticRegressor import LogisticRegression

# Initialize Flask app
app = Flask(__name__)

# Load logistic regression model, encoder, and scaler
logistic_model = joblib.load('logistic_model.pkl')
encoder = joblib.load('encoder.pkl')
scaler = joblib.load('scaler.pkl')

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
        categorical_data = pd.DataFrame({
            'fuel': [fuel],
            'owner': [owner], 
            'brand': [brand]
        })
        
        numerical_data = pd.DataFrame({
            'km_driven': [km_driven],
            'engine': [engine]
        })

        # Preprocess data
        encoded_categorical = encoder.transform(categorical_data)
        scaled_numerical = scaler.transform(numerical_data)
        
        # Combine features with correct order
        features = np.hstack([
            scaled_numerical,
            encoded_categorical,
            np.array([[seats, year]])
        ])

        # Make prediction
        prediction = logistic_model.predict(features)[0]  # Get class label
        price_category = int(prediction)

        # Map categories to meaningful labels if needed
        category_labels = {
            0: "0:Budget",
            1: "1:Mid-Range", 
            2: "2:Premium"
        }
        
        return render_template(
            'result.html',
            prediction=f"Predicted Price Category: {category_labels.get(price_category, price_category)}"
        )

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return render_template(
            'index.html',
            error="Invalid input data. Please check your values and try again."
        )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)