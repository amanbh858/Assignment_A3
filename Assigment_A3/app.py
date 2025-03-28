import numpy as np
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from CustomRegressor import CustomRegression, LassoRegressor, RidgeRegressor, LassoRegularization, RidgeRegularization
from LogisticRegressor import LogisticRegression

# Initialize Flask app
app = Flask(__name__)

# Load models, encoder, and scaler
models = {
    'linear': joblib.load('linear_regression_model.pkl'),
    'ridge': joblib.load('ridge_model.pkl'),
    'lasso': joblib.load('lasso_model.pkl'),
    'logistic': joblib.load('logistic_model.pkl')
}
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
        model_choice = request.form['model']

        # Ensure model choice is valid
        if model_choice not in models:
            return render_template('index.html', error="Invalid model selection. Please choose a valid model.")

        # Format categorical & numerical data as DataFrames with column names
        categorical_data = pd.DataFrame({'fuel': [fuel], 'owner': [owner], 'brand': [brand]})
        numerical_data = pd.DataFrame({'km_driven': [km_driven], 'engine': [engine]})

        # Encoding & Scaling
        encoded_categorical_data = encoder.transform(categorical_data)  
        scaled_numerical_data = scaler.transform(numerical_data)  

        # Stack all features together
        input_data = np.hstack([scaled_numerical_data, encoded_categorical_data, np.array([[year, engine]])])

        # Select the appropriate model
        model_ = models[model_choice]

        # Prediction logic
        if model_choice in ['ridge', 'lasso']:
            prediction = model_._make_prediction(input_data)
        elif model_choice == 'logistic':
            prediction = int(model_.predict1(input_data)[0])  # Get class label
        else:
            prediction = model_.predict(input_data)

        # Return result
        if model_choice == 'logistic':
            return render_template('result.html', prediction=f"Price Category: {prediction}", model_name=model_choice.capitalize())
        else:
            return render_template('result.html', prediction=np.exp(int(prediction[0])), model_name=model_choice.capitalize())

    except Exception as e:
        print(f"Error: {e}")  # Log error for debugging
        return render_template('index.html', error="Invalid input data. Please try again.")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
