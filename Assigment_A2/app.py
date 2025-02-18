from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import pandas as pd
from CustomRegressor import CustomRegression, LassoRegressor, RidgeRegressor, LassoRegularization, RidgeRegularization
        
# Initializing the Flask app
app = Flask(__name__)

# Loading the trained models, encoder, and scaler
models = {
    'linear': joblib.load('car_price_linear_regression_model.pkl'),
    'ridge': joblib.load('ridge_model.pkl'),
    'lasso': joblib.load('lasso_model.pkl')
}
encoder = joblib.load('encoder.pkl')
scaler = joblib.load('scaler.pkl')

# Defining the categorical columns for encoding
# categorical_columns = ['fuel', 'owner', 'brand']
# numerical_columns = ['km_driven']s

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the required attributes from the data
        fuel = request.form['fuel']
        owner = request.form['owner']
        brand = request.form['brand']
        km_driven = float(request.form['km_driven'])
        seats = int(request.form['seats'])
        year = int(request.form['year'])
        engine = int(request.form['engine'])
        model_choice = request.form['model']
        

        # Creating an input DataFrame
        categorical_data = pd.DataFrame({'fuel':fuel, 'owner':owner, 'brand':brand}, index=[0])
        numerical_data = pd.DataFrame({'km_driven':km_driven, 'engine':engine}, index=[0])
        
        
        print(categorical_data)
        print(numerical_data)
        
        # Encoding the categorical data
        encoded_categorical_data = encoder.transform(categorical_data.values)
        # Scaling the numerical data
        print(encoded_categorical_data)
        scaled_numerical_data = scaler.transform(numerical_data.values)
        print(scaled_numerical_data)
        # Combining the categorical and numerical features
        input_data = np.hstack([scaled_numerical_data, encoded_categorical_data, [[year, engine]]])
        print(input_data.shape)
        print(input_data)

        # Select the model based on user input (linear, ridge, or lasso)
        model_ = models[str(model_choice)]
        if model_choice=='linear':
            prediction =model_.predict(input_data)
        else:
            print(model_choice)
            prediction=model_._make_prediction(input_data)
        # print("prediction of {}".format(model_choice,prediction))
        print(prediction)

        # Checking if model is properly selected
        if model_ is None:
            raise ValueError("Model choice is invalid. Please select a valid model.")

        # Predicting the car price
        print(model_)
        # prediction = model_(input_data)[0]

        # Rendering the result.html with the predicted price
        return render_template('result.html', prediction=np.exp(int((prediction[0]))), model_name=model_choice.capitalize())

    except Exception as e:
        # Returning error message if something goes wrong(using exception)
        print(f"Error: {e}")  # Log the error for debugging
        return render_template('index.html', error="Invalid input data. Please try again.")

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000,debug=True)
