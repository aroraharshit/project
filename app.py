from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
model = joblib.load('heart_disease_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
   
    data = request.json
    
    # List of feature names in the same order as the model expects
    feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                     'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

    # Check if all required features are in the input data
    for feature in feature_names:
        if feature not in data:
            return jsonify({'error': f'Missing feature: {feature}'}), 400

    # Convert input data to a DataFrame
    input_data = pd.DataFrame([data], columns=feature_names)

    # Ensure all values are numeric
    input_data = input_data.apply(pd.to_numeric, errors='coerce')

    prediction = model.predict(input_data)

    # Check for any NaN values which indicates conversion issues
    if input_data.isnull().values.any():
        return jsonify({'error': 'Invalid input: non-numeric values found'}), 400


    result = 'The Person has Heart Disease' if prediction[0] == 1 else 'The Person does not have a Heart Disease'
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)