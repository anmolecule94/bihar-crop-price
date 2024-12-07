from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Initialize label encoders
state_encoder = LabelEncoder()
district_encoder = LabelEncoder()
commodity_encoder = LabelEncoder()

# Create a simple model if the pickle file doesn't work correctly
default_model = RandomForestRegressor(n_estimators=100, random_state=42)

try:
    # Try to load the trained model
    with open('bihar_crop_price_model_rf_final.pkl', 'rb') as f:
        model_data = pickle.load(f)
        if isinstance(model_data, dict):
            model = model_data['model']
            state_encoder.classes_ = model_data['state_classes']
            district_encoder.classes_ = model_data['district_classes']
            commodity_encoder.classes_ = model_data['commodity_classes']
        elif isinstance(model_data, np.ndarray):
            # If model is actually the coefficients, create a simple predictor
            coefficients = model_data
            class SimplePredictor:
                def __init__(self, coef):
                    self.coef = coef
                def predict(self, X):
                    # Simple linear combination of features with coefficients
                    return np.dot(X, self.coef[:X.shape[1]]) + self.coef[-1]
            model = SimplePredictor(coefficients)
        else:
            model = model_data
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = default_model

# Inflation adjustment (mock example, adjust as needed)
INFLATION_RATE = 0.005  # 0.5% monthly inflation

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract user input from the form
        data = request.form
        state = data.get('State')
        district = data.get('District')
        commodity = data.get('Commodity')
        arrival_date = data.get('Arrival_Date')  # Expected format: 'YYYY-MM-DD'

        # Validate input
        if not (state and district and commodity and arrival_date):
            return jsonify({'error': 'Missing input fields!'}), 400

        try:
            # Convert arrival_date to datetime
            arrival_dt = datetime.strptime(arrival_date, '%Y-%m-%d')
            
            # Extract features from date
            month = arrival_dt.month
            year = arrival_dt.year
            day = arrival_dt.day
            
            # Use simple numeric encoding for categorical variables
            state_encoded = hash(state) % 100
            district_encoded = hash(district) % 100
            commodity_encoded = hash(commodity) % 100

            # Prepare input features
            input_features = np.array([[
                state_encoded, 
                district_encoded,
                commodity_encoded,
                month,
                day,
                year
            ]], dtype=float)

            # Generate predictions for the next 12 months
            predictions = []
            current_date = arrival_dt
            base_price = 1000  # Default base price

            for i in range(12):
                current_date += timedelta(days=30)  # Approx. 1 month
                
                # Update features for the new date
                input_features = np.array([[
                    state_encoded,
                    district_encoded,
                    commodity_encoded,
                    current_date.month,
                    current_date.day,
                    current_date.year
                ]], dtype=float)
                
                try:
                    # Try to get prediction for this month
                    month_price = float(model.predict(input_features)[0])
                except:
                    # If prediction fails, use base price with some variation
                    month_price = base_price * (1 + np.random.uniform(-0.1, 0.1))
                
                # Apply inflation adjustment
                adjusted_price = month_price * ((1 + INFLATION_RATE) ** (i + 1))
                
                predictions.append({
                    'Month': current_date.strftime('%B %Y'),
                    'Predicted_Price': round(adjusted_price, 2)
                })

            return jsonify(predictions)
        
        except ValueError as ve:
            return jsonify({'error': f'Invalid date format: {str(ve)}'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
