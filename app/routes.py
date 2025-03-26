from flask import render_template, request
from app import app
import os
import numpy as np
import joblib

# Load model
model_path = os.path.join(os.path.dirname(__file__), '../models/fraud_model.pkl')
model = joblib.load(model_path)

@app.route('/', methods=['GET', 'POST'])  # Handle both GET and POST
def home():
    if request.method == 'POST':
        try:
            payment_count = int(request.form['payment_count'])
            amount = float(request.form['amount'])
            
            # Feature processing
            features = [
                np.log(payment_count + 1),
                np.log(amount + 1),
                payment_count * amount
            ]
            
            # Prediction
            prob = model.predict_proba([features])[0][1]
            result = "Fraudulent" if prob > 0.65 else "Legitimate"
            
            return render_template('index.html',
                                result=result,
                                probability=f"{prob:.1%}",
                                payment_count=payment_count,
                                amount=amount)
        
        except Exception as e:
            return render_template('index.html', error=str(e))
    
    # GET request
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    try:
        payment_count = int(request.form['payment_count'])
        amount = float(request.form['amount'])
        
        features = [
            np.log(payment_count + 1),
            np.log(amount + 1),
            payment_count * amount
        ]
        
        prob = model.predict_proba([features])[0][1]
        
        return {
            'result': "Fraudulent" if prob > 0.65 else "Legitimate",
            'probability': f"{prob:.1%}",
            'payment_count': payment_count,
            'amount': amount
        }
    
    except Exception as e:
        return {'error': str(e)}, 400