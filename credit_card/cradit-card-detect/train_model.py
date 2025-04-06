import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Sample data - replace with your actual dataset
data = {
    'payment_count': [1, 15, 3, 8, 2, 20, 5, 10],
    'amount': [50.0, 2000.0, 120.0, 5000.0, 80.0, 3000.0, 150.0, 4000.0],
    'is_fraud': [0, 1, 0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

# Feature engineering
X = np.column_stack([
    np.log(df['payment_count'] + 1),
    np.log(df['amount'] + 1),
    df['payment_count'] * df['amount']
])
y = df['is_fraud']

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, 'models/fraud_model.pkl')
print("Model trained and saved!")