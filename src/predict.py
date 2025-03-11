import joblib
import numpy as np

# Load model
model = joblib.load("models/churn_model.pkl")

def predict_churn(sample):
    prediction = model.predict(np.array(sample).reshape(1, -1))
    return "Churn" if prediction[0] == 1 else "Not Churn"

# Example input (modify based on your dataset structure)
sample_customer = [600, 1, 40, 3, 60000, 2, 1, 1, 50000]  # Sample feature values
print("Prediction:", predict_churn(sample_customer))
