from preprocess import load_and_preprocess
from xgboost import XGBClassifier
import joblib

X_train, X_test, y_train, y_test = load_and_preprocess("data/churn_data.csv")

# Train model
model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "models/churn_model.pkl")

print("Model trained and saved successfully!")
