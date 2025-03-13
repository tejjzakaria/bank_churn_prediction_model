import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from preprocess import load_and_preprocess  # Import preprocessing function

# Load and preprocess data
X_train, X_test, y_train, y_test = load_and_preprocess("data/churn_data.csv")

# Train Baseline Model (Logistic Regression)
baseline_model = LogisticRegression()
baseline_model.fit(X_train, y_train)
baseline_preds = baseline_model.predict(X_test)

# Train Advanced Model (XGBoost)
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

# Define a function to calculate evaluation metrics
def get_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-score": f1_score(y_true, y_pred),
        "AUC-ROC": roc_auc_score(y_true, y_pred)
    }

# Store results in a DataFrame
results = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-score", "AUC-ROC"],
    "Baseline (Logistic Regression)": list(get_metrics(y_test, baseline_preds).values()),
    "XGBoost Model": list(get_metrics(y_test, xgb_preds).values())
})

# Save results for comparison
results.to_csv("results/model_comparison.csv", index=False)
print("Model comparison saved in 'results/model_comparison.csv'.")

# Save trained models
joblib.dump(baseline_model, "models/baseline_model.pkl")
joblib.dump(xgb_model, "models/xgb_model.pkl")
print("Models saved successfully.")
