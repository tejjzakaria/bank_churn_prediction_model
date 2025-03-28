import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Preprocessing function
def load_and_preprocess(filepath):
    # Load the dataset
    df = pd.read_csv(filepath)

    # Drop unnecessary columns
    df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)

    # Encode categorical variables
    label_encoder = LabelEncoder()
    df['Gender'] = label_encoder.fit_transform(df['Gender'])  # Male=1, Female=0
    df = pd.get_dummies(df, columns=['Geography'], drop_first=True)  # One-hot encode 'Geography'

    # Define features and target variable
    X = df.drop(columns=['Exited'])  # Features
    y = df['Exited']  # Target variable

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

# Load the best models
best_log_reg = joblib.load('models/best_log_reg.pkl')
best_xgb = joblib.load('models/best_xgb.pkl')

# Load and preprocess the data
filepath = '../data/churn_data.csv'  # Replace with your file path
X_train, X_test, y_train, y_test = load_and_preprocess(filepath)

# Make predictions
log_reg_preds = best_log_reg.predict(X_test)
xgb_preds = best_xgb.predict(X_test)

# Calculate metrics for Logistic Regression
log_reg_accuracy = accuracy_score(y_test, log_reg_preds)
log_reg_precision = precision_score(y_test, log_reg_preds)
log_reg_recall = recall_score(y_test, log_reg_preds)
log_reg_f1 = f1_score(y_test, log_reg_preds)
log_reg_auc = roc_auc_score(y_test, best_log_reg.predict_proba(X_test)[:, 1])

# Calculate metrics for XGBoost
xgb_accuracy = accuracy_score(y_test, xgb_preds)
xgb_precision = precision_score(y_test, xgb_preds)
xgb_recall = recall_score(y_test, xgb_preds)
xgb_f1 = f1_score(y_test, xgb_preds)
xgb_auc = roc_auc_score(y_test, best_xgb.predict_proba(X_test)[:, 1])

# Create a DataFrame to store the comparison results
metrics_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'XGBoost'],
    'Accuracy': [log_reg_accuracy, xgb_accuracy],
    'Precision': [log_reg_precision, xgb_precision],
    'Recall': [log_reg_recall, xgb_recall],
    'F1 Score': [log_reg_f1, xgb_f1],
    'AUC-ROC': [log_reg_auc, xgb_auc]
})

# Save the comparison metrics to a CSV file
metrics_df.to_csv('models/model_comparison_metrics.csv', index=False)

# Confusion Matrix for Logistic Regression
log_reg_cm = confusion_matrix(y_test, log_reg_preds)

# Confusion Matrix for XGBoost
xgb_cm = confusion_matrix(y_test, xgb_preds)

# Plot confusion matrices
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Logistic Regression Confusion Matrix
sns.heatmap(log_reg_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted No', 'Predicted Yes'], yticklabels=['True No', 'True Yes'], ax=ax[0])
ax[0].set_title('Logistic Regression Confusion Matrix')

# XGBoost Confusion Matrix
sns.heatmap(xgb_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted No', 'Predicted Yes'], yticklabels=['True No', 'True Yes'], ax=ax[1])
ax[1].set_title('XGBoost Confusion Matrix')

plt.tight_layout()
plt.savefig('models/confusion_matrices.png')
plt.close()

# Plot metrics comparison as a bar chart
metrics_df.set_index('Model').plot(kind='bar', figsize=(10, 6))
plt.title('Model Comparison Metrics')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('models/model_comparison_bar_chart.png')

print("Evaluation metrics saved to 'models/model_comparison_metrics.csv'")
print("Confusion matrices saved to 'models/confusion_matrices.png'")
print("Metrics bar chart saved to 'models/model_comparison_bar_chart.png'")
