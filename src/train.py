import joblib
import pandas as pd
import os
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
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

# Load and preprocess the data
filepath = '../data/churn_data.csv'  # Replace with your file path
X_train, X_test, y_train, y_test = load_and_preprocess(filepath)

# Define and Train Logistic Regression using Grid Search
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization strength
    'penalty': ['l1', 'l2'],  # Regularization type
    'solver': ['liblinear', 'saga']  # Solvers that support L1/L2
}

log_reg = LogisticRegression()

# Perform Grid Search for Logistic Regression
grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)

# Best Logistic Regression Model
best_log_reg = grid_search.best_estimator_
print("Best Parameters for Logistic Regression:", grid_search.best_params_)

# Define and Train XGBoost using Randomized Search
xgb_param_grid = {
    'n_estimators': [100, 300, 500, 700],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3],
    'reg_lambda': [0.1, 1, 10],
    'reg_alpha': [0, 0.1, 1, 10]
}

xgb = XGBClassifier()

# Perform Randomized Search for XGBoost
random_search = RandomizedSearchCV(xgb, xgb_param_grid, n_iter=50, cv=5, scoring='f1', n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)

# Best XGBoost Model
best_xgb = random_search.best_estimator_
print("Best Parameters for XGBoost:", random_search.best_params_)

# Train the best models
best_log_reg.fit(X_train, y_train)
best_xgb.fit(X_train, y_train)

# Ensure the 'models' directory exists
os.makedirs('models', exist_ok=True)

# Save the optimized models
joblib.dump(best_log_reg, "models/best_log_reg.pkl")
joblib.dump(best_xgb, "models/best_xgb.pkl")
