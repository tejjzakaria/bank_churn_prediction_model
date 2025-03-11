import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

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
