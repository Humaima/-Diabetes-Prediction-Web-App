import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def preprocess_data():
    # Load data
    df = pd.read_csv('diabetes.csv')
    
    # Handle missing values (0s in medical data often mean missing)
    cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_to_replace] = df[cols_to_replace].replace(0, pd.NA)
    df.fillna(df.median(), inplace=True)
    
    # Split data
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Save scaler for later use
    joblib.dump(scaler, 'scaler.joblib')
    
    return X_train, X_test, y_train, y_test, df.columns