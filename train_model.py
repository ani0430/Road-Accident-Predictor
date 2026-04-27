import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

def train_model():
    # Load dataset
    df = pd.read_csv('data/accidents.csv')

    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['weather', 'road_type', 'time_of_day', 'alcohol', 'visibility', 'road_condition']

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Target encoding
    target_le = LabelEncoder()
    df['accident_severity'] = target_le.fit_transform(df['accident_severity'])
    label_encoders['accident_severity'] = target_le

    # Features and target
    X = df.drop('accident_severity', axis=1)
    y = df['accident_severity']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_le.classes_))

    # Save model and encoders
    os.makedirs('model', exist_ok=True)
    with open('model/accident_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('model/label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    with open('model/feature_names.pkl', 'wb') as f:
        pickle.dump(list(X.columns), f)

    print("\n✅ Model saved successfully!")
    return accuracy

if __name__ == '__main__':
    train_model()
