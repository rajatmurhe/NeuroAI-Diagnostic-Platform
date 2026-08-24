# src/preprocess.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/oasis_cross-sectional.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "../models")
os.makedirs(MODELS_DIR, exist_ok=True)

def map_diagnosis(cdr):
    if cdr == 0:
        return "Healthy"
    elif cdr == 0.5:
        return "MCI"
    else:
        return "Alzheimer"

def load_and_preprocess_data():
    df = pd.read_csv(DATA_PATH)
    
    # Target mapping from Clinical Dementia Rating (CDR)
    df["Diagnosis"] = df["CDR"].apply(map_diagnosis)
    
    # Drop non-predictive or redundant columns
    drop_cols = ["ID", "SubjectID", "MRI ID", "Delay", "Hand"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # One-hot encode categorical features (M/F)
    target = df["Diagnosis"]
    cdr_col = df["CDR"]
    features_df = df.drop(columns=["Diagnosis", "CDR"])
    features_df = pd.get_dummies(features_df, drop_first=True)
    
    # Handle missing values: median for numeric, mode for categorical
    for col in features_df.columns:
        if features_df[col].dtype in ["float64", "int64"]:
            features_df[col] = features_df[col].fillna(features_df[col].median())
        else:
            features_df[col] = features_df[col].fillna(features_df[col].mode()[0])
            
    return features_df, target

def get_train_test_splits(test_size=0.2, random_state=42):
    X, y = load_and_preprocess_data()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # Standardize numerical features for ML baselines
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Label encoding for models
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    
    # Save artifacts for inference and dashboard
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.joblib"))
    joblib.dump(X.columns.tolist(), os.path.join(MODELS_DIR, "feature_names.joblib"))
    
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, y_train_enc, y_test_enc, le

def serialize_rows(df):
    """Converts tabular rows into natural language key-value strings for Transformers."""
    return df.astype(str).apply(lambda r: ", ".join([f"{c}: {r[c]}" for c in df.columns]), axis=1).tolist()

if __name__ == "__main__":
    X_train, X_test, _, _, y_train, y_test, _, _, le = get_train_test_splits()
    print(f"Data loaded successfully. Train size: {X_train.shape}, Test size: {X_test.shape}")
    print(f"Target classes: {le.classes_}")