import argparse
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

def load_data(path):
    return pd.read_csv(path)

def build_pipeline(X):
    numeric_features = X.select_dtypes(include=['int64','float64']).columns.drop(['default','loan_id'], errors='ignore').tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)])
    clf = Pipeline(steps=[('pre', preprocessor), ('clf', RandomForestClassifier(n_estimators=200, random_state=42))])
    return clf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    df = load_data(args.input)
    X = df.drop(columns=['default'], errors='ignore')
    y = df['default'] if 'default' in df.columns else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe = build_pipeline(X_train)
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:,1] if hasattr(pipe, 'predict_proba') else None
    print(classification_report(y_test, preds))
    if proba is not None:
        print("ROC AUC:", roc_auc_score(y_test, proba))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, args.out)
    print("Saved model to", args.out)

if __name__ == "__main__":
    main()