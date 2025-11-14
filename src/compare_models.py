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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=['int64','float64']).columns.drop(['default'], errors='ignore').tolist()
    if 'loan_id' in numeric_features:
        numeric_features.remove('loan_id')
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    if 'loan_id' in categorical_features:
        categorical_features.remove('loan_id')
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)], remainder='drop')
    return preprocessor

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.input)
    if 'loan_id' in df.columns:
        df = df.drop(columns=['loan_id'])
    X = df.drop(columns=['default'], errors='ignore')
    y = df['default']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pre = build_preprocessor(X_train)
    models = {
        'logreg': LogisticRegression(max_iter=500),
        'rf': RandomForestClassifier(n_estimators=200, random_state=42)
    }
    results = []
    best = None
    best_score = -1
    for name, clf in models.items():
        pipe = Pipeline(steps=[('pre', pre), ('clf', clf)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:,1] if hasattr(pipe, 'predict_proba') else None
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        roc = roc_auc_score(y_test, proba) if proba is not None else None
        results.append({'model': name, 'accuracy': acc, 'f1': f1, 'roc_auc': roc})
        if roc is not None and roc > best_score:
            best_score = roc
            best = (name, pipe)
    res_df = pd.DataFrame(results).sort_values('roc_auc', ascending=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(args.out, index=False)
    # save best model next to the comparison csv
    if best is not None:
        joblib.dump(best[1], Path(args.out).parent / 'best_model.joblib')
    print(res_df)

if __name__ == '__main__':
    main()
