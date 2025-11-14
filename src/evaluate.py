# src/evaluate.py  (replace existing file)
import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
import matplotlib.pyplot as plt

def safe_load_csv(path):
    return pd.read_csv(path)

def ensure_columns_for_pipeline(pipe, X):
    """
    Ensure X contains all columns the pipeline was fitted with.
    If a column is missing, add it with NaNs. This avoids ValueError during transform.
    """
    # try to read feature names from the fitted ColumnTransformer inside pipeline
    try:
        # If pipeline has a ColumnTransformer named 'pre' or first step
        pre = None
        if hasattr(pipe, "named_steps"):
            # find a ColumnTransformer inside the pipeline steps
            for name, step in pipe.named_steps.items():
                from sklearn.compose import ColumnTransformer
                if isinstance(step, ColumnTransformer):
                    pre = step
                    break
        if pre is None:
            # fallback: try first step (commonly 'pre')
            first = list(pipe.named_steps.values())[0]
            if isinstance(first, ColumnTransformer):
                pre = first
    except Exception:
        pre = None

    # fallback: use columns of training X stored in pipeline (if any)
    expected = None
    if hasattr(pipe, "feature_names_in_"):
        expected = list(pipe.feature_names_in_)
    if expected is None and pre is not None:
        # collect column names that transformers were fitted on
        expected = []
        for _, trans, cols in pre.transformers_:
            # cols can be slice, list-like, or 'remainder'
            try:
                if isinstance(cols, (list, tuple, np.ndarray)):
                    expected.extend(list(cols))
            except Exception:
                pass
        expected = list(dict.fromkeys([c for c in expected if isinstance(c, str)]))  # unique order-preserving

    if expected:
        for col in expected:
            if col not in X.columns:
                X[col] = np.nan
    return X

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python src/evaluate.py <path-to-model.joblib> <test-csv>")
        raise SystemExit(1)

    model_path = sys.argv[1]
    test_csv = sys.argv[2]

    pipe = joblib.load(model_path)
    df = safe_load_csv(test_csv)

    # Keep loan_id (do NOT drop it) — model expects same columns as training X
    # We drop only the target if present:
    X = df.drop(columns=['default'], errors='ignore')
    y = df['default'] if 'default' in df.columns else None

    # Ensure columns match pipeline expectations (adds missing columns as NaN)
    X = ensure_columns_for_pipeline(pipe, X)

    preds = pipe.predict(X)
    proba = pipe.predict_proba(X)[:,1] if hasattr(pipe, "predict_proba") else None

    if y is not None:
        print(classification_report(y, preds))
    if proba is not None and y is not None:
        print("ROC AUC:", roc_auc_score(y, proba))

    Path('src/figures').mkdir(parents=True, exist_ok=True)
    if y is not None:
        cm = confusion_matrix(y, preds)
        plt.figure(figsize=(4,4))
        plt.imshow(cm, cmap='Blues', interpolation='nearest')
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig('src/figures/confusion_matrix.png', bbox_inches='tight')
        plt.close()
        print("Saved confusion matrix -> src/figures/confusion_matrix.png")

    else:
        print("No true labels (y) found in test CSV; predictions saved only.")
