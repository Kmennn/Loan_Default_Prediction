"""
Robust feature importance extractor for a pipeline saved at src/models/model.joblib

- Tries to get the post-processor feature names via:
    1) pre.get_feature_names_out() (best)
    2) pipeline.feature_names_in_
    3) build from numeric + categorical lists (fallback)
- Aligns the feature list length with classifier.feature_importances_ by truncating or padding (with warnings).
- Saves CSV -> reports/feature_importance.csv and PNG -> src/figures/feature_importance.png
"""
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

MODEL_PATH = "src/models/model.joblib"
OUT_CSV = Path("reports/feature_importance.csv")
OUT_PNG = Path("src/figures/feature_importance.png")

# Load model
pipe = joblib.load(MODEL_PATH)

# Find classifier and preprocessor
clf = None
pre = None
if hasattr(pipe, "named_steps"):
    clf = pipe.named_steps.get("clf", None)
    pre = pipe.named_steps.get("pre", None)
else:
    # If pipe is not a Pipeline, assume it's the classifier
    clf = pipe

if clf is None:
    raise SystemExit("Classifier not found inside pipeline (expected 'clf' step).")

# Get importances
importances = getattr(clf, "feature_importances_", None)
if importances is None:
    raise SystemExit("Classifier has no attribute 'feature_importances_'. Feature importance available only for tree-based models.")

# Try to get feature names in order after preprocessing
feature_names = []

# 1) Best: if preprocessor exists and supports get_feature_names_out
if pre is not None:
    try:
        # sklearn ColumnTransformer/Transformer support get_feature_names_out in newer versions
        feature_names = list(pre.get_feature_names_out())
    except Exception:
        # try to use transformers_ to build names manually
        try:
            # transformers_ is a list of (name, transformer, columns)
            names = []
            for tname, transformer, cols in pre.transformers_:
                # cols may be slice/list/ndarray of column names
                if hasattr(cols, "__iter__") and not isinstance(cols, str):
                    # If transformer has get_feature_names_out (like OneHotEncoder), try to call it
                    try:
                        # create a simple mask row to get output names
                        if hasattr(transformer, "get_feature_names_out"):
                            out = transformer.get_feature_names_out(cols)
                            names.extend(list(out))
                        else:
                            names.extend(list(cols))
                    except Exception:
                        names.extend(list(cols))
                else:
                    # last transformer may be 'remainder' or similar
                    pass
            feature_names = names
        except Exception:
            feature_names = []

# 2) fallback: check pipeline attribute feature_names_in_
if not feature_names:
    if hasattr(pipe, "feature_names_in_"):
        feature_names = list(getattr(pipe, "feature_names_in_"))

# 3) final fallback: try to read pre.transformers_ as raw lists (no one-hot)
if not feature_names and pre is not None:
    try:
        names = []
        for _, _, cols in pre.transformers_:
            if hasattr(cols, "__iter__") and not isinstance(cols, str):
                names.extend(list(cols))
        feature_names = names
    except Exception:
        feature_names = []

# If we still have no feature names, fallback to a generic numbering
if not feature_names:
    warnings.warn("Could not determine feature names from preprocessor. Using generic feature_0..N-1.")
    feature_names = [f"feature_{i}" for i in range(len(importances))]

# Align lengths: if mismatch, truncate or pad feature_names to match importances length
if len(feature_names) != len(importances):
    warnings.warn(
        f"Mismatch between inferred feature names ({len(feature_names)}) "
        f"and importance vector ({len(importances)}). Will align by truncating/padding."
    )
    if len(feature_names) > len(importances):
        feature_names = feature_names[: len(importances)]
    else:
        # pad with generic names
        start = len(feature_names)
        feature_names = feature_names + [f"pad_feature_{i}" for i in range(start, len(importances))]

# Build DataFrame
df = pd.DataFrame({"feature": feature_names, "importance": importances})
df = df.sort_values("importance", ascending=False).reset_index(drop=True)

# Save outputs
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV, index=False)

# Plot top 15 features
topn = min(15, len(df))
plt.figure(figsize=(8, max(4, 0.35 * topn)))
plt.barh(df["feature"].iloc[:topn].iloc[::-1], df["importance"].iloc[:topn].iloc[::-1])
plt.xlabel("Importance")
plt.title("Top feature importances")
plt.tight_layout()
plt.savefig(OUT_PNG, bbox_inches="tight")

print(f"Saved CSV -> {OUT_CSV}")
print(f"Saved PNG -> {OUT_PNG}")
