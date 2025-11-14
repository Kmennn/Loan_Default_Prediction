# Loan Default Prediction - Starter Project

This starter contains a small synthetic dataset and scripts to begin a Loan Default Prediction project.

Folders:
- data/: contains loan_data_sample.csv
- src/: contains preprocessing, training, and evaluation scripts
- notebooks/: placeholder for Jupyter notebook
- reports/: report template

Usage:
1. Create a venv and install requirements:
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   pip install -r requirements.txt
2. Run basic training:
   python src/train_model.py --input data/loan_data_sample.csv --out src/models/model.joblib
3. Run EDA notebook.