import pandas as pd
from pathlib import Path

def basic_preprocess(df):
    df = df.copy()
    # fill negative or zero incomes if any
    df['income'] = df['income'].apply(lambda x: x if x>0 else df['income'].median())
    # simple feature: debt_to_income
    df['debt_to_income'] = df['loan_amount'] / (df['income'] + 1)
    return df

if __name__ == "__main__":
    df = pd.read_csv('data/loan_data_sample.csv')
    df = basic_preprocess(df)
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    df.to_csv('data/processed/loan_data_processed.csv', index=False)