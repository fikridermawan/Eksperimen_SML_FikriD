"""
generate_dataset.py
===================
Script helper untuk mengunduh dataset Credit Risk dari sumber publik
(Kaggle / GitHub) atau membuat dataset dummy untuk keperluan testing.

CATATAN:
Dataset asli yang direkomendasikan:
  → "Credit Risk Dataset" oleh Laotse
  → https://www.kaggle.com/datasets/laotse/credit-risk-dataset

Menjalankan:
    # Opsi 1: Unduh dari Kaggle (perlu Kaggle API key)
    pip install kaggle
    kaggle datasets download -d laotse/credit-risk-dataset
    unzip credit-risk-dataset.zip

    # Opsi 2: Jalankan script ini untuk dataset dummy
    python generate_dataset.py
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 32581

person_age               = np.random.randint(20, 65, N)
person_income            = np.random.randint(10000, 120000, N).astype(float)
person_home_ownership    = np.random.choice(["RENT", "OWN", "MORTGAGE", "OTHER"], N)
person_emp_length        = np.random.choice(
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, np.nan], N
)
loan_intent              = np.random.choice(
    ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"], N
)
loan_grade               = np.random.choice(["A", "B", "C", "D", "E", "F", "G"], N)
loan_amnt                = np.random.randint(500, 35000, N).astype(float)
loan_int_rate            = np.round(np.random.uniform(5.42, 23.22, N), 2)
loan_int_rate[np.random.choice(N, 3116, replace=False)] = np.nan
loan_percent_income      = np.round(loan_amnt / person_income, 2)
cb_person_default_on_file = np.random.choice(["N", "Y"], N, p=[0.82, 0.18])
cb_person_cred_hist_length = np.random.randint(2, 30, N).astype(float)
loan_status              = np.random.choice([0, 1], N, p=[0.78, 0.22])

df = pd.DataFrame({
    "person_age"               : person_age,
    "person_income"            : person_income,
    "person_home_ownership"    : person_home_ownership,
    "person_emp_length"        : person_emp_length,
    "loan_intent"              : loan_intent,
    "loan_grade"               : loan_grade,
    "loan_amnt"                : loan_amnt,
    "loan_int_rate"            : loan_int_rate,
    "loan_percent_income"      : loan_percent_income,
    "cb_person_default_on_file": cb_person_default_on_file,
    "cb_person_cred_hist_length": cb_person_cred_hist_length,
    "loan_status"              : loan_status,
})

output_path = "credit_risk_dataset_raw.csv"
df.to_csv(output_path, index=False)
print(f"Dataset dummy disimpan: {output_path}")
print(f"Shape: {df.shape}")
print(df.head())
