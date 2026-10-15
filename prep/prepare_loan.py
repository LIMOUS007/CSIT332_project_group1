"""
prepare_loan.py
===============
CSIT332 Week 2 - Data Preparation Pipeline
Dataset : Loan Approval Status  (OpenML ID 46526)
Author  : Soummil Goel
Seed    : 42

Run
---
    python prep/prepare_loan.py

Outputs (written to data/)
--------------------------
    data/loan_clean.csv   - cleaned feature matrix (f1..fk) + target y
    data/loan_split.json  - frozen 70/30 stratified split indices
"""

# ---------------------------------------------------------------------------
# 0. Imports
# ---------------------------------------------------------------------------
import hashlib
import json
from pathlib import Path

import numpy as np
import openml
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# 1. Constants
# ---------------------------------------------------------------------------
NAME        = "loan"
OPENML_ID   = 46526
SEED        = 42
TEST_SIZE   = 0.30

# OpenML ships the categoricals already one-hot encoded with every level kept,
# so each group sums to 1. Drop one level per group (the alphabetically first,
# matching get_dummies(drop_first=True) in the other scripts).
# column -> reason. Logged in docs/cleaning_decisions.md.
DROP_COLS = {
    "defaults_no"             : "reference level of defaults_* (= 1 - defaults_yes)",
    "gender_female"           : "reference level of gender_*",
    "home_mortgage"           : "reference level of home_*",
    "intent_debtconsolidation": "reference level of intent_*",
}
ONE_HOT_GROUPS = ["defaults_", "gender_", "home_", "intent_"]

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = REPO_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

OUT_CSV   = DATA_DIR / f"{NAME}_clean.csv"
OUT_SPLIT = DATA_DIR / f"{NAME}_split.json"

# ---------------------------------------------------------------------------
# 2. Download from OpenML
# ---------------------------------------------------------------------------
print("=" * 60)
print(f"[1/9] Downloading {NAME} (OpenML {OPENML_ID}) ...")
dataset = openml.datasets.get_dataset(
    OPENML_ID,
    download_data=True,
    download_qualities=False,
    download_features_meta_data=False,
)
X_raw, y_raw, _, _ = dataset.get_data(
    dataset_format="dataframe",
    target=dataset.default_target_attribute,
)
print(f"      Raw shape : {X_raw.shape}  |  target name : '{dataset.default_target_attribute}'")

# ---------------------------------------------------------------------------
# 3. Binarise target
# ---------------------------------------------------------------------------
print("\n[2/9] Binarising target ...")
print(y_raw.value_counts(dropna=False).to_string())
y = (y_raw.astype(str).str.strip() == "loan_approved").astype(int).reset_index(drop=True)
prevalence = y.mean()
print(f"      Positive class (approved) : {y.sum():,}  |  Prevalence : {prevalence:.4f}")
assert abs(prevalence - 0.2222) < 0.001, f"Unexpected prevalence {prevalence:.4f}"

# ---------------------------------------------------------------------------
# 4. Drop redundant one-hot reference levels
# ---------------------------------------------------------------------------
print("\n[3/9] Dropping redundant one-hot reference levels ...")
X = X_raw.reset_index(drop=True)
for g in ONE_HOT_GROUPS:
    cols = [c for c in X.columns if c.startswith(g)]
    assert (X[cols].sum(axis=1) == 1).all(), f"{g}* is not a full one-hot group"
for c, why in DROP_COLS.items():
    print(f"      {c:25s} : {why}")
X = X.drop(columns=list(DROP_COLS))

# ---------------------------------------------------------------------------
# 5. Handle missing values
# ---------------------------------------------------------------------------
print("\n[4/9] Handling missing values ...")
n_missing = int(X.isna().sum().sum())
print(f"      Missing cells: {n_missing}  |  rows dropped: 0")
assert n_missing == 0

# person_age (max 144) and person_emp_exp (max 125) hold impossible values.
# They are left as-is and flagged in docs/cleaning_decisions.md.
print(f"      person_age > 100     : {(X['person_age'] > 100).sum()} rows (kept, flagged)")
print(f"      person_emp_exp > 60  : {(X['person_emp_exp'] > 60).sum()} rows (kept, flagged)")

# ---------------------------------------------------------------------------
# 6. Encode categorical features
# ---------------------------------------------------------------------------
print("\n[5/9] Encoding categorical features ...")
print("      None - categoricals arrive one-hot encoded; person_education is an "
      "ordinal 0-4 code, kept numeric")

# ---------------------------------------------------------------------------
# 7. Float conversion & drop zero-variance columns
# ---------------------------------------------------------------------------
print("\n[6/9] Casting to float and removing zero-variance columns ...")
X = X.astype(float)
stds = X.std(ddof=0)
const_cols = stds[stds == 0].index.tolist()
print(f"      Dropping constant columns: {const_cols or 'none'}")
X = X.drop(columns=const_cols)

# ---------------------------------------------------------------------------
# 8. Standardise (population std) and rename columns f1..fk
# ---------------------------------------------------------------------------
print("\n[7/9] Standardising (mu=0, sigma=1) and renaming columns ...")
X = (X - X.mean()) / X.std(ddof=0)
k = X.shape[1]
X.columns = [f"f{i}" for i in range(1, k + 1)]
print(f"      Feature count k = {k}  ->  columns f1 ... f{k}")

# ---------------------------------------------------------------------------
# 9. Sanity checks
# ---------------------------------------------------------------------------
print("\n[8/9] Running sanity checks ...")
assert X.isna().sum().sum() == 0, "NaN values remain"
assert set(y.unique()) == {0, 1}, f"Unexpected y values: {y.unique()}"
assert all(X.dtypes == np.float64), "Non-float columns remain"
assert X.mean().abs().max() < 1e-6, "Mean not ~0"
assert (X.std(ddof=0) - 1).abs().max() < 1e-6, "Std not ~1"
print("      OK  no NaN | y in {0,1} | all float64 | mean 0, std 1")

df_out = X.copy()
df_out["y"] = y.values
df_out.to_csv(OUT_CSV, index=False)
csv_md5 = hashlib.md5(OUT_CSV.read_bytes()).hexdigest()
print(f"      Saved : {OUT_CSV}  {df_out.shape}  MD5 {csv_md5}")

# ---------------------------------------------------------------------------
# 10. Frozen 70/30 stratified split
# ---------------------------------------------------------------------------
print(f"\n[9/9] Computing frozen 70/30 stratified split (seed={SEED}) ...")
pool_idx, test_idx = train_test_split(
    np.arange(len(y)), test_size=TEST_SIZE, random_state=SEED, stratify=y,
)
pool_prev = y.iloc[pool_idx].mean()
test_prev = y.iloc[test_idx].mean()
print(f"      Pool size : {len(pool_idx):,}  |  prevalence = {pool_prev:.4f}")
print(f"      Test size : {len(test_idx):,}  |  prevalence = {test_prev:.4f}")
assert abs(pool_prev - prevalence) < 0.005 and abs(test_prev - prevalence) < 0.005

split_record = {
    "dataset"   : NAME,
    "openml_id" : OPENML_ID,
    "seed"      : SEED,
    "test_size" : TEST_SIZE,
    "n_total"   : int(len(y)),
    "n_pool"    : int(len(pool_idx)),
    "n_test"    : int(len(test_idx)),
    "prevalence": float(round(prevalence, 6)),
    "pool_prev" : float(round(pool_prev, 6)),
    "test_prev" : float(round(test_prev, 6)),
    "csv_md5"   : csv_md5,
    "pool_idx"  : sorted(pool_idx.tolist()),
    "test_idx"  : sorted(test_idx.tolist()),
}
with open(OUT_SPLIT, "w") as fh:
    json.dump(split_record, fh, indent=2)
print(f"      Saved : {OUT_SPLIT}")

print("\n" + "=" * 60)
print(f"PROFILE | {NAME} | rows {len(y):,} | features {k} | "
      f"prevalence {prevalence:.4f} | seed {SEED}")
print("=" * 60)
