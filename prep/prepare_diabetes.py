"""
prepare_diabetes.py
===================
CSIT332 Week 2 - Data Preparation Pipeline
Dataset : Diabetes 130-US Hospitals  (OpenML ID 4541)
Author  : Soummil Goel
Seed    : 42

Run
---
    python prep/prepare_diabetes.py

Outputs (written to data/, which is git-ignored)
-------------------------------------------------
    data/diabetes_clean.csv   - cleaned feature matrix (f1..fk) + target y
    data/diabetes_split.json  - frozen 70/30 stratified split indices
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
NAME        = "diabetes"
OPENML_ID   = 4541
SEED        = 42
TEST_SIZE   = 0.30
MISSING_CAP = 0.30          # drop columns exceeding this missing fraction

# Identifier columns - never features.
ID_COLS = ["encounter_id", "patient_nbr"]

# Columns whose "None" means "test not performed". The register counts that as
# missing (94.7% and 83.3%), so they fall under the 30% rule and are dropped.
NONE_AS_MISSING = ["max_glu_serum", "A1Cresult"]

# Numeric-looking ID codes that are really categories (lookup tables in the
# UCI documentation), so they are one-hot encoded rather than standardised.
CODE_COLS = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]

DIAG_COLS = ["diag_1", "diag_2", "diag_3"]

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = REPO_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

OUT_CSV   = DATA_DIR / f"{NAME}_clean.csv"
OUT_SPLIT = DATA_DIR / f"{NAME}_split.json"


def icd9_group(code):
    """Map an ICD-9 code to one of the 9 groups used by Strack et al. (2014),
    the paper that published this dataset. Cuts ~700 levels per column to 9."""
    if pd.isna(code):
        return np.nan
    code = str(code)
    if code.startswith(("V", "E")):
        return "Other"
    v = float(code)
    if int(v) == 250:
        return "Diabetes"
    if 390 <= v <= 459 or int(v) == 785:
        return "Circulatory"
    if 460 <= v <= 519 or int(v) == 786:
        return "Respiratory"
    if 520 <= v <= 579 or int(v) == 787:
        return "Digestive"
    if 800 <= v <= 999:
        return "Injury"
    if 710 <= v <= 739:
        return "Musculoskeletal"
    if 580 <= v <= 629 or int(v) == 788:
        return "Genitourinary"
    if 140 <= v <= 239:
        return "Neoplasms"
    return "Other"


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
print(f"      Unique patients : {X_raw['patient_nbr'].nunique():,} "
      f"(some patients have several encounters)")

# ---------------------------------------------------------------------------
# 3. Binarise target
# ---------------------------------------------------------------------------
print("\n[2/9] Binarising target ...")
print(y_raw.value_counts(dropna=False).to_string())
y = (y_raw.astype(str).str.strip() == "<30").astype(int)
y = y.reset_index(drop=True)
prevalence = y.mean()
print(f"      Positive class (<30) : {y.sum():,}  |  Prevalence : {prevalence:.4f}")
assert abs(prevalence - 0.1116) < 0.001, f"Unexpected prevalence {prevalence:.4f}"

# Frozen 70/30 stratified split. It only needs y, so it comes first: every
# later step that learns from the data (missing-column cap, imputation fills,
# constant columns, standardisation) uses pool rows only, so the frozen test
# rows never influence the cleaned values. No rows are dropped after this.
print(f"\n      Computing frozen 70/30 stratified split (seed={SEED}) ...")
pool_idx, test_idx = train_test_split(
    np.arange(len(y)), test_size=TEST_SIZE, random_state=SEED, stratify=y,
)
pool_prev = y.iloc[pool_idx].mean()
test_prev = y.iloc[test_idx].mean()
print(f"      Pool size : {len(pool_idx):,}  |  prevalence = {pool_prev:.4f}")
print(f"      Test size : {len(test_idx):,}  |  prevalence = {test_prev:.4f}")
assert abs(pool_prev - prevalence) < 0.005 and abs(test_prev - prevalence) < 0.005

# ---------------------------------------------------------------------------
# 4. Drop identifier columns
# ---------------------------------------------------------------------------
print(f"\n[3/9] Dropping identifier columns {ID_COLS} ...")
X = X_raw.drop(columns=ID_COLS).reset_index(drop=True)
# work on plain object dtype so new labels can be assigned freely
for c in X.select_dtypes(include="category").columns:
    X[c] = X[c].astype(object)

# ---------------------------------------------------------------------------
# 5. Handle missing values
# ---------------------------------------------------------------------------
print("\n[4/9] Handling missing values ...")
X = X.replace("?", np.nan)
X["gender"] = X["gender"].replace("Unknown/Invalid", np.nan)
for c in NONE_AS_MISSING:
    X[c] = X[c].replace("None", np.nan)

miss_frac = X.iloc[pool_idx].isna().mean()
high_miss = miss_frac[miss_frac > MISSING_CAP].sort_values(ascending=False)
print("      Dropping (>30% missing in the pool):")
print(high_miss.apply(lambda v: f"{v:.2%}").to_string())
X = X.drop(columns=high_miss.index.tolist())

# Diagnosis codes -> 9 ICD-9 groups (missing stays missing until imputation)
for c in DIAG_COLS:
    X[c] = X[c].map(icd9_group)

# Age bracket "[70-80)" -> midpoint 75 (ordinal, keeps one numeric column)
X["age"] = X["age"].str.extract(r"\[(\d+)-")[0].astype(float) + 5

for c in CODE_COLS:
    X[c] = X[c].astype(str)

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

imputed = 0
for col in X.columns:
    n = X[col].isna().sum()
    if n == 0:
        continue
    pool_col = X[col].iloc[pool_idx]
    fill = pool_col.median() if col in num_cols else pool_col.mode()[0]
    X[col] = X[col].fillna(fill)
    print(f"      Imputed {col:30s} with {fill!r} : {n} values")
    imputed += n
print(f"      Total imputed: {imputed} cells  |  rows dropped: 0")

# ---------------------------------------------------------------------------
# 6. Encode categorical features
# ---------------------------------------------------------------------------
print("\n[5/9] Encoding categorical features ...")
for col in cat_cols:
    card = X[col].nunique()
    flag = "  (constant - will be dropped)" if card == 1 else ""
    print(f"      {col:30s}  cardinality={card}{flag}")

n_before = X.shape[1]
X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=float)
n_after = X.shape[1]
print(f"      Columns: {n_before} -> {n_after}  "
      f"({len(cat_cols)} categorical cols -> {n_after - len(num_cols)} dummies)")

# ---------------------------------------------------------------------------
# 7. Float conversion & drop zero-variance columns
# ---------------------------------------------------------------------------
print("\n[6/9] Casting to float and removing zero-variance columns ...")
X = X.astype(float)
stds = X.iloc[pool_idx].std(ddof=0)
const_cols = stds[stds == 0].index.tolist()
print(f"      Dropping columns constant in the pool: {const_cols or 'none'}")
X = X.drop(columns=const_cols)

# ---------------------------------------------------------------------------
# 8. Standardise (population std) and rename columns f1..fk
# ---------------------------------------------------------------------------
print("\n[7/9] Standardising with pool mean / std and renaming columns ...")
pool_X = X.iloc[pool_idx]
X = (X - pool_X.mean()) / pool_X.std(ddof=0)
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
assert np.isfinite(X.to_numpy()).all(), "Non-finite values remain"
assert X.iloc[pool_idx].mean().abs().max() < 1e-6, "Pool mean not ~0"
assert (X.iloc[pool_idx].std(ddof=0) - 1).abs().max() < 1e-6, "Pool std not ~1"
print("      OK  no NaN | y in {0,1} | all float64 | pool mean 0, std 1")

df_out = X.copy()
df_out["y"] = y.values
df_out.to_csv(OUT_CSV, index=False)
csv_md5 = hashlib.md5(OUT_CSV.read_bytes()).hexdigest()
print(f"      Saved : {OUT_CSV}  {df_out.shape}  MD5 {csv_md5}")

# ---------------------------------------------------------------------------
# 10. Save the frozen split (computed in step 3)
# ---------------------------------------------------------------------------
print("\n[9/9] Saving frozen 70/30 split ...")
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
