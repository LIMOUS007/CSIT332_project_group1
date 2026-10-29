"""
prepare_adult.py
================
CSIT332 Week 2 – Data Preparation Pipeline
Dataset : Adult / Census Income  (OpenML ID 1590)
Author  : Sheikh Mohsin
Seed    : 332

Run
---
    python prep/prepare_adult.py

Outputs (written to data/, which is git-ignored)
-------------------------------------------------
    data/adult_clean.csv   – cleaned feature matrix (f1..fk) + target y
    data/adult_split.json  – frozen 70/30 stratified split indices
"""

# ---------------------------------------------------------------------------
# 0. Imports
# ---------------------------------------------------------------------------
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import openml
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# 1. Constants
# ---------------------------------------------------------------------------
NAME        = "adult"
OPENML_ID   = 1590
SEED        = 332
TEST_SIZE   = 0.30
MISSING_CAP = 0.30          # drop columns exceeding this missing fraction
DROP_COLS   = ["education-num"]   # documented duplicate – see docs/dataset_register.md

REPO_ROOT   = Path(__file__).resolve().parent.parent
DATA_DIR    = REPO_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

OUT_CSV     = DATA_DIR / "adult_clean.csv"
OUT_SPLIT   = DATA_DIR / "adult_split.json"

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
X_raw, y_raw, _, attr_names = dataset.get_data(
    dataset_format="dataframe",
    target=dataset.default_target_attribute,
)
print(f"      Raw shape : {X_raw.shape}  |  target name : '{dataset.default_target_attribute}'")

# ---------------------------------------------------------------------------
# 3. Inspection logging
# ---------------------------------------------------------------------------
print("\n[2/9] Inspection ...")
print(f"      Rows x Cols : {X_raw.shape[0]} x {X_raw.shape[1]}")
print("\n      --- dtypes ---")
print(X_raw.dtypes.to_string())

# Recode sentinel '?' to NaN for accurate missingness accounting
X_inspect = X_raw.replace("?", np.nan)
miss_frac = X_inspect.isnull().mean()
print("\n      --- missing % (after recoding '?') ---")
nonzero_miss = miss_frac[miss_frac > 0]
if len(nonzero_miss):
    print(nonzero_miss.apply(lambda v: f"{v:.2%}").to_string())
else:
    print("      None")

print(f"\n      --- raw target distribution ---")
print(y_raw.value_counts(dropna=False).to_string())

# ---------------------------------------------------------------------------
# 4. Binarise target
# ---------------------------------------------------------------------------
print("\n[3/9] Binarising target ...")
y = (y_raw.astype(str).str.strip().str.rstrip(".") == ">50K").astype(int)
n_pos = y.sum()
prevalence = n_pos / len(y)
print(f"      Positive class (>50K) : {n_pos:,}  |  Prevalence : {prevalence:.4f}")
assert abs(prevalence - 0.2393) < 0.001, (
    f"Unexpected prevalence {prevalence:.4f}; expected ~0.2393"
)

# Frozen 70/30 stratified split. It only needs y, so it comes first: every
# later step that learns from the data (missing-column cap, imputation fills,
# constant columns, standardisation) uses pool rows only, so the frozen test
# rows never influence the cleaned values. No rows are dropped after this.
print(f"\n      Computing frozen 70/30 stratified split (seed={SEED}) ...")
pool_idx, test_idx = train_test_split(
    np.arange(len(y)),
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=y,
)

pool_prev = y.iloc[pool_idx].mean()
test_prev = y.iloc[test_idx].mean()
print(f"      Pool size : {len(pool_idx):,}  |  prevalence = {pool_prev:.4f}")
print(f"      Test size : {len(test_idx):,}  |  prevalence = {test_prev:.4f}")

assert abs(pool_prev - prevalence) < 0.005, "Pool prevalence drifts > 0.5 pp from overall"
assert abs(test_prev - prevalence) < 0.005, "Test prevalence drifts > 0.5 pp from overall"
print("      OK  Pool and test prevalences match overall prevalence (within 0.5 pp)")

# ---------------------------------------------------------------------------
# 5. Drop documented duplicate column
# ---------------------------------------------------------------------------
print(f"\n[4/9] Dropping columns {DROP_COLS} ...")
X = X_raw.drop(columns=[c for c in DROP_COLS if c in X_raw.columns])
print(f"      Shape after drop : {X.shape}")

# ---------------------------------------------------------------------------
# 6. Handle missing values
# ---------------------------------------------------------------------------
print("\n[5/9] Handling missing values ...")

# Recode sentinel
X = X.replace("?", np.nan)

# Check >30% missing threshold (measured on the pool)
miss_frac = X.iloc[pool_idx].isnull().mean()
high_miss = miss_frac[miss_frac > MISSING_CAP]
if not high_miss.empty:
    print(f"      WARNING: Dropping {list(high_miss.index)} (>{MISSING_CAP:.0%} missing in the pool)")
    X = X.drop(columns=high_miss.index.tolist())
else:
    print("      No column exceeds the 30% missing cap - none dropped.")

# Separate numeric and categorical
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

# Impute numeric -> pool median
imputed_num = 0
for col in num_cols:
    n = X[col].isnull().sum()
    if n > 0:
        X[col] = X[col].fillna(X[col].iloc[pool_idx].median())
        print(f"      Imputed {col:30s} (numeric , median) : {n} values")
        imputed_num += n

# Impute categorical -> pool mode
imputed_cat = 0
for col in cat_cols:
    n = X[col].isnull().sum()
    if n > 0:
        mode_val = X[col].iloc[pool_idx].mode()[0]
        X[col] = X[col].fillna(mode_val)
        print(f"      Imputed {col:30s} (categ.  , mode={mode_val!r}) : {n} values")
        imputed_cat += n

print(f"      Total imputed: {imputed_num} numeric + {imputed_cat} categorical cells")

# ---------------------------------------------------------------------------
# 7. Encode categorical features
# ---------------------------------------------------------------------------
print("\n[6/9] Encoding categorical features ...")
for col in cat_cols:
    card = X[col].nunique()
    print(f"      {col:30s}  cardinality={card}")
    if card >= 50:
        print(f"        WARNING: cardinality >= 50; proceeding anyway.")

n_before = X.shape[1]
X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=float)
n_after  = X.shape[1]
print(f"      Columns: {n_before} -> {n_after}  (+{n_after - n_before} dummies from {len(cat_cols)} categorical cols)")

# ---------------------------------------------------------------------------
# 8. Float conversion & drop zero-variance columns
# ---------------------------------------------------------------------------
print("\n[7/9] Casting to float and removing zero-variance columns ...")
X = X.astype(float)

stds = X.iloc[pool_idx].std(ddof=0)
const_cols = stds[stds == 0].index.tolist()
if const_cols:
    print(f"      Dropping columns constant in the pool: {const_cols}")
    X = X.drop(columns=const_cols)
else:
    print("      No constant columns found.")

print(f"      Shape after float cast + constant drop: {X.shape}")

# ---------------------------------------------------------------------------
# 9. Standardise (population std) and rename columns f1..fk
# ---------------------------------------------------------------------------
print("\n[8/9] Standardising with pool mean / std and renaming columns ...")
col_means = X.iloc[pool_idx].mean()
col_stds  = X.iloc[pool_idx].std(ddof=0)
X = (X - col_means) / col_stds

k = X.shape[1]
rename_map = {old: f"f{i+1}" for i, old in enumerate(X.columns)}
X.rename(columns=rename_map, inplace=True)
print(f"      Feature count k = {k}  ->  columns f1 ... f{k}")

# ---------------------------------------------------------------------------
# 10. Sanity checks
# ---------------------------------------------------------------------------
print("\n[9/9] Running sanity checks ...")

nan_count = X.isnull().sum().sum()
assert nan_count == 0, f"NaN values remaining: {nan_count}"
print(f"      OK  Zero NaN values")

assert set(y.unique()) == {0, 1}, f"Unexpected y values: {y.unique()}"
print(f"      OK  y in {{0, 1}}")

non_float = [c for c in X.columns if X[c].dtype != np.float64]
assert len(non_float) == 0, f"Non-float columns: {non_float}"
print(f"      OK  All features are float64")

assert np.isfinite(X.to_numpy()).all(), "Non-finite values remain"
print(f"      OK  All values finite")

mean_err = X.iloc[pool_idx].mean().abs().max()
std_err  = (X.iloc[pool_idx].std(ddof=0) - 1.0).abs().max()
assert mean_err < 1e-6, f"Pool mean not ~0; max abs mean = {mean_err}"
assert std_err  < 1e-6, f"Pool std  not ~1; max abs std err = {std_err}"
print(f"      OK  Pool standardisation check: max|mean|={mean_err:.2e}, max|std-1|={std_err:.2e}")

# ---------------------------------------------------------------------------
# 11. Save cleaned dataset
# ---------------------------------------------------------------------------
print("\n[Export] Saving cleaned dataset ...")
df_out = X.copy()
df_out["y"] = y.values
df_out.to_csv(OUT_CSV, index=False)
print(f"      Saved : {OUT_CSV}")
print(f"      Shape : {df_out.shape}  (rows x cols including y)")

# Quick hash for reproducibility logging
csv_md5 = hashlib.md5(OUT_CSV.read_bytes()).hexdigest()
print(f"      MD5   : {csv_md5}")

# ---------------------------------------------------------------------------
# 12. Save the frozen split (computed in step 4)
# ---------------------------------------------------------------------------
print("\n[Split] Saving frozen 70/30 split ...")
split_record = {
    "dataset"   : NAME,
    "openml_id" : OPENML_ID,
    "seed"      : SEED,
    "test_size" : TEST_SIZE,
    "n_total"   : int(len(y)),
    "n_pool"    : int(len(pool_idx)),
    "n_test"    : int(len(test_idx)),
    "prevalence": float(round(prevalence, 6)),
    "pool_prev" : float(round(pool_prev,  6)),
    "test_prev" : float(round(test_prev,  6)),
    "csv_md5"   : csv_md5,
    "pool_idx"  : sorted(pool_idx.tolist()),
    "test_idx"  : sorted(test_idx.tolist()),
}
with open(OUT_SPLIT, "w") as fh:
    json.dump(split_record, fh, indent=2)
print(f"      Saved : {OUT_SPLIT}")

# ---------------------------------------------------------------------------
# 13. Final summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("DONE - Adult preparation complete.")
print(f"  Cleaned CSV  : {OUT_CSV.name}  {df_out.shape}")
print(f"  Split JSON   : {OUT_SPLIT.name}")
print(f"  Features (k) : {k}")
print(f"  Prevalence   : {prevalence:.4f}")
print(f"  Seed         : {SEED}")
print("=" * 60)
