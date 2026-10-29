"""
prepare_kick.py
===============
CSIT332 Week 2 - Data Preparation Pipeline
Dataset : Don't Get Kicked  (OpenML ID 41162)
Author  : Soummil Goel
Seed    : 42

Run
---
    python prep/prepare_kick.py

Outputs (written to data/, which is git-ignored)
-------------------------------------------------
    data/kick_clean.csv   - cleaned feature matrix (f1..fk) + target y
    data/kick_split.json  - frozen 70/30 stratified split indices
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
NAME        = "kick"
OPENML_ID   = 41162
SEED        = 42
TEST_SIZE   = 0.30
MISSING_CAP = 0.30          # drop columns exceeding this missing fraction
TOP_N       = 20            # high-cardinality columns keep their top 20 levels
HIGH_CARD   = 50            # ... when they have more than this many levels

# column -> reason. Logged in docs/cleaning_decisions.md.
DROP_COLS = {
    "BYRNO"      : "buyer identifier",
    "VNZIP1"     : "vendor ZIP code - identifier-like, location already in VNST",
    "WheelTypeID": "numeric code duplicating WheelType",
    "VehYear"    : "duplicates VehicleAge (purchase year minus VehYear)",
}

# Missing WheelType is known to be strongly linked to bad buys, so a missing
# value becomes its own category instead of being filled with the mode.
MISSING_AS_LEVEL = ["WheelType", "Trim"]

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
# 3. Binarise target (already 0/1, stored as a category)
# ---------------------------------------------------------------------------
print("\n[2/9] Binarising target ...")
print(y_raw.value_counts(dropna=False).to_string())
y = (y_raw.astype(str).str.strip() == "1").astype(int).reset_index(drop=True)
prevalence = y.mean()
print(f"      Positive class (bad buy) : {y.sum():,}  |  Prevalence : {prevalence:.4f}")
assert abs(prevalence - 0.1230) < 0.001, f"Unexpected prevalence {prevalence:.4f}"

# Frozen 70/30 stratified split. It only needs y, so it comes first: every
# later step that learns from the data (missing-column cap, imputation fills,
# top-N levels, constant columns, standardisation) uses pool rows only, so the
# frozen test rows never influence the cleaned values. No rows are dropped
# after this.
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
# 4. Drop identifier / duplicate columns
# ---------------------------------------------------------------------------
print("\n[3/9] Dropping identifier / duplicate columns ...")
for c, why in DROP_COLS.items():
    print(f"      {c:15s} : {why}")
X = X_raw.drop(columns=list(DROP_COLS)).reset_index(drop=True)
for c in X.select_dtypes(include="category").columns:
    X[c] = X[c].astype(object)

# "Manual" and "MANUAL" are the same transmission
X["Transmission"] = X["Transmission"].str.upper()
X["IsOnlineSale"] = X["IsOnlineSale"].astype(float)

# ---------------------------------------------------------------------------
# 5. Handle missing values
# ---------------------------------------------------------------------------
print("\n[4/9] Handling missing values ...")
miss_frac = X.iloc[pool_idx].isna().mean()
high_miss = miss_frac[miss_frac > MISSING_CAP].sort_values(ascending=False)
print("      Dropping (>30% missing in the pool):")
print(high_miss.apply(lambda v: f"{v:.2%}").to_string())
X = X.drop(columns=high_miss.index.tolist())

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

imputed = 0
for col in X.columns:
    n = X[col].isna().sum()
    if n == 0:
        continue
    if col in MISSING_AS_LEVEL:
        fill = "Missing"
    elif col in num_cols:
        fill = X[col].iloc[pool_idx].median()
    else:
        fill = X[col].iloc[pool_idx].mode()[0]
    X[col] = X[col].fillna(fill)
    print(f"      Imputed {col:35s} with {fill!r} : {n} values")
    imputed += n
print(f"      Total imputed: {imputed} cells  |  rows dropped: 0")

# ---------------------------------------------------------------------------
# 6. Encode categorical features
# ---------------------------------------------------------------------------
print("\n[5/9] Encoding categorical features ...")
for col in cat_cols:
    card = X[col].nunique()
    note = ""
    if card > HIGH_CARD:
        top = X[col].iloc[pool_idx].value_counts().index[:TOP_N]
        covered = X[col].iloc[pool_idx].isin(top).mean()
        X[col] = X[col].where(X[col].isin(top), "Other")
        note = f"  -> pool top {TOP_N} kept (cover {covered:.1%} of pool rows), rest = 'Other'"
    print(f"      {col:20s}  cardinality={card}{note}")

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
