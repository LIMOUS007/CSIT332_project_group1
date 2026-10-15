"""
prepare_covertype.py
CSIT332 Week 2 — Data Preparation Pipeline

Dataset: Covertype (OpenML ID 1596)
Target: class 2 (Lodgepole Pine) versus all other classes.
Seed: 332 (provisional; align with the group convention if needed).

Run from the repository root:
    python prep/prepare_covertype.py

Outputs:
    data/covertype_clean.csv
    data/covertype_split.json
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import openml
import pandas as pd
from sklearn.model_selection import train_test_split

# Configuration
NAME = "covertype"
OPENML_ID = 1596
SEED = 332
TEST_SIZE = 0.30

EXPECTED_ROWS = 581_012
EXPECTED_FEATURES = 54
EXPECTED_CLASS_COUNTS = {
    "1": 211_840,
    "2": 283_301,
    "3": 35_754,
    "4": 2_747,
    "5": 9_493,
    "6": 17_367,
    "7": 20_510,
}

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

OUT_CSV = DATA_DIR / f"{NAME}_clean.csv"
OUT_SPLIT = DATA_DIR / f"{NAME}_split.json"


def main():
    print("=" * 65)
    print(f"Downloading {NAME} from OpenML (ID {OPENML_ID})")

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

    # 1. Verify the source and target before preprocessing.
    print(f"Raw feature shape: {X_raw.shape}")
    print(f"Target column: {dataset.default_target_attribute}")

    assert len(X_raw) == EXPECTED_ROWS, (
        f"Expected {EXPECTED_ROWS} rows, got {len(X_raw)}. "
        "Check the downloaded OpenML dataset."
    )
    assert X_raw.shape[1] == EXPECTED_FEATURES, (
        f"Expected {EXPECTED_FEATURES} features, got {X_raw.shape[1]}."
    )
    assert dataset.default_target_attribute == "class", (
        "Unexpected target column; inspect the source before proceeding."
    )

    target_strings = y_raw.astype(str).str.strip()
    class_counts = target_strings.value_counts().to_dict()
    class_counts = {str(k): int(v) for k, v in class_counts.items()}

    assert class_counts == EXPECTED_CLASS_COUNTS, (
        f"Unexpected class counts: {class_counts}"
    )

    # 2. Binarise the target: class 2 = positive; all others = negative.
    y = (target_strings == "2").astype(np.int8).to_numpy()
    prevalence = float(y.mean())

    assert int(y.sum()) == EXPECTED_CLASS_COUNTS["2"]
    assert abs(prevalence - 0.4876) < 0.001

    print(f"Target counts: {class_counts}")
    print(f"Positive class count: {int(y.sum()):,}")
    print(f"Positive prevalence: {prevalence:.4f}")

    # 3. Confirm features are numeric and contain no missing values.
    X = X_raw.copy()

    missing_count = int(X.isna().sum().sum())
    assert missing_count == 0, (
        f"Found {missing_count} missing feature cells. "
        "Inspect and document them before proceeding."
    )

    # OpenML may expose numeric indicator columns as categorical types.
    # Convert their values to numbers, failing if any value is not numeric.
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col].astype(str), errors="raise")

    X = X.astype(np.float64)

    # 4. Remove any zero-variance features.
    stds = X.std(ddof=0)
    constant_cols = stds[stds == 0].index.tolist()

    if constant_cols:
        print(f"Dropping zero-variance columns: {constant_cols}")
        X = X.drop(columns=constant_cols)

    # 5. Standardise using population standard deviation.
    means = X.mean()
    stds = X.std(ddof=0)

    assert (stds > 0).all(), "A zero-variance feature remains."
    X = (X - means) / stds

    # Rename features without exposing original feature names.
    X.columns = [f"f{i + 1}" for i in range(X.shape[1])]

    # 6. Sanity checks.
    assert X.shape[0] == EXPECTED_ROWS
    assert X.shape[1] > 0
    assert X.isna().sum().sum() == 0
    assert np.isfinite(X.to_numpy()).all()
    assert set(np.unique(y)) == {0, 1}
    assert X.columns.tolist() == [
        f"f{i + 1}" for i in range(X.shape[1])
    ]
    assert X.mean().abs().max() < 1e-6
    assert (X.std(ddof=0) - 1).abs().max() < 1e-6

    # 7. Save clean data. CSVs are intended to remain local.
    clean_df = X.copy()
    clean_df["y"] = y
    clean_df.to_csv(OUT_CSV, index=False)

    md5 = hashlib.md5(OUT_CSV.read_bytes()).hexdigest()

    # 8. Freeze a stratified 70/30 split using row indices.
    train_idx, test_idx = train_test_split(
        np.arange(len(y)),
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )

    train_prevalence = float(y[train_idx].mean())
    test_prevalence = float(y[test_idx].mean())

    assert len(train_idx) + len(test_idx) == len(y)
    assert len(set(train_idx) & set(test_idx)) == 0
    assert abs(train_prevalence - prevalence) < 0.005
    assert abs(test_prevalence - prevalence) < 0.005

    split_data = {
        "name": NAME,
        "seed": SEED,
        "md5": md5,
        "train_indices": train_idx.tolist(),
        "test_indices": test_idx.tolist(),
    }

    with OUT_SPLIT.open("w") as f:
        json.dump(split_data, f)

    # 9. Report the actual output profile.
    print("\nSUCCESS: Covertype preparation completed.")
    print(f"Clean CSV: {OUT_CSV}")
    print(f"Split JSON: {OUT_SPLIT}")
    print(f"Rows: {len(clean_df):,}")
    print(f"Features: {X.shape[1]}")
    print(f"Prevalence: {prevalence:.4f}")
    print(f"Train/pool rows: {len(train_idx):,}")
    print(f"Test rows: {len(test_idx):,}")
    print(f"Train/pool prevalence: {train_prevalence:.4f}")
    print(f"Test prevalence: {test_prevalence:.4f}")
    print(f"CSV MD5: {md5}")
    print("=" * 65)


if __name__ == "__main__":
    main()
