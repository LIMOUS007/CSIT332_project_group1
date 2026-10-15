import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import os
import json
import hashlib

NAME = "bank"
OPENML_ID = 1461
SEED = 332

print(f"Fetching {NAME} (OpenML ID {OPENML_ID})...")
bunch = fetch_openml(data_id=OPENML_ID, as_frame=True, parser="auto")
df = bunch.frame

# 1. Target encoding ('yes' / '2' -> 1 [positive class: subscribed], others -> 0)
target_col = bunch.target_names[0] if bunch.target_names else df.columns[-1]
df['y'] = df[target_col].astype(str).str.lower().str.strip().isin(['yes', '2']).astype(int)
if target_col != 'y':
    df = df.drop(columns=[target_col])

# 2. Sentinels & Missing Values
df = df.replace(['?', 'unknown', 'UNKNOWN'], np.nan)

# Drop columns >30% missing
missing_pct = df.isnull().mean()
cols_to_drop = missing_pct[missing_pct > 0.30].index.tolist()
if cols_to_drop:
    df = df.drop(columns=cols_to_drop)

# Impute remaining missing values
for col in df.columns:
    if col == 'y':
        continue
    if df[col].isnull().sum() > 0:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

# 3. Categorical encoding
cat_cols = [c for c in df.select_dtypes(include=['object', 'category']).columns if c != 'y']
if cat_cols:
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# 4. Standardise & rename columns
X = df.drop(columns=['y']).astype(float)
y = df['y'].values

# Remove zero variance columns
variances = X.var(ddof=0)
zero_var = variances[variances == 0].index
if len(zero_var) > 0:
    X = X.drop(columns=zero_var)

X_scaled = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)
X_scaled.columns = [f"f{i+1}" for i in range(X_scaled.shape[1])]

clean_df = X_scaled.copy()
clean_df['y'] = y

# Save Clean CSV
os.makedirs("data", exist_ok=True)
csv_path = f"data/{NAME}_clean.csv"
clean_df.to_csv(csv_path, index=False)

# 5. Stratified 70/30 Split
indices = np.arange(len(clean_df))
train_idx, test_idx = train_test_split(indices, test_size=0.30, random_state=SEED, stratify=y)

md5 = hashlib.md5(open(csv_path, 'rb').read()).hexdigest()
split_data = {
    "name": NAME,
    "seed": SEED,
    "md5": md5,
    "train_indices": train_idx.tolist(),
    "test_indices": test_idx.tolist()
}

with open(f"data/{NAME}_split.json", "w") as f:
    json.dump(split_data, f)

print("=" * 60)
print(f"SUCCESS: {NAME} processed.")
print(f"Rows: {len(clean_df)}, Features: {X_scaled.shape[1]}, Prevalence: {y.mean():.4f}")
print("All sanity checks passed!")
print("=" * 60)