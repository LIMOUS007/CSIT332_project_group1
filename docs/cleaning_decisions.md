# Cleaning Decisions – Adult Dataset (OpenML 1590)

**Author:** Sheikh Mohsin  
**Date:** 2026-10-11  
**Script:** `prep/prepare_adult.py`  
**Seed:** 332

---

## Adult (OpenML 1590), prepared by Sheikh Mohsin

- **Target:** `income.strip().rstrip('.') == '>50K'` → 1, else 0; prevalence = 0.2393 (11,687 positive out of 48,842)
- **Missing values:** Sentinel `?` recoded to NaN; no column exceeded 30% missing (none dropped); filled 6,465 cells in 3 columns using mode (`workclass` 2,799, `occupation` 2,809, `native-country` 857); rows dropped: 0
- **Categoricals:** 8 columns one-hot encoded (`drop_first=True`) → 83 new dummy columns (96 total features)
- **Dropped columns:** `education-num`: integer duplicate of `education` (same information, different scale) — see `docs/dataset_register.md`
- **Final:** 48,842 rows, 96 features, prevalence 0.2393, seed 332

---



## 1. Dataset Identity

| Field | Value |
|---|---|
| Name | Adult / Census Income |
| Source | [OpenML 1590](https://www.openml.org/d/1590) / [UCI 2](https://archive.ics.uci.edu/dataset/2) |
| Raw rows | 48,842 |
| Raw features | 14 |
| Licence | CC BY 4.0 – redistribution permitted |

---

## 2. Target Binarisation

**Column:** `income`  
**Rule:** `y = (income.strip().rstrip('.') == '>50K').astype(int)`

The raw target contains two notations for the same class (`>50K` in the training split and `>50K.` in the test split). Stripping trailing periods before comparison is essential; omitting it yields 16.05% prevalence instead of the correct 23.93%.

| Metric | Value |
|---|---|
| Positive class | `>50K` (annual income exceeds $50 K) |
| Negative class | `<=50K` |
| Positive count | 11,687 |
| Total rows | 48,842 |
| **Prevalence** | **0.2393** |

---

## 3. Dropped Columns

| Column | Reason |
|---|---|
| `education-num` | Integer encoding of `education` – exact duplicate. Documented in `docs/dataset_register.md`. Removing it avoids multicollinearity and redundant information leakage. |

---

## 4. Missing Value Treatment

The Adult dataset uses `?` as a sentinel for missing values in three columns. These were recoded to `NaN` before imputation.

**Missing value threshold:** Columns with > 30% missing are dropped. No column in this dataset exceeded the threshold.

| Column | Missing count | Missing % | Imputation strategy | Value used |
|---|---|---|---|---|
| `workclass` | 2,799 | 5.73% | Mode | `Private` |
| `occupation` | 2,809 | 5.75% | Mode | `Prof-specialty` |
| `native-country` | 857 | 1.75% | Mode | `United-States` |

**Total imputed cells:** 6,465 (0 numeric + 6,465 categorical)

---

## 5. Categorical Encoding

All string/object columns were one-hot encoded using `pd.get_dummies(drop_first=True, dtype=float)`. `drop_first=True` removes one level per column to avoid the dummy-variable trap.

| Column | Unique levels | Dummies created |
|---|---|---|
| `workclass` | 8 | 7 |
| `education` | 16 | 15 |
| `marital-status` | 7 | 6 |
| `occupation` | 14 | 13 |
| `relationship` | 6 | 5 |
| `race` | 5 | 4 |
| `sex` | 2 | 1 |
| `native-country` | 41 | 40 |

**Total:** 8 categorical columns → 83 dummy features (96 total columns after combining with 13 numerics and dropping 1 original col)  
All cardinalities are < 50; no high-cardinality warning was triggered.

---

## 6. Numeric Feature Count (pre-encoding)

After dropping `education-num`, the 5 numeric columns retained were:

`age`, `fnlwgt`, `capital-gain`, `capital-loss`, `hours-per-week`

---

## 7. Zero-Variance Columns

After one-hot encoding and float casting, all columns were checked for zero variance (std == 0). **No constant columns were found** and none were dropped.

---

## 8. Standardisation

All features were standardised using population statistics (ddof=0):

```
X_std = (X - X.mean()) / X.std(ddof=0)
```

Post-standardisation checks confirmed:
- `max|mean| < 1e-6`
- `max|std − 1| < 1e-6`

Columns were then renamed `f1, f2, …, f96`.

---

## 9. Final Dataset Shape

| Metric | Value |
|---|---|
| Rows | 48,842 |
| Feature columns | 96 (f1 … f96) |
| Target column | y |
| Total columns in CSV | 97 |
| Prevalence | 0.2393 |
| NaN values | 0 |

---

## 10. Frozen Stratified Split

| Parameter | Value |
|---|---|
| Seed | 332 |
| Test fraction | 30% |
| Pool (train+val) size | 34,189 |
| Test size | 14,653 |
| Pool prevalence | ≈ 0.2393 |
| Test prevalence | ≈ 0.2393 |
| Split file | `data/adult_split.json` |

The split indices are frozen at generation time. Running `prepare_adult.py` a second time produces bit-identical `adult_clean.csv` and `adult_split.json` files (verified by MD5 hash stored inside the JSON).

---

## 11. Files Committed

| File | Committed? |
|---|---|
| `prep/prepare_adult.py` | Yes |
| `docs/cleaning_decisions.md` | Yes |
| `data/adult_clean.csv` | **No** – listed in `.gitignore` |
| `data/adult_split.json` | **No** – listed in `.gitignore` |
