# Cleaning Decisions - Component A, Group 1 (Week 2)

Every dataset is prepared by its own script in `prep/`, which writes:

- `data/<name>_clean.csv` - features `f1..fk`, all standardised to mean 0 / SD 1
  (population SD), plus the binary target `y`. **Not committed** (too large for
  GitHub; regenerate by running the script).
- `data/<name>_split.json` - the frozen, stratified 70/30 split stored as row
  indices into the clean CSV, plus the CSV's MD5. **Committed**, so the split is
  fixed even if a library upgrade changes how it would be recomputed.

Each split seed is recorded per dataset in the table below and inside its
`_split.json`.

## Shared pipeline

All scripts follow the same steps, in this order:

1. Download from OpenML.
2. Binarise the target; assert the prevalence matches the register.
3. Drop identifier and duplicate columns.
4. Recode sentinels (`?`, etc.) to missing; drop columns with **>30% missing**.
5. Impute the rest: numeric -> median, categorical -> mode (exceptions noted per dataset). No rows are dropped.
6. One-hot encode categoricals with `drop_first=True`.
7. Cast to float, drop zero-variance columns.
8. Standardise (`ddof=0`) and rename columns `f1..fk`.
9. Sanity checks (no NaN, `y` in {0,1}, all float64, mean 0 / SD 1), then save the CSV.
10. Stratified 70/30 split; assert pool and test prevalence are within 0.5 pp of overall.

## Profile table

| Dataset | OpenML ID | Prepared by | Final rows | Final features | Prevalence | Pool / test | Split seed |
|---|---|---|---|---|---|---|---|
| Adult / Census Income | 1590 | Sheikh Mohsin | 48,842 | 96 | 0.2393 | 34,189 / 14,653 | 332 |
| Diabetes 130-US Hospitals | 4541 | Soummil Goel | 101,766 | 136 | 0.1116 | 71,236 / 30,530 | 42 |
| Don't Get Kicked | 41162 | Soummil Goel | 72,983 | 180 | 0.1230 | 51,088 / 21,895 | 42 |
| Bank Marketing | 1461 | Priyansha Arora | 45,211 | 39 | 0.1170 | 31,647 / 13,564 | 332 |
| Covertype | 1596 | Tejas Bhardwaj | 581,012 | 54 | 0.4876 | 406,708 / 174,304 | 332 |

---

## Adult / Census Income (OpenML 1590), prepared by Sheikh Mohsin

**Script:** `prep/prepare_adult.py`

- **Target:** `income.strip().rstrip('.') == '>50K'` -> 1, else 0. Prevalence 0.2393 (11,687 of 48,842).
  - The raw target uses two spellings of each class (`>50K` and `>50K.`, from the original train and test files). Stripping the trailing period is essential; without it prevalence comes out at 0.1605.
- **Missing values:**
  - Sentinel `?` recoded to missing. No column exceeded 30%, so none dropped.
  - Filled 6,465 cells in 3 columns with the mode: `workclass` 2,799 (5.73%, `Private`), `occupation` 2,809 (5.75%, `Prof-specialty`), `native-country` 857 (1.75%, `United-States`). Rows dropped: 0.
- **Categoricals:** 8 columns one-hot encoded -> 91 dummy columns (`workclass` 7, `education` 15, `marital-status` 6, `occupation` 13, `relationship` 5, `race` 4, `sex` 1, `native-country` 40). All cardinalities < 50.
- **Dropped columns:** `education-num`: integer encoding of `education`, same information (see `docs/dataset_register.md`).
- **Zero-variance columns:** none.
- **Final:** 48,842 rows, 96 features (5 numeric - `age`, `fnlwgt`, `capital-gain`, `capital-loss`, `hours-per-week` - plus 91 dummies), prevalence 0.2393, seed 332. Pool 34,189 / test 14,653, both at prevalence 0.2393.
- **Licence:** CC BY 4.0 (UCI 2).

---

## Diabetes 130-US Hospitals (OpenML 4541), prepared by Soummil Goel

**Script:** `prep/prepare_diabetes.py`

- **Target:** `readmitted == '<30'` (readmitted within 30 days) -> 1; `>30` and `NO` -> 0. Raw counts NO 54,864 / >30 35,545 / <30 11,357; prevalence 0.1116 (11,357 of 101,766).
- **Missing values:**
  - Sentinels recoded to missing: `?` (all columns), `Unknown/Invalid` in `gender` (3 rows), and `None` in `max_glu_serum` / `A1Cresult`, where it means "test not performed" (the register counts that as missing).
  - Columns dropped (>30% missing): `weight` 96.86%, `max_glu_serum` 94.75%, `A1Cresult` 83.28%, `medical_specialty` 49.08%, `payer_code` 39.56%.
  - Filled 4,078 cells in 5 columns with the mode: `race` 2,273 (Caucasian), `gender` 3 (Female), `diag_1` 21, `diag_2` 358, `diag_3` 1,423 (Circulatory). Rows dropped: 0.
- **Recoding before encoding:**
  - `age` brackets (`[70-80)`) -> bracket midpoint (75), kept as one numeric column since the brackets are ordered.
  - `diag_1/2/3` had 717 / 749 / 790 raw ICD-9 codes. Each was mapped to the 9 diagnosis groups used by Strack et al. (2014), the paper that released the dataset: Circulatory, Respiratory, Digestive, Diabetes, Injury, Musculoskeletal, Genitourinary, Neoplasms, Other.
  - `admission_type_id`, `discharge_disposition_id`, `admission_source_id` are lookup codes, not quantities, so they are treated as categorical.
- **Categoricals:** 33 columns one-hot encoded -> 127 dummy columns.
- **Dropped columns:** `encounter_id`, `patient_nbr` (identifiers); the five >30%-missing columns above.
- **Zero-variance columns:** `examide` and `citoglipton` have a single value (`No`), so they produce no dummy columns.
- **Final:** 101,766 rows, 136 features (9 numeric + 127 dummies), prevalence 0.1116, seed 42. Pool 71,236 / test 30,530, both at prevalence 0.1116.
- **Caveats:**
  - Rows are encounters, not patients: 71,518 unique patients across 101,766 rows. The same patient can appear in both the pool and the test set. We kept every row to stay consistent with the register's shipped row count.
  - `discharge_disposition_id` includes "expired" codes. Those patients cannot be readmitted, so they are always `y = 0`. This is known at discharge, so it is not leakage, but it makes some rows trivially predictable.

---

## Don't Get Kicked (OpenML 41162), prepared by Soummil Goel

**Script:** `prep/prepare_kick.py`

- **Target:** `IsBadBuy` - already binary (stored as a category); `'1'` -> 1. Prevalence 0.1230 (8,976 of 72,983).
- **Missing values:**
  - No sentinel values in this file; missing values are real NaN.
  - Columns dropped (>30% missing): `PRIMEUNIT` 95.32%, `AUCGUART` 95.32%.
  - `WheelType` (3,174) and `Trim` (2,360): missing filled with its own category `Missing`, not the mode. A missing wheel type is a known strong signal of a bad buy, so filling it with the mode would erase real information.
  - Numeric columns filled with the median: four `MMRAcquisition*` prices (18 each), four `MMRCurrent*` prices (315 each), `VehBCost` (68).
  - Categorical columns filled with the mode: `SubModel` 8, `Color` 8, `Transmission` 9, `Nationality` 5, `Size` 5, `TopThreeAmericanName` 5.
  - Total 6,974 cells filled; rows dropped: 0.
- **Recoding before encoding:** `Transmission` values `Manual` and `MANUAL` merged (upper-cased) -> 2 levels.
- **High-cardinality columns** (>50 levels): keep the 20 most frequent levels, group the rest as `Other`:
  - `Model` 1,063 levels (top 20 cover 31.4% of rows)
  - `SubModel` 863 levels (54.1%)
  - `Trim` 135 levels (84.4%)
- **Categoricals:** 12 columns one-hot encoded -> 166 dummy columns.
- **Dropped columns:**
  - `BYRNO`: buyer identifier
  - `VNZIP1`: vendor ZIP code, identifier-like, location already covered by `VNST`
  - `WheelTypeID`: numeric code duplicating `WheelType`
  - `VehYear`: duplicates `VehicleAge` (purchase year - `VehYear`)
  - `PRIMEUNIT`, `AUCGUART`: >30% missing
- **Zero-variance columns:** none.
- **Final:** 72,983 rows, 180 features (14 numeric + 166 dummies), prevalence 0.1230, seed 42. Pool 51,088 / test 21,895, both at prevalence 0.1230.
- **Caveats:**
  - The register lists 30 features; OpenML ships 32 predictors (the extra two are `WheelTypeID` and `VehYear`).
  - Some `MMR*` price columns contain 0, which is likely a "price unknown" sentinel in the original Kaggle data. These are left as-is (not recoded) and flagged here.

---

## Bank Marketing (OpenML 1461), prepared by Priyansha Arora

**Script:** `prep/prepare_bank.py`

- **Target:** Converted term deposit subscription (`Class == '2'`) -> 1, else 0. Prevalence 0.1170 (5,289 of 45,211).
- **Dropped columns:** `V12` (`duration`): call length, only known after the call ends, so it leaks the outcome (UCI's documentation says to drop it for realistic models).
- **Missing values:** Replaced `unknown` and `?` sentinels with NaN, except `V16` (`poutcome`), where `unknown` means "not in a previous campaign" and is kept as a real level. `V9` (`contact`, 28.8% unknown) gets its own `Missing` category instead of the mode, since the unknowns cluster in the early calls. Imputed numerical columns with median and categorical with mode. No columns exceeded 30% missing, so none were dropped.
- **Categoricals:** One-hot encoded categorical features using `pd.get_dummies(drop_first=True)`.
- **Final:** 45,211 rows, 39 standardized features (f1..fk), binary target y in {0, 1}, stratified 70/30 split with seed 332. Pool 31,647 / test 13,564, both at prevalence 0.1170.

---

## Covertype (OpenML 1596), prepared by Tejas Bhardwaj

**Script:** `prep/prepare_covertype.py`

- **Target:** `class == 2` -> 1; classes 1, 3, 4, 5, 6 and 7 -> 0. Positive count: 283,301 of 581,012 rows; prevalence 0.4876.
- **Missing values:** No missing feature cells found. No rows or columns were dropped for missingness.
- **Feature types:** The 54 source features comprise 10 numerical terrain/cartographic measurements and 44 binary indicators: 4 wilderness-area indicators and 40 soil-type indicators. OpenML exposes the indicator columns as categorical types; the script converts their existing numeric values to numeric columns.
- **Categoricals:** No additional one-hot encoding was needed because the wilderness-area and soil-type indicators are already separate columns.
- **Dropped columns:** None reported in the successful run; all 54 source features were retained.
- **Zero-variance columns:** None removed; final feature count remains 54.
- **Standardisation:** Features converted to float64, standardised using population SD (`ddof=0`), and renamed `f1` through `f54`. Target is stored as binary `y`.
- **Final:** 581,012 rows, 54 features, prevalence 0.4876, split seed 332. Pool 406,708 / test 174,304; both reported at prevalence 0.4876.
- **Split integrity:** Stratified 70/30 split saved in `data/covertype_split.json`; clean CSV is regenerated by running the script and is not committed.
- **Licence:** CC BY 4.0.
