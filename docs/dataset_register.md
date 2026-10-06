# Dataset Register

This document records candidate datasets considered for the project.

The datasets are evaluated based on:
- Size
- Feature count
- Target definition
- Class balance
- Missingness
- Licensing
- Data quality
- Suitability for the project

## Conventions

- **Rows** is the shipped row count, before any deduplication (ambiguity A5).
- Datasets need at least **10,000 rows** to be admitted.
- Columns with more than **30% missing** are dropped during preparation.
- **Class prevalence** is the positive-class fraction after applying the binarisation rule.
- **Missingness** is measured after sentinel values (`-1`, `?`, `unknown`, etc.) are recoded to missing where one exists.
- Datasets are **not** committed to this repository — use the source links.

## Register

| Name | Source and URL | Rows | Features | Target and binarisation rule | Class prevalence | Missingness | Licence |
|---|---|---|---|---|---|---|---|
| Higgs | [OpenML 42769](https://www.openml.org/d/42769) | 1,000,000 | 28 | `target` - already binary. Positive = 0 (background) vs 1 (signal). | 0.4701 | None - 0 of 28 cols | Public / CC BY 4.0 - redistribution permitted |
| Porto Seguro Safe Driver | [OpenML 42742](https://www.openml.org/d/42742) | 595,212 | 55 | `target` - already binary. Positive = 1 (driver filed a claim in the next year) vs 0. | 0.0364 | 0.52% in 11 of 55 cols | Public on OpenML - FLAG: original is Kaggle competition data; cite the OpenML-hosted copy |
| Covertype | [OpenML 1596](https://www.openml.org/d/1596) / [UCI 31](https://archive.ics.uci.edu/dataset/31) | 581,012 | 54 | `class` - 7 classes. RULE: class 2 (Lodgepole Pine) versus all others. Counts 1:211,840 2:283,301 3:35,754 4:2,747 5:9,493 6:17,367 7:20,510. | 0.4876 | None - 0 of 54 cols | CC BY 4.0 - redistribution permitted |
| MiniBooNE | [OpenML 41150](https://www.openml.org/d/41150) / [UCI 199](https://archive.ics.uci.edu/dataset/199) | 130,064 | 50 | `signal` - already binary. Positive = True (electron neutrino) vs False (muon neutrino background). | 0.2806 | None - 0 of 50 cols | CC0 / CC BY 4.0 - redistribution permitted |
| Diabetes 130-US Hospitals | [OpenML 4541](https://www.openml.org/d/4541) / [UCI 296](https://archive.ics.uci.edu/dataset/296) | 101,766 | 40 | `readmitted` - 3 classes. RULE: `<30` (readmitted within 30 days) versus `>30` and `NO` combined. Counts NO:54,864 >30:35,545 <30:11,357. | 0.1116 | 0.10% in 4 of 40 cols | CC BY 4.0 - redistribution permitted |
| Uttar Pradesh Schools | [Kaggle (UDISE+)](https://www.kaggle.com/datasets/soummilgoel/uttar-pradesh-schools-dataset) | 89,873 | 155 | `library_availability` - no native target. RULE: == 2 (no library) versus == 1. UDISE+ codes 1=Yes, 2=No. Alternatives verified: internet 40.01%, playground_available 25.66%, approachable_road 7.37%. | 0.1386 | None - 0 of 155 cols | CC BY 4.0 - redistribution permitted |
| APS Failure (Scania Trucks) | [OpenML 41138](https://www.openml.org/d/41138) / [UCI 421](https://archive.ics.uci.edu/dataset/421) | 76,000 | 159 | `class` - already binary. Positive = pos (failure in the air pressure system) vs neg (failure elsewhere). | 0.0181 | 4.53% in 158 of 159 cols | CC0 / CC BY 4.0 - redistribution permitted |
| Don't Get Kicked | [OpenML 41162](https://www.openml.org/d/41162) | 72,983 | 30 | `IsBadBuy` - already binary. Positive = 1 (auction car was a bad buy) vs 0. | 0.123 | 0.46% in 18 of 30 cols | CC0 on OpenML - FLAG: original is Kaggle competition data; cite the OpenML-hosted copy |
| Cardiovascular Disease | [OpenML 45547](https://www.openml.org/d/45547) | 70,000 | 11 | `cardio` - already binary. Positive = 1 (cardiovascular disease present) vs 0. | 0.4997 | None - 0 of 11 cols | FLAG: LICENCE UNSTATED on OpenML and upstream. Provenance is documented but redistribution is NOT established. |
| jannis | [OpenML 45021](https://www.openml.org/d/45021) | 57,580 | 54 | `class` - already binary in this version. Positive = 1 vs 0. Derived from the 4-class ChaLearn original by the benchmark authors. | 0.5 | None - 0 of 54 cols | Public - redistribution permitted |
| Adult / Census Income | [UCI 2](https://archive.ics.uci.edu/dataset/2) / [OpenML 1590](https://www.openml.org/d/1590) | 48,842 | 14 | `income` - already binary, BUT strip the trailing period first: `income.strip().rstrip('.') == '>50K'`. The test half writes `>50K.`; skipping the strip gives 16.05% instead of 23.93%. | 0.2393 | 0.94% in 3 of 14 cols | CC BY 4.0 - redistribution permitted |
| Bank Marketing (bank-full) | [UCI 222](https://archive.ics.uci.edu/dataset/222) / [OpenML 1461](https://www.openml.org/d/1461) | 45,211 | 15 | `y` - already binary. Positive = yes (subscribed a term deposit) vs no. On OpenML 1461 the identical rule reads `Class == '2'` vs `'1'`. | 0.117 | 2.24% in 3 of 15 cols | CC BY 4.0 - redistribution permitted |
| Loan Approval Status | [OpenML 46526](https://www.openml.org/d/46526) | 45,000 | 23 | `loan_status` - already binary. Positive = loan_approved vs loan_rejected. | 0.2222 | None - 0 of 23 cols | CC BY 4.0 - redistribution permitted |
| Bank Marketing (bank-additional-full) | [UCI 222](https://archive.ics.uci.edu/dataset/222) | 41,188 | 20 | `y` - already binary. Positive = yes (subscribed a term deposit) vs no. Drop `duration` for a realistic benchmark (post-hoc leakage), leaving 19 features. | 0.1127 | 1.54% in 6 of 20 cols | CC BY 4.0 - redistribution permitted |
| Click Prediction (small) | [OpenML 42733](https://www.openml.org/d/42733) | 39,948 | 11 | `click` - already binary. Positive = 1 (ad was clicked) vs 0. | 0.1684 | None - 0 of 11 cols | Public - redistribution permitted |
| Nomao | [OpenML 1486](https://www.openml.org/d/1486) / [UCI 227](https://archive.ics.uci.edu/dataset/227) | 34,465 | 118 | `Class` - already binary. Positive = 1 (the two place records do not match) vs 2 (they match). | 0.2856 | None - 0 of 118 cols | Public / CC BY 4.0 - redistribution permitted |
| Amazon Employee Access | [OpenML 4135](https://www.openml.org/d/4135) | 32,769 | 9 | `target` - already binary. Positive = 0 (access request denied) vs 1 (approved). | 0.0579 | None - 0 of 9 cols | Public on OpenML - FLAG: original is Kaggle competition data; cite the OpenML-hosted copy |
| Default of Credit Card Clients | [UCI 350](https://archive.ics.uci.edu/dataset/350) | 30,000 | 23 | `Y` - already binary. Positive = 1 (default on next month's payment) vs 0. | 0.2212 | None - 0 of 23 cols | CC BY 4.0 - redistribution permitted |
| Toronto COVID-19 Cases | [OpenML 43480](https://www.openml.org/d/43480) | 14,911 | 7 | `Ever_Hospitalized` - no native target. RULE: == 'Yes' versus 'No'. Six leaking outcome columns (Outcome, Currently_Hospitalized, Currently_in_ICU, Currently_Intubated, Ever_in_ICU, Ever_Intubated) and both date columns dropped. Alternative: Outcome == 'FATAL' vs rest = 7.52%. | 0.1239 | 1.16% in 3 of 7 cols | CC0 / OGL Toronto - redistribution permitted |
| Phishing Websites | [UCI 327](https://archive.ics.uci.edu/dataset/327) | 11,055 | 30 | `result` - already binary. Positive = 1 (phishing) vs -1 (legitimate). | 0.4431 | None - 0 of 30 cols | CC BY 4.0 - redistribution permitted |

## Flags at a glance

- **Admitted with reservation**
  - Cardiovascular Disease — licence unstated upstream; do not republish the file.
  - Phishing Websites — 47% duplicate rows; fails the 10,000-row floor if counted on distinct rows.
- **Kaggle competition origin — cite the OpenML-hosted copy**
  - Porto Seguro Safe Driver, Don't Get Kicked, Amazon Employee Access.
- **Overlapping pair — pick one if independent datasets are required**
  - Bank Marketing (bank-full) and Bank Marketing (bank-additional-full).
