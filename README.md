# CSIT332 ML Course Project

## Group
Group: 1

## Members
- Soummil Goel — 240889
- <Name> — <Roll No.>
- <Name> — <Roll No.>

## Project
<Short project title>

## Objective
<What the project is trying to investigate/build>

## Assigned Tasks

| Member | Roll No. 
|---|---|
| Soummil Goel | 240889 |
| <Name> | <Roll No.> |
| <Name> | <Roll No.> |

### Component A — the truth arm (Soummil Goel)

Establishes the reference ("truth") AUC of each classifier on each dataset.

1. **Split once, permanently.** Divide the population 70 / 30. The 30% is
   frozen as the truth test set and is never touched again by anything. The
   70% becomes the study pool.
2. **Train ten times, not once.** Fit each classifier on ten different 90%
   subsets of the study pool. A single fit would give a truth value that is
   itself noisy.
3. **Average to get truth.** Every fit is evaluated on the same frozen 30%.
   The mean AUC across the ten fits is that classifier's true value.
4. **Check that truth deserves the name.** The spread across the ten fits must
   be under ±0.005. Also fit at 40 / 60 / 80 / 100% of the pool — if the
   learning curve is still climbing, the dataset is dropped.

The thresholds in step 4 are written into the pre-registration **before any
results exist**, so no dataset can ever be dropped for being inconvenient.

Deliverables:
- Pre-registered thresholds (spread < ±0.005; learning-curve rule), committed before any results
- Frozen 70 / 30 split per dataset, saved so it cannot drift
- Ten fits per classifier per dataset on 90% subsets of the study pool
- Truth table: mean AUC and spread across the ten fits, per classifier per dataset
- Learning-curve check at 40 / 60 / 80 / 100% of the pool, with the list of any datasets dropped

Work completed so far:
- Profiled 20 candidate datasets (size, features, target rule, class balance,
  missingness, duplicates, licence, caveats) — see
  [`docs/dataset_register.md`](docs/dataset_register.md)
- Wrote a reusable dataset profiler, `dataset_profile.py`

## Dataset

Dataset pool: 20 candidates under evaluation — the full register with source
links, target/binarisation rules and quality flags is in
[`docs/dataset_register.md`](docs/dataset_register.md).

No dataset files are stored in this repository. Datasets are pulled from the
OpenML / UCI / Kaggle links in the register; anything downloaded locally goes
under `data/`, which is git-ignored.

## Repository Structure

- `docs/` — project documentation
  - `dataset_register.md` — candidate dataset table with sources and caveats
  - `ML Project.xlsx` — spreadsheet the register was built from
- `data/` — local copies of datasets (git-ignored, nothing is committed)
- `dataset_profile.py` — reusable profiler for triaging tabular datasets
  (`quick_look`, `describe_dataset`, `compare_datasets`)
- `trying_out_openml.py` — scratch script for pulling a dataset from OpenML
  and printing a column summary

Folders for source code, notebooks, experiments, results and reports will be
added once that work starts.

## Setup

```bash
git clone https://github.com/LIMOUS007/CSIT332_project_group1.git
cd CSIT332_project_group1
conda create -n ml-project python=3.13 numpy pandas openml -c conda-forge
conda activate ml-project
```

OpenML downloads are cached locally by the `openml` package.

## Reproducibility

Python version: 3.13

Environment: conda (per course instructions). An `environment.yml` with pinned
versions will be added once the project dependencies are settled.

## Status

Current stage: **Dataset selection**

- [x] Candidate datasets profiled and recorded in the dataset register
- [ ] Final dataset pool chosen
- [ ] Pre-registration written (Component A thresholds)
- [ ] Preprocessing
- [ ] Experiments
- [ ] Report
