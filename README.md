# CSIT332 ML Course Project — Group 1

## Project
**Are published classifier comparisons reliable?**

CSIT332 · Principles of Machine Learning · semester research project, split
across the whole class in four components.

## Objective
Papers routinely compare a handful of classifiers on a few hundred samples and
declare a winner. The project measures how often that winner is wrong, and
builds a tool that tells a researcher how much data they need to choose
between models.

The idea: take a large real dataset and treat it as the whole population. With
abundant data, work out each classifier's *true* performance. Then simulate
thousands of small studies drawn from the same data, let each one pick a
winner the way a real researcher would, and count how often it picks the wrong
one.

## Group
Group 1 — **Component A, The Engine**

## Members

| Member | Roll No. |
|---|---|
| Soummil Goel | 240889 |
| Sheikh Mohsin | 240341 |
| Tejas Bhardwaj | 240324 |
| Priyansha Arora | 240435 |

All four members work on Component A together.

## Assigned component: A — The Engine

What we own: **datasets, the frozen split, the subsample generator, the truth arm.**

### 1. Datasets
About 40 large real datasets, chosen to vary deliberately in class imbalance,
dimensionality and separability. Each gets a written binarisation rule. The
register so far (20 profiled) is in
[`docs/dataset_register.md`](docs/dataset_register.md).

### 2. The frozen split
Each dataset is divided 70 / 30 once, permanently. The 30% is the truth test
set and is never touched again by anything. The 70% is the study pool.

### 3. The subsample generator
Every simulated study is defined by three keys — and the same three always
return exactly the same rows, on any machine, forever:

| Key | Values |
|---|---|
| `dataset_id` | which population |
| `n_sub` | 50, 100, 200, 500, 1000, 2000 |
| `replicate_id` | 1 to 1000 |

Draws come from the study pool only; the study arm never sees the frozen 30%.

### 4. The truth arm
1. **Split once, permanently.** (Step 2 above.)
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

## Dataset

Dataset pool: 20 candidates profiled so far, target ~40. The full register with
source links, target/binarisation rules and quality flags is in
[`docs/dataset_register.md`](docs/dataset_register.md).

No dataset files are stored in this repository. Datasets are pulled from the
OpenML / UCI / Kaggle links in the register; anything downloaded locally goes
under `data/`, which is git-ignored.

## Reflection — dataset selection

The “Binary target, or a target with a stated binarisation rule” criterion was the hardest to apply because, for many datasets, there were multiple possible target columns and several ways in which a target could be binarised. We had to examine the dataset as a whole to determine which column would be the most suitable target and then decide on an appropriate binarisation rule, and we are still not completely certain that we have chosen the best option for every dataset. To make the selection process more efficient, we first applied the basic filters, such as minimum number of rows and data format, so that we did not spend time evaluating datasets that clearly failed the requirements. Each group member then collected around 10 potential datasets, giving us approximately 40 candidates in total, from which we selected the 20 that we found most suitable and interesting for further consideration. Thus, around 20 datasets were rejected during the selection process, primarily because we chose to focus on the candidates that were most relevant and interesting to our group.

## Repository Structure

- `docs/` — project documentation
  - `dataset_register.md` — candidate dataset table with sources and caveats
  - `ML Project.xlsx` — spreadsheet the register was built from
- `data/` — local copies of datasets (git-ignored, nothing is committed)
- `dataset_profile.py` — reusable profiler for triaging tabular datasets
  (`quick_look`, `describe_dataset`, `compare_datasets`)
- `trying_out_openml.py` — scratch script for pulling a dataset from OpenML
  and printing a column summary

Folders for source code, tests and results will be added once that work starts.

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

- [x] 20 candidate datasets profiled and recorded in the dataset register
- [ ] Register extended toward ~40 datasets varied in imbalance, dimensionality and separability
- [ ] Pre-registration of truth-arm thresholds
- [ ] Frozen 70 / 30 split per dataset
- [ ] Subsample generator (`dataset_id`, `n_sub`, `replicate_id`)
- [ ] Truth arm run + learning-curve check
