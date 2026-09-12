# CSIT332 ML Course Project — Group 1

## Project
**Are published classifier comparisons reliable?**

CSIT332 · Principles of Machine Learning · semester research project.
Class-wide: 29 students, 8 groups, 4 components, one journal paper.

## Objective
Thousands of papers a year compare a handful of classifiers on a few hundred
samples and declare a winner. The project measures how often that winner is
the wrong one, and builds the tool that tells a researcher how much data they
need to *choose between* models (existing sample-size formulas only cover
*developing* one).

The design manufactures the truth: take a large real dataset and treat it as
the whole population. A **truth arm** with abundant data establishes each
classifier's true AUC. A **study arm** draws small samples (50 – 2,000 rows),
behaves exactly like a real researcher — cross-validates *within* the sample
and declares a winner — and is scored against the truth: winner-correct rate,
Kendall's tau over the ranking, optimism of the reported number, and power to
detect a known gap. Repeated 1,000 times per sample size, per protocol, across
~40 datasets, this yields

```
winner-correct rate ≈ f(sample size, true gap, class prevalence, number of candidates, protocol)
```

which is then used to audit ~400 published papers.

This is **not** "small samples give noisier results": estimating is not
ranking, the models' errors are correlated, and the winner's curse is a bias,
not noise.

## Group
Group 1 — **Component A, The Engine.**

Group 2 works the same component independently from the same written
specification, without talking to us. How far the two groups diverge is a
result that goes in the paper.

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

### Interfaces
The four components connect only through interface files whose columns are
fixed in Week 1 and never change. Component A produces the truth values and
the subsamples that Components B and C consume.

## How the class is organised

| Groups | Component | What they own |
|---|---|---|
| **1 · 2** | **A — The Engine** | **Datasets, the frozen split, the subsample generator, the truth arm** |
| 3 · 4 | B — The Runs | Six classifiers, six protocols, the full grid of simulated studies |
| 5 · 6 | C — The Statistics | Tau, the winner's curse, the power surface, the calculator, every figure |
| 7 · 8 | D — The Audit | 400 papers screened, dual-coded, mapped onto the power surface |

## Timeline

| Weeks | Milestone |
|---|---|
| W1–2 | Specs frozen. Dataset hunt. Audit protocol drafted. |
| W3 | Pilot + go/no-go on scale. |
| W4–7 | Registry built, protocols implemented, screening runs. |
| W8–12 | The grid runs. First real curves. |
| W13–14 | Hierarchical model, the inversion, the mapping. |

## Graded artifacts (cannot be delegated)
- Our own test suite — what did we think could go wrong?
- Our decision log — every ambiguity found in the spec
- A code review of Group 2's implementation
- A ten-minute oral defence of the component

## Dataset

Dataset pool: 20 candidates profiled so far, target ~40. The full register with
source links, target/binarisation rules and quality flags is in
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

Folders for source code, tests, the decision log and results will be added
once that work starts.

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

Current stage: **W1–2 — dataset hunt**

- [x] 20 candidate datasets profiled and recorded in the dataset register
- [ ] Register extended toward ~40 datasets varied in imbalance, dimensionality and separability
- [ ] Interface file columns frozen (Week 1)
- [ ] Pre-registration of truth-arm thresholds
- [ ] Frozen 70 / 30 split per dataset
- [ ] Subsample generator (`dataset_id`, `n_sub`, `replicate_id`)
- [ ] Truth arm run + learning-curve check
- [ ] Pilot / go-no-go on scale (W3)
