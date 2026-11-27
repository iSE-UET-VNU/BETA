# BETA

**Learning Dataset Similarity from Preprocessing Behavior for Preprocessing Pipeline Optimization**

Choosing a data-preprocessing pipeline for a new tabular classification dataset is usually done by
trial and error, or by transferring pipelines from datasets that *look* similar in their
meta-features. BETA instead learns which historical datasets *respond* similarly to preprocessing 
(their behavioral similarity) and transfers preprocessing knowledge across datasets. A Siamese
encoder is trained so that datasets with similar preprocessing-response profiles land close in
embedding space. For a new dataset, BETA retrieves its behavioral neighbors, pools their best-known
pipelines into a per-operator search prior, and refines that prior with Ant Colony Optimization
before returning a single recommended pipeline. This repository implements the method and its
equations as described in the paper.


## Architecture & Core Components

### BETA Architecture

- **Behavioral Similarity Learning.** A Siamese network embeds OpenML-style dataset meta-features so
  that proximity in embedding space reflects *preprocessing behavior* rather than raw meta-feature
  similarity. The training target is each dataset's relative behavioral profile over the reference
  pipeline set.
- **Heuristic Transferring.** For a target dataset, BETA retrieves the top-`K` behaviorally similar
  reference datasets, takes each neighbor's top-`H` historical pipelines, and aggregates them into a
  per-operator heuristic `eta` that biases the search
  toward operators that worked on similar datasets.
- **Pipeline Refining (ACO).** Ant Colony Optimization samples candidate pipelines step by step from
  `eta` and a pheromone trail `tau` (weighted by `alpha`, `beta`), scores them with a fast proxy or
  with AutoGluon, and reinforces the elite pipelines each iteration. The best searched pipeline is
  compared against the best directly-transferred pipeline, and the higher-scoring one is returned.

The pipeline prototype is a fixed six-step sequence:

```
imputation -> scaling -> encoding -> outlier_removal -> feature_selection -> dimensionality_reduction
```

with `{6, 5, 2, 6, 4, 3}` operators per step — 4,320 candidate pipelines.

Supporting modules: a logistic-regression proxy evaluator and an optional AutoGluon downstream
evaluator; a reference-library loader that holds out the paper's 30 evaluation datasets; a
meta-feature module that looks up or computes `m(D)` for the similarity encoder.

## Installation

### Setup

```bash
# Clone the repository
git clone https://github.com/iSE-UET-VNU/BETA.git
cd BETA

# Prepare environment  (Python >= 3.10, tested on 3.11)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .                 # core: proxy evaluator + similarity training
pip install -e ".[autogluon]"    # + AutoGluon downstream evaluator (the paper's protocol)
pip install -e ".[openml]"       # + OpenML client
```

`numpy < 2.0` is pinned — AutoGluon and the pinned `torch` build both require it.

`assets/` ships a pretrained similarity encoder, a 12x901 reference performance matrix, its
meta-features, and the reference pipeline definitions, so `beta recommend` runs out of the box.

## Configuration

Runtime behavior is defined in `configs/default.yaml`. Pass it with `--config`, and override
individual keys inline with `--set key=value`.

Example `configs/default.yaml`:

```yaml
similarity:
  hidden_dim: 64
  embed_dim: 64
  epochs: 100
  lr: 0.001
  similarity_target: row_zscore_cosine
  metric_objective: embedding_cosine     # embedding_cosine (Eq. 7) | projector_product (learned projector)

transfer:
  top_k: 1          # K - behaviorally similar datasets retrieved
  top_h: 3          # H - transferred pipelines per neighbor

aco:
  n_ants: 20
  n_iterations: 10
  alpha: 1.0        # pheromone weight
  beta: 2.0         # heuristic (eta) weight
  evaporation: 0.2
  elite_size: 3
  update_strategy: linear   # Eq. 13 (default) | exponential | rank | uniform

split:
  val_ratio: 0.2
  test_ratio: 0.2
  stratify: true    # false reproduces the unstratified splits of earlier versions

proxy_model: logreg
downstream_evaluator: proxy   # proxy | autogluon
seed: 42
```

`transfer.top_k: 1` retrieves a single neighbor; if that neighbor's top pipelines are tied in score
the transferred prior is uninformative. Raise it (`--set transfer.top_k=5`) for a more
discriminative prior.

## Usage

### Quickstart (pretrained)

```python
from beta import BETA

model = BETA.from_pretrained()
result = model.recommend(openml_id=1066)          # or: model.recommend(X=X, y=y)

result.pipeline     # {'imputation': 'median', 'scaling': 'robust', ...}
result.eta          # per-step transferred heuristic
result.neighbors    # retrieved datasets and their similarities
result.source       # 'search' or 'transfer' - which pipeline won
result.score
```

```bash
beta recommend --openml 1066
beta recommend --csv mydata.csv --target label
beta recommend --openml 1066 --downstream autogluon --set aco.n_ants=25 --set transfer.top_k=5
```

### Rebuilding the knowledge base

To retrain BETA on your own dataset collection, run the three steps in order.

**1. Build the reference performance matrix** — runs AutoGluon for every (dataset, reference
pipeline) pair (slow).

```bash
python3 -m beta.cli build-matrix --datasets ids.txt --out assets/performance_matrix.csv
```

**2. Train the similarity encoder** — fits the Siamese behavioral-similarity model on the matrix.

```bash
python3 -m beta.cli train-similarity --matrix assets/performance_matrix.csv --out assets/similarity_encoder.pt
```

**3. Recommend** — retrieval, heuristic transfer, and ACO refinement all run inside `recommend`.

```bash
python3 -m beta.cli recommend --openml <id> --config configs/default.yaml
```

(`python3 -m beta.cli <cmd>` and the installed `beta <cmd>` entry point are equivalent.)

## Evaluation-dataset holdout

The paper's 30 evaluation datasets are removed from the reference library at load time, with an
assertion that the cleaned library is disjoint from that id set. `experiments/verify_holdout.py`
retrains the similarity encoder from the held-out library and checks that it reproduces
`assets/similarity_encoder.pt`.

## Reproducing the experiments

All experiments run in one CPU-only environment (Python 3.11, AutoGluon 1.5.0; exact versions in
`requirements-lock.txt`):

```bash
docker build -t beta .
alias run='docker run --rm -v "$PWD":/beta beta'

run python experiments/prepare_data.py       # 30 datasets of Table 2 -> data/eval/ (hashes in data/manifest.json)
run python experiments/verify_holdout.py     # encoder was trained with every evaluation dataset held out
run python experiments/run_rq1.py --datasets 1066 876 --data-dir data/eval --out results/rq1.jsonl
run python experiments/run_rq1.py --summarize results/rq1.jsonl
run python -m pytest tests -q
```

`prepare_data.py` rewrites `data/manifest.json`; `git diff data/manifest.json` shows any file that
differs from the one used in the paper. `run_rq1.py` writes one line per (dataset, seed, method) and
resumes where it stopped.

Every table runs through `experiments/run_rq1.py` (one line per dataset, seed and method; `--predictions`
also stores the test predictions, so accuracy and macro-F1 are both available). Each RQ2/RQ3 variant
changes one setting with `--set`; everything else stays at the defaults of `configs/default.yaml`:

```bash
IDS="1066 1047 862 40663 1054 876 18 1520 1548 378 1485 14 27 44956 1037 42932 40668 1471 100000 42165 41027 41671 1046 46597 30 802 722 40922 1119 1497"
RUN="run python experiments/run_rq1.py --data-dir data/eval --seeds 0 1 2 3 4 --predictions"

$RUN --datasets $IDS --out results/rq1.jsonl                                   # Table 3: BETA and AutoGluon default
$RUN --datasets <ids> --arms beta --set ablation.similarity=metafeature_cosine --out results/rq2/t5.jsonl   # Table 5
$RUN --datasets <ids> --arms beta --set ablation.heuristic=uniform --out results/rq2/t6.jsonl             # Table 6
$RUN --datasets <ids> --arms beta --set ablation.optimizer=random --out results/rq2/t7_random.jsonl       # Table 7: random | greedy | sa | ga | mcts | tpe
$RUN --datasets <ids> --arms beta --set ablation.operator_space=no_cross_feature --out results/rq2/t8.jsonl  # Table 8
$RUN --datasets <ids> --arms beta --set ablation.mode=search_only --out results/rq2/t9_search.jsonl       # Table 9: search_only | transfer_only
$RUN --datasets <ids> --arms beta --metric f1_macro --set transfer.top_k=3 --out results/rq3/k3.jsonl     # Table 10: transfer.top_k = 1 | 3 | 5
#   Table 11: transfer.top_h = 1..5   Table 12: aco.update_strategy = linear | exponential | rank | uniform   Table 13: aco.n_ants = 5..25
```

The other optimizers of Table 7 use the same proxy and the same budget (`aco.n_ants` x `aco.n_iterations`
evaluations); `--set aco.time_budget=<seconds>` caps BETA's search time. `experiments/save_splits.py`
writes (or checks) the train/validation/test splits of each dataset and seed, so other methods can be run
on exactly the same rows.

## Status

A from-scratch reimplementation of BETA's method as described in the paper.

## Citation

See `CITATION.cff`.
