# Decision-Focused Learning and Differentiable Optimization

<!-- portfolio-umbrella:start -->
## Portfolio role

This repository is the primary umbrella repository for this Jors Academy research area. Related projects have been consolidated under `projects/` so the methods, implementations, experiments, and case studies can be maintained and explored from one place.

### Included projects

- [`differentiable-black-box-supplier-selection-pytorch`](projects/differentiable-black-box-supplier-selection-pytorch/)
- [`differentiable-optimization-portfolio`](projects/differentiable-optimization-portfolio/)
- [`differentiable-optimization-pytorch`](projects/differentiable-optimization-pytorch/)
- [`distributionally-robust-decision-focused-learning`](projects/distributionally-robust-decision-focused-learning/)
- [`predict-then-optimize-production-planning-spo-plus-pytorch`](projects/predict-then-optimize-production-planning-spo-plus-pytorch/)
- [`optimization-over-trained-ml-models`](projects/optimization-over-trained-ml-models/)

Each consolidated project keeps its own files and a `SOURCE_REPOSITORY.md` provenance record. The snapshot preserves the source repository's default-branch files at consolidation time; repository-level history and metadata remain separate from the snapshot.
<!-- portfolio-umbrella:end -->

A compact research-oriented implementation of **Smart Predict-then-Optimize (SPO)** and the **SPO+** surrogate for downstream combinatorial optimization.

> **License:** source-available for non-commercial use only. See `LICENSE`.

## Motivation

A standard machine-learning pipeline predicts unknown optimization costs by minimizing a prediction loss such as mean squared error (MSE), then solves the downstream optimization problem using those predictions. Prediction accuracy and decision quality are not the same objective: two cost predictions with similar MSE can induce very different decisions.

This repository compares:

- **Predict-then-optimize (MSE):** train a predictor for cost accuracy, then optimize.
- **Decision-focused SPO+:** train the predictor using a surrogate that explicitly depends on the downstream optimization oracle.

The implementation follows the research line introduced by Elmachtoub and Grigas, *Smart "Predict, then Optimize"*.

## Downstream problems

### Fixed-cardinality selection

For each instance, a feature vector `x` determines an unknown cost vector `c` over `n` candidate items. The downstream decision selects exactly `k` items:

\[
\min_{w \in \{0,1\}^n} c^T w
\quad \text{s.t.}\quad \sum_i w_i = k.
\]

The exact oracle selects the `k` smallest costs.

### Contextual shortest path

The repository also includes a directed square grid graph with right/down arcs from the top-left source to the bottom-right sink. Contextual features determine unknown edge costs, and the downstream problem is

\[
\min_{w \in \mathcal{P}} c^T w,
\]

where `P` is the set of valid source-to-sink paths and `w` is an edge-incidence vector.

Because the grid is a DAG, the exact shortest-path oracle is implemented with dynamic programming. The same oracle can also optimize the reflected SPO+ costs `2ĉ-c`, including negative reflected edge costs.

## SPO loss and SPO+ surrogate

Let `w*(c)` denote an optimal downstream decision for true costs `c`, and let `ĉ` be predicted costs. Decision regret is

\[
\ell_{SPO}(\hat c,c)=c^T w^*(\hat c)-c^T w^*(c).
\]

SPO+ uses the optimization oracle at the reflected costs `2ĉ-c` and gives a tractable surrogate with subgradient

\[
2\left(w^*(c)-w^*(2\hat c-c)\right).
\]

The code implements the corresponding surrogate directly with PyTorch autograd while treating discrete oracle outputs as fixed subgradient selections.

## Features

- synthetic contextual cost datasets with controlled model misspecification
- exact fixed-cardinality selection oracle
- exact grid shortest-path oracle
- prediction MSE and downstream regret metrics
- SPO+ surrogate training for both downstream problems
- matched MSE vs SPO+ experiments
- repeated-seed mean/std/95% CI summaries for both benchmarks
- normalized regret relative to the clairvoyant optimum
- explicit MSE-minus-SPO+ normalized-regret improvement summary for shortest path
- unit tests for oracle optimality, regret, SPO+ gradients, statistical summaries, and training smoke tests
- GitHub Actions CI with Python 3.11/3.12, pytest, and Ruff

## Project structure

```text
.
├── src/dfl_spo/
│   ├── data.py
│   ├── oracle.py
│   ├── losses.py
│   ├── models.py
│   ├── training.py
│   ├── evaluation.py
│   └── shortest_path.py
├── scripts/
│   ├── compare_methods.py
│   └── compare_shortest_path.py
├── tests/
├── .github/workflows/ci.yml
├── pyproject.toml
└── LICENSE
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## Run the selection comparison

```bash
python scripts/compare_methods.py --epochs 100 --seeds 0 1 2 3 4
```

## Run the shortest-path comparison

```bash
python scripts/compare_shortest_path.py \
  --grid-size 5 \
  --features 8 \
  --train-samples 512 \
  --test-samples 256 \
  --epochs 100 \
  --seeds 0 1 2 3 4
```

For every seed the script creates matched MSE and SPO+ models, trains both on the same contextual shortest-path dataset, and evaluates:

- prediction MSE,
- achieved path cost,
- clairvoyant optimal path cost,
- mean regret,
- mean normalized regret,
- zero-regret decision rate.

It then aggregates every metric across seeds using:

- arithmetic mean,
- sample standard deviation,
- normal-approximation 95% confidence interval half-width, `1.96 * s / sqrt(n)`.

The final `spo+_improvement` row reports

\[
\mathrm{NR}_{MSE}-\mathrm{NR}_{SPO+},
\]

where `NR` is mean normalized regret. Positive values therefore favor SPO+.

The intended comparison is decision quality, not only cost-prediction accuracy. Confidence intervals quantify seed-to-seed uncertainty; they are not a substitute for a full hypothesis test.

## Interpretation

A useful decision-focused result can look like this qualitatively:

- the MSE model has lower prediction error,
- the SPO+ model has similar or worse prediction error,
- but the SPO+ model has lower downstream regret.

No numerical performance claims are hard-coded into this repository; run the experiments to obtain results for your environment and random seeds.

## Tests

```bash
pytest
ruff check .
ruff format --check .
```

## References

1. Elmachtoub, A. N., & Grigas, P. (2022). Smart "Predict, then Optimize". *Management Science*, 68(1), 9-26. Earlier version: arXiv:1710.08005.
2. Liu, H., & Grigas, P. (2021). Risk Bounds and Calibration for a Smart Predict-then-Optimize Method. arXiv:2108.08887.

## License

This repository is licensed under the **PolyForm Noncommercial License 1.0.0**. Commercial use is not permitted. See `LICENSE` for the full terms.
