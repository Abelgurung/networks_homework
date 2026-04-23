# Assignment 4 - Arbitrary Traffic Matrix (Gurobi)

Joint topology-design + multi-commodity-flow MILP for the hose-model
instance with `n = 8` nodes and `d = 4` in/out links per node.

## Files

| File | Purpose |
|------|---------|
| `topology_design.py` | Gurobi MILP model and helpers |
| `run_experiments.py` | Driver that solves the MILP on several sample hose matrices |
| `sample_output.txt`  | Captured output from `run_experiments.py` |

## Requirements

```bash
pip install gurobipy numpy
```

A free academic or the default restricted Gurobi licence is sufficient for
`n = 8`, `d = 4`.

## Usage

```bash
python run_experiments.py
```

The script prints, for each sample traffic matrix:

* the matrix itself and its row/column sums,
* the optimal concurrent-flow throughput lambda*,
* the integer link-multiplicity matrix `c[u, v]` chosen by the MILP.

## Model summary

Decision variables

* `c[u, v]` - integer in `[0, d]`, number of unit-capacity directed links u -> v.
* `g[t, u, v]` - non-negative flow destined for `t` on edge `(u, v)`
  (destination-aggregated MCF; equivalent to full per-pair flow for
  fractional routing).
* `lam` - concurrent-flow throughput to maximise.

Constraints

1. Out-degree:     `sum_v c[u, v] <= d`
2. In-degree:      `sum_u c[u, v] <= d`
3. No self-loops:  `c[u, u] = 0`
4. Flow conservation at every `(t, u)`.
5. Capacity:       `sum_t g[t, u, v] <= c[u, v]`.

Objective: `maximise lam`.
