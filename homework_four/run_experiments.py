"""Run the topology-design MILP on several hose-model traffic matrices.

Execute with  ``python run_experiments.py``.
Requires gurobipy with a valid (academic) licence.
"""

from __future__ import annotations

import numpy as np

from topology_design import random_hose_matrix, solve_topology


def pretty(mat: np.ndarray) -> str:
    return "\n".join("  " + " ".join(f"{x:7.3f}" for x in row) for row in mat)


def run(name: str, T: np.ndarray, n: int = 8, d: int = 4) -> None:
    print("=" * 72)
    print(f"Instance: {name}")
    print("Traffic matrix T:")
    print(pretty(T))
    row_sums = T.sum(axis=1)
    col_sums = T.sum(axis=0)
    print(f"row sums = {row_sums.round(3)}")
    print(f"col sums = {col_sums.round(3)}")

    result = solve_topology(T, d=d)
    print(f"Optimal lambda = {result['lambda']:.6f}")
    print("Chosen capacity matrix c[u,v]:")
    print(pretty(result["capacity"].astype(float)))
    print(f"(row sums = {result['capacity'].sum(axis=1)},"
          f"  col sums = {result['capacity'].sum(axis=0)})")


def main() -> None:
    rng = np.random.default_rng(42)
    n, d = 6, 2

    # 1. Uniform traffic matrix - sanity check: T_ij = d/(n-1).
    T_uniform = np.full((n, n), d / (n - 1))
    np.fill_diagonal(T_uniform, 0.0)
    run("uniform (d/(n-1))", T_uniform, n, d)

    # 2. Permutation-style matrix: each node talks to a single destination
    #    with demand d.
    perm = rng.permutation(n)
    # avoid fixed points
    while np.any(perm == np.arange(n)):
        perm = rng.permutation(n)
    T_perm = np.zeros((n, n))
    for i in range(n):
        T_perm[i, perm[i]] = d
    run("permutation (demand d on one edge per source)", T_perm, n, d)

    # 3. Skewed matrix: half the sources send all traffic to one destination
    T_skew = np.zeros((n, n))
    for i in range(n):
        dests = rng.choice([j for j in range(n) if j != i], size=2, replace=False)
        T_skew[i, dests[0]] = 3.0
        T_skew[i, dests[1]] = 1.0
    # project onto hose polytope via Sinkhorn
    for _ in range(200):
        rs = T_skew.sum(axis=1, keepdims=True); rs[rs == 0] = 1
        T_skew = T_skew * (d / rs); np.fill_diagonal(T_skew, 0.0)
        cs = T_skew.sum(axis=0, keepdims=True); cs[cs == 0] = 1
        T_skew = T_skew * (d / cs); np.fill_diagonal(T_skew, 0.0)
    run("skewed / Sinkhorn-normalised", T_skew, n, d)

    # 4. Random dense hose matrix
    T_rand = random_hose_matrix(n=n, d=d, rng=rng, tight=True)
    run("random hose (tight row/col sums = d)", T_rand, n, d)


if __name__ == "__main__":
    main()
