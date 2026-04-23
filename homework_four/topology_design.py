"""
Assignment 4 - Arbitrary Traffic Matrix topology design.

We jointly design a directed topology on n nodes (each node has at most d
out-going and d in-coming unit-capacity links) and route a hose-model
traffic matrix T to maximise the concurrent flow throughput lambda.

This is implemented as a mixed-integer linear program solved with Gurobi.

We use a destination-aggregated multi-commodity-flow formulation: flows
sharing the same destination can always be merged without loss for
fractional routing, which reduces the number of flow variables from
O(n^4) to O(n^3).

Decision variables
------------------
c[u, v]   : integer in [0, d]   -- number of unit-capacity links from u to v
g[t, u, v]: continuous >= 0    -- flow destined for t on edge (u, v)
lam       : continuous >= 0    -- concurrent flow throughput

Objective
---------
maximise  lam

Constraints
-----------
(1)  sum_v c[u, v] <= d                 for every u        (out-degree)
(2)  sum_u c[u, v] <= d                 for every v        (in-degree)
(3)  c[u, u] = 0                        for every u        (no self-loops)
(4)  flow conservation for every destination t and every node u:
        sum_v g[t, u, v] - sum_v g[t, v, u]
            = { + lam * T[u, t]         if u != t
                - lam * sum_s T[s, t]   if u == t }
(5)  sum_t g[t, u, v] <= c[u, v]        for every directed pair (u, v)
"""

from __future__ import annotations

import numpy as np
import gurobipy as gp
from gurobipy import GRB


def solve_topology(T: np.ndarray, d: int = 4, time_limit: float | None = None,
                   verbose: bool = False) -> dict:
    """Solve the joint topology design + multi-commodity flow MILP.

    Parameters
    ----------
    T : (n, n) array of non-negative floats with T[i, i] = 0 and
        row / column sums <= d  (hose-model traffic matrix).
    d : out-degree = in-degree bound per node (unit-capacity links).
    time_limit : optional wall-clock limit in seconds passed to Gurobi.
    verbose : print Gurobi log if True.

    Returns
    -------
    dict with keys
        lambda   : optimal concurrent flow value
        capacity : (n, n) integer matrix c[u, v]
        flow     : dict mapping (t, u, v) -> flow value (destination-based)
        status   : Gurobi status code
    """
    T = np.asarray(T, dtype=float)
    n = T.shape[0]
    assert T.shape == (n, n), "T must be square"
    assert np.allclose(np.diag(T), 0.0), "diagonal of T must be zero"

    model = gp.Model("topology_design")
    model.Params.OutputFlag = 1 if verbose else 0
    if time_limit is not None:
        model.Params.TimeLimit = time_limit

    # ---- decision variables -------------------------------------------------
    c = model.addVars(n, n, vtype=GRB.INTEGER, lb=0, ub=d, name="c")
    for i in range(n):
        c[i, i].ub = 0  # no self-loops

    # only destinations that actually receive traffic need a commodity
    destinations = [t for t in range(n) if T[:, t].sum() > 1e-12]

    g = model.addVars(destinations, range(n), range(n),
                      lb=0.0, vtype=GRB.CONTINUOUS, name="g")

    lam = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="lambda")

    # ---- degree constraints -------------------------------------------------
    for u in range(n):
        model.addConstr(gp.quicksum(c[u, v] for v in range(n)) <= d,
                        name=f"out_deg_{u}")
        model.addConstr(gp.quicksum(c[v, u] for v in range(n)) <= d,
                        name=f"in_deg_{u}")

    # ---- flow conservation (destination based) -----------------------------
    for t in destinations:
        for u in range(n):
            outflow = gp.quicksum(g[t, u, v] for v in range(n) if v != u)
            inflow = gp.quicksum(g[t, v, u] for v in range(n) if v != u)
            if u == t:
                rhs = -lam * float(T[:, t].sum())
            else:
                rhs = lam * float(T[u, t])
            model.addConstr(outflow - inflow == rhs,
                            name=f"flow_{t}_{u}")

    # forbid flow that "enters" its destination only to leave again:
    # a valid solution never needs g[t, t, v] > 0.  (optional tightening)
    for t in destinations:
        for v in range(n):
            g[t, t, v].ub = 0.0

    # ---- capacity constraints ----------------------------------------------
    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            model.addConstr(
                gp.quicksum(g[t, u, v] for t in destinations) <= c[u, v],
                name=f"cap_{u}_{v}")

    # ---- objective ----------------------------------------------------------
    model.setObjective(lam, GRB.MAXIMIZE)
    model.optimize()

    capacity = np.zeros((n, n), dtype=int)
    flow: dict[tuple[int, int, int], float] = {}
    lam_val = None
    if model.SolCount > 0:
        lam_val = lam.X
        for u in range(n):
            for v in range(n):
                capacity[u, v] = int(round(c[u, v].X))
        for t in destinations:
            for u in range(n):
                for v in range(n):
                    if u == v:
                        continue
                    val = g[t, u, v].X
                    if val > 1e-9:
                        flow[(t, u, v)] = val

    return {
        "lambda": lam_val,
        "capacity": capacity,
        "flow": flow,
        "status": model.Status,
    }


# ---------------------------------------------------------------------------
# helper: sample a random hose-model traffic matrix
# ---------------------------------------------------------------------------

def random_hose_matrix(n: int = 8, d: int = 4, rng: np.random.Generator | None = None,
                       tight: bool = True) -> np.ndarray:
    """Produce a random traffic matrix in the hose set Those.

    When ``tight`` is True we return a matrix whose row *and* column sums are
    exactly d (the hardest case for concurrent flow).  The construction uses
    Sinkhorn-style iterative scaling followed by zeroing the diagonal and a
    final re-scaling.
    """
    if rng is None:
        rng = np.random.default_rng()
    T = rng.random((n, n))
    np.fill_diagonal(T, 0.0)
    if tight:
        # iterative row/column normalisation to reach row = col sum = d
        for _ in range(200):
            row_sums = T.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            T = T * (d / row_sums)
            np.fill_diagonal(T, 0.0)

            col_sums = T.sum(axis=0, keepdims=True)
            col_sums[col_sums == 0] = 1.0
            T = T * (d / col_sums)
            np.fill_diagonal(T, 0.0)
    else:
        T = T * (d / T.sum(axis=1, keepdims=True))
        np.fill_diagonal(T, 0.0)
    return T
