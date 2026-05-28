"""Verification cost model for CaDDTree, fitted via convex regression."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch


class CostModel(ABC):
    """Abstract cost model interface for CaDDTree."""

    @property
    @abstractmethod
    def Cd(self) -> float:
        """Draft cost per round (seconds)."""

    @abstractmethod
    def Cv(self, n: int, context_length: int) -> float:
        """Verification cost for *n* tree nodes at *context_length* (seconds)."""

    def marginal_Cv(self, n: int, context_length: int) -> float:
        """Marginal verification cost: c_n = Cv(n) - Cv(n-1)."""
        if n <= 0:
            return 0.0
        return self.Cv(n, context_length) - self.Cv(n - 1, context_length)

    def theta_star(self, n: int, phi_n: float, context_length: int) -> float:
        """Surrogate throughput (1 + Phi*(n)) / (Cd + Cv(n; ℓ))."""
        return (1.0 + phi_n) / (self.Cd + self.Cv(n, context_length))


def _convex_regression(y: np.ndarray) -> np.ndarray:
    """
    Monotone + convex regression via constrained least squares.

    Solves:
        min_β  (1/2)||y - β||₂²
        s.t.   D²β ≥ 0   (convexity: slopes non-decreasing)
               Δβ  ≥ 0   (monotonicity: all slopes ≥ 0)
    """
    try:
        import cvxpy as cp
    except ImportError:
        raise ImportError(
            "cvxpy is required for convex regression fitting. "
            "Install with: pip install cvxpy"
        )

    n = len(y)
    if n < 3:
        return y.copy()

    D2 = np.zeros((n - 2, n))
    for i in range(n - 2):
        D2[i, i] = 1.0
        D2[i, i + 1] = -2.0
        D2[i, i + 2] = 1.0

    beta = cp.Variable(n)
    objective = cp.Minimize(0.5 * cp.sum_squares(y - beta))
    constraints = [D2 @ beta >= 0, cp.diff(beta) >= 0]
    problem = cp.Problem(objective, constraints)

    problem.solve(solver=cp.CLARABEL, verbose=False)
    if beta.value is None:
        problem.solve(solver=cp.SCS, verbose=False)
    if beta.value is None:
        print("Convex regression failed; returning raw data.")
        return y.copy()

    return np.array(beta.value)


class ConvexCostModel(CostModel):
    """Verification cost model fitted via convex regression over profiled context lengths."""

    def __init__(
        self,
        context_lengths: list[int],
        node_counts: np.ndarray,
        fitted_curves: dict[int, np.ndarray],
        draft_cost: float,
    ):
        self._draft_cost = draft_cost
        self._ctx_lengths = sorted(context_lengths)
        self._node_counts = np.asarray(node_counts, dtype=np.float64)
        self._fitted_curves = {
            ctx: np.asarray(vals, dtype=np.float64)
            for ctx, vals in fitted_curves.items()
        }
        self._max_n = int(self._node_counts[-1]) if len(self._node_counts) > 0 else 0

    @property
    def Cd(self) -> float:
        return self._draft_cost

    def _lookup(self, n: int, ctx: int) -> float:
        curve = self._fitted_curves[ctx]
        if n < 0:
            n = 0
        if n < len(curve):
            return float(curve[n])
        # Linear extrapolation from last two points
        if len(curve) >= 2:
            slope = curve[-1] - curve[-2]
            return float(curve[-1] + slope * (n - len(curve) + 1))
        return float(curve[-1])

    def Cv(self, n: int, context_length: int) -> float:
        ctx = self._ctx_lengths
        if len(ctx) == 0:
            return 0.0
        if context_length <= ctx[0]:
            return self._lookup(n, ctx[0])
        if context_length >= ctx[-1]:
            return self._lookup(n, ctx[-1])
        for i in range(len(ctx) - 1):
            if ctx[i] <= context_length <= ctx[i + 1]:
                alpha = (context_length - ctx[i]) / (ctx[i + 1] - ctx[i])
                v_lo = self._lookup(n, ctx[i])
                v_hi = self._lookup(n, ctx[i + 1])
                return (1 - alpha) * v_lo + alpha * v_hi
        return self._lookup(n, ctx[-1])

    def save(self, path: str) -> None:
        torch.save({
            "type": "convex",
            "context_lengths": self._ctx_lengths,
            "node_counts": self._node_counts.tolist(),
            "fitted_curves": {
                ctx: vals.tolist() for ctx, vals in self._fitted_curves.items()
            },
            "draft_cost": self._draft_cost,
        }, path)

    @classmethod
    def load(cls, path: str) -> "ConvexCostModel":
        data = torch.load(path, weights_only=False, map_location="cpu")
        assert data["type"] == "convex"
        return cls(
            context_lengths=data["context_lengths"],
            node_counts=np.array(data["node_counts"]),
            fitted_curves={
                ctx: np.array(vals) for ctx, vals in data["fitted_curves"].items()
            },
            draft_cost=data["draft_cost"],
        )

    @classmethod
    def from_profile_data(
        cls,
        profile_path: str,
        draft_cost: float,
        max_nodes: int = 1024,
    ) -> "ConvexCostModel":
        data = torch.load(profile_path, weights_only=False, map_location="cpu")
        pdata = data["profile_data"]
        all_ctx = pdata["context_lengths"]

        fitted_curves = {}
        max_n_global = 0

        for ctx_len in all_ctx:
            medians = pdata["medians"][ctx_len]
            ns = np.array(sorted(medians.keys()), dtype=float)
            lats = np.array([medians[int(n)] for n in ns])
            mask = ns <= max_nodes
            ns, lats = ns[mask], lats[mask]
            if len(ns) < 3:
                print(f"Skipping ctx_len={ctx_len}: only {len(ns)} data points after filtering.")
                continue

            print(f"Fitting ctx_len={ctx_len} ({len(ns)} points)...")
            beta = _convex_regression(lats)

            max_n = int(ns[-1])
            full_ns = np.arange(0, max_n + 1, dtype=float)
            full_vals = np.interp(full_ns, ns, beta)
            fitted_curves[ctx_len] = full_vals
            max_n_global = max(max_n_global, max_n)

        node_counts = np.arange(0, max_n_global + 1)
        return cls(
            context_lengths=[c for c in all_ctx if c in fitted_curves],
            node_counts=node_counts,
            fitted_curves=fitted_curves,
            draft_cost=draft_cost,
        )


def load_cost_model(path: str) -> ConvexCostModel:
    """Load a saved ConvexCostModel."""
    data = torch.load(path, weights_only=False, map_location="cpu")
    assert data.get("type") == "convex", f"Expected convex cost model, got: {data.get('type')}"
    return ConvexCostModel(
        context_lengths=data["context_lengths"],
        node_counts=np.array(data["node_counts"]),
        fitted_curves={
            ctx: np.array(vals) for ctx, vals in data["fitted_curves"].items()
        },
        draft_cost=data["draft_cost"],
    )
