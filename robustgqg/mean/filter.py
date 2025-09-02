from __future__ import annotations
from robustgqg.mean.base import BaseMeanEstimator
from robustgqg.utils import normalize_simplex
from robustgqg.types import Array

__all__ = ["FilterMean"]

class FilterMean(BaseMeanEstimator):
    """
    Algorithm 2: Filter algorithm for robust mean estimation (bounded covariance model).

    Idea (paper): maintain unnormalized scores c_i, update
        c_i ← c_i · (1 - η_k g_i),   with g_i = (v^T (x_i - μ_q))^2,
    where v is the top eigenvector of Σ_q, then renormalize to the simplex:
        q = c / sum(c)  (projection to Δ_n).
    """

    def _project_onto_simplex(self, q: Array) -> Array:
        return normalize_simplex(q)