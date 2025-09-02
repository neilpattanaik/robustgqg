from __future__ import annotations
import numpy as np
from robustgqg.types import Array
from robustgqg.mean.base import BaseMeanEstimator
from robustgqg.utils import project_capped_simplex_kl

__all__ = ["ExplicitLowRegretMean"]


class ExplicitLowRegretMean(BaseMeanEstimator):
    """Algorithm 1: Explicit low-regret robust mean estimation (bounded covariance).

    Update rule:
        q̃_i = q_i * (1 - η_k * g_i),   where g_i = (vᵀ(x_i - µ_q))²,  v = top-eigenvector(Σ_q)
        q   = Proj_{Δ_{n,ε}}(q̃) using KL projection.
    """

    def _project_onto_simplex(self, z: Array) -> Array:
        return project_capped_simplex_kl(np.maximum(z, 0.0), cap=self.cap, total=1.0)

    
