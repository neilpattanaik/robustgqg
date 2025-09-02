from __future__ import annotations
import math
import numpy as np
from typing import Optional, Dict
from robustgqg.types import Array
from robustgqg.mean.base import BaseMeanEstimator
from robustgqg.mean.options import MeanOptions
from robustgqg.mean.utils import (
    weights_uniform,
    normalize_simplex,
    F_cov_spectral,
)
from robustgqg.mean.projections import project_capped_simplex_kl

__all__ = ["ExplicitLowRegretMean"]

class ExplicitLowRegretMean(BaseMeanEstimator):
    """Algorithm 1: Explicit low-regret robust mean estimation (bounded covariance).

    Update rule:
        q̃_i = q_i * (1 - η_k * g_i),   where g_i = (vᵀ(x_i - µ_q))²,  v = top-eigenvector(Σ_q)
        q   = Proj_{Δ_{n,ε}}(q̃) using either L2 (default) or KL projection.
    """

    def _project_deleted_simplex(self, z: Array) -> Array:
        return project_capped_simplex_kl(np.maximum(z, 0.0), cap=self.cap, total=1.0)


    def run(self, q0: Optional[Array] = None) -> Dict[str, Array | float | int]:
        # Initialize weights q: uniform over n points if not provided
        q = weights_uniform(self.n) if q0 is None else normalize_simplex(q0)
        last_F = math.inf
        for k in range(self.opts.max_iter):
            # Step 1: compute current objective and certificate
            #   μ_q = weighted mean, Σ_q = weighted covariance, v = top eigenvector(Σ_q)
            F, mu, v = F_cov_spectral(self.X, q)

            # Optionally print progress
            if self.opts.verbose and (k % 25 == 0 or k == 0):
                print(f"[Explicit] iter={k}  F={F:.6g}")

            # Early stop if we already satisfy target covariance bound
            if self.opts.target_xi is not None and F <= self.opts.target_xi:
                return {"q": q, "mu": mu, "F": F, "iters": k}
            
            # Step 2: form quasi‑gradient g_i = (vᵀ(x_i − μ_q))² for each point
            Xc_v = (self.X - mu) @ v
            g = Xc_v ** 2

            # Step 3: set learning rate η_k = η / (2 B_k), with B_k = max g_i
            Bk = max(1e-12, float(np.max(g)))
            eta_k = self.opts.eta / (2.0 * Bk)

            # Step 4: multiplicative update q̃_i = q_i (1 − η_k g_i)
            q_tilde = q * (1.0 - eta_k * g)

            # Step 5: project back to feasible set Δ_{n,ε}
            q = self._project_deleted_simplex(q_tilde)

            # Step 6: check relative progress on F for convergence
            if last_F < math.inf and abs(F - last_F) <= self.opts.tol_rel * max(1.0, last_F):
                return {"q": q, "mu": mu, "F": F, "iters": k + 1}
            last_F = F
        
        # Final evaluation after exhausting max_iter
        F, mu, _ = F_cov_spectral(self.X, q)
        return {"q": q, "mu": mu, "F": F, "iters": self.opts.max_iter}
