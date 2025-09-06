from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np

from robustgqg.types import Array
from robustgqg.utils import (
    F_cov_spectral,
    normalize_simplex,
    weights_uniform,
)


class BaseMeanEstimator(ABC):
    """Shared interface for robust mean estimators.

    This base class implements the generic multiplicative‑weights
    filtering scheme for mean estimation under bounded covariance.  Given
    weights :math:`q` on samples :math:`x_i`, the algorithm computes the weighted
    mean :math:`\mu_q` and covariance :math:`\Sigma_q`, selects the leading
    eigenvector :math:`v` of :math:`\Sigma_q`, and updates

    .. math::

        q_i = q_i\\,(1 - \\eta_k\\, g_i),\\quad
        g_i = (v^T (x_i - \\mu_q))^2,

    followed by a projection back to a simplex that is supplied by the
    subclass via :meth:`_project_onto_simplex`.

    Subclasses specify the feasible set:

    - ``FilterMean`` projects to the full simplex :math:`\\Delta_n`.
    - ``ExplicitLowRegretMean`` projects to the capped simplex :math:`\\Delta_{n,\\varepsilon} = \{q \\ge 0: \\sum q_i = 1,\\; q_i \\le 1/((1-\\varepsilon)n)\}`.
    """

    def __init__(
        self,
        X: Array,
        eps,
        max_iter: int = 2000,
        eta: float = 1.0,
        target_xi: Optional[float] = None,
        tol_rel: float = 1e-6,
        verbose: bool = False,
    ):
        """Initialize a mean estimator.

        Parameters
        ----------
        X : ndarray
            Data matrix of shape (n, d).
        eps : float
            Contamination upper bound ε in (0, 0.5).
        max_iter : int
            Maximum number of iterations.
        eta : float
            Base step size.
        target_xi : Optional[float]
            Optional early-stop threshold on covariance spectral norm.
        tol_rel : float
            Relative improvement tolerance for early stopping.
        verbose : bool
            If True, prints periodic progress.
        """
        self.X = np.asarray(X, dtype=float)
        self.n, self.d = self.X.shape
        self.eps = float(eps)
        self.max_iter = int(max_iter)
        self.eta = float(eta)
        self.target_xi = target_xi
        self.tol_rel = float(tol_rel)
        self.verbose = bool(verbose)

        if not (0.0 < self.eps < 0.5):
            raise ValueError("eps must be in (0, 0.5)")
        self.cap = 1.0 / ((1.0 - self.eps) * self.n)

    def run(self, q0: Optional[Array] = None) -> Dict[str, Array | float | int]:
        """Run the multiplicative‑weights mean estimator.

        Parameters
        ----------
        q0 : ndarray of shape ``(n,)``, optional
            Initial weights.  If ``None``, uses the uniform distribution.

        Returns
        -------
        dict
            A dictionary with keys ``'q'`` (final weights), ``'mu'``
            (weighted mean), ``'F'`` (top eigenvalue of the weighted
            covariance) and ``'iters'`` (iterations used).

        Examples
        --------
        >>> import numpy as np
        >>> from robustgqg.mean import FilterMean
        >>> X = np.random.default_rng(0).normal(size=(100, 2))
        >>> algo = FilterMean(X, eps=0.1)
        >>> out = algo.run()
        >>> out['mu'].shape, out['q'].shape
        ((2,), (100,))
        """
        # Initialize weights q: uniform over n points if not provided
        q = weights_uniform(self.n) if q0 is None else normalize_simplex(q0)
        last_F = math.inf
        for k in range(self.max_iter):
            # Step 1: compute current objective and certificate
            #   μ_q = weighted mean, Σ_q = weighted covariance, v = top eigenvector(Σ_q)
            F, mu, v = F_cov_spectral(self.X, q)

            # Optionally print progress
            if self.verbose and (k % 25 == 0 or k == 0):
                print(f"iter={k}  F={F:.6g}")

            # Early stop if we already satisfy target covariance bound
            if self.target_xi is not None and F <= self.target_xi:
                return {"q": q, "mu": mu, "F": F, "iters": k}

            # Step 2: form quasi‑gradient g_i = (vᵀ(x_i − μ_q))² for each point
            Xc_v = (self.X - mu) @ v
            g = Xc_v**2

            # Step 3: set learning rate η_k = η / (2 B_k), with B_k = max g_i
            Bk = max(1e-12, float(np.max(g)))
            eta_k = self.eta / (2.0 * Bk)

            # Step 4: multiplicative update q̃_i = q_i (1 − η_k g_i)
            q_tilde = q * (1.0 - eta_k * g)

            # Step 5: project back to feasible set Δ_{n,ε} or Δ_n (depending on subclass algorithm)
            q = self._project_onto_simplex(q_tilde)

            # Step 6: check relative progress on F for convergence
            if last_F < math.inf and abs(F - last_F) <= self.tol_rel * max(1.0, last_F):
                return {"q": q, "mu": mu, "F": F, "iters": k + 1}
            last_F = F

        # Final evaluation after exhausting max_iter
        F, mu, _ = F_cov_spectral(self.X, q)
        return {"q": q, "mu": mu, "F": F, "iters": self.max_iter}

    @abstractmethod
    def _project_onto_simplex(self, q: Array) -> Array:
        pass
