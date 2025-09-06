from __future__ import annotations

import numpy as np

from robustgqg.mean.base import BaseMeanEstimator
from robustgqg.types import Array
from robustgqg.utils import project_capped_simplex_kl

__all__ = ["ExplicitLowRegretMean"]


class ExplicitLowRegretMean(BaseMeanEstimator):
    """Explicit low‑regret robust mean (bounded covariance model).

    Performs the same multiplicative update as :class:`FilterMean` but
    projects onto the capped simplex :math:`\\Delta_{n, \\varepsilon}` using a
    KL (I‑projection).

    Examples
    --------
    >>> import numpy as np
    >>> from robustgqg.mean import ExplicitLowRegretMean
    >>> X = np.random.default_rng(123).normal(size=(40, 2))
    >>> est = ExplicitLowRegretMean(X, eps=0.1)
    >>> out = est.run()
    >>> out['q'].max() <= 1/((1-0.1)*40) + 1e-12
    True
    """

    def _project_onto_simplex(self, z: Array) -> Array:
        return project_capped_simplex_kl(np.maximum(z, 0.0), cap=self.cap, total=1.0)
