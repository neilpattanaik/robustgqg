from __future__ import annotations

from robustgqg.mean.base import BaseMeanEstimator
from robustgqg.types import Array
from robustgqg.utils import normalize_simplex

__all__ = ["FilterMean"]


class FilterMean(BaseMeanEstimator):
    r"""Filter algorithm for robust mean estimation (bounded covariance).

    This variant projects onto the full probability simplex :math:`\Delta_n`
    at each iteration (i.e., just renormalizes the non‑negative scores).

    Examples
    --------
    >>> import numpy as np
    >>> from robustgqg.mean import FilterMean
    >>> X = np.random.default_rng(42).normal(size=(50, 3))
    >>> est = FilterMean(X, eps=0.1, verbose=False)
    >>> out = est.run()
    >>> out['F'] >= 0 and out['q'].sum() == 1.0
    True
    """

    def _project_onto_simplex(self, q: Array) -> Array:
        return normalize_simplex(q)
