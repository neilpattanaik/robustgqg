from __future__ import annotations
import numpy as np
from typing import Optional, Dict
from robustgqg.types import Array
from robustgqg.mean.options import MeanOptions


class BaseMeanEstimator:
    """Shared interface for mean estimators.

    Subclasses must implement `run()`.
    """
    def __init__(self, X: Array, opts: MeanOptions):
        self.X = np.asarray(X, dtype=float)
        self.n, self.d = self.X.shape
        self.opts = opts
        if not (0.0 < opts.eps < 0.5):
            raise ValueError("opts.eps must be in (0, 0.5)")
        self.cap = 1.0 / ((1.0 - opts.eps) * self.n)


    def run(self, q0: Optional[Array] = None) -> Dict[str, Array | float | int]: # pragma: no cover
        raise NotImplementedError