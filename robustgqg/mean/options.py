from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class MeanOptions:
    """Hyperparameters and stopping criteria for mean estimation algorithms.


    Attributes
    ----------
    eps : float
        Contamination upper bound ε in (0, 0.5).
    max_iter : int
        Maximum number of iterations.
    eta : float
        Base stepsize; algorithms use η_k = η / (2·B_k) with B_k derived from current g.
    target_xi : Optional[float]
        Optional target threshold ξ for early stopping when F(q)=||Σ_q|| ≤ ξ.
    tol_rel : float
        Relative improvement tolerance for early stopping.
    verbose : bool
        Print progress every ~25 iters.
    """
    eps: float
    max_iter: int = 2000
    eta: float = 1.0
    target_xi: Optional[float] = None
    tol_rel: float = 1e-6
    verbose: bool = False