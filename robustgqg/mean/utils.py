from __future__ import annotations
import numpy as np
from numpy.linalg import eigh
from robustgqg.types import Array


__all__ = [
"weights_uniform",
"normalize_simplex",
"weighted_mean",
"weighted_covariance",
"top_eigvec_symmetric",
"F_cov_spectral",
]


def weights_uniform(n: int) -> Array:
    return np.full(n, 1.0 / n)




def normalize_simplex(q: Array) -> Array:
    q = np.maximum(q, 0.0)
    s = q.sum()
    if s <= 0:
        return np.full_like(q, 1.0 / len(q))
    return q / s



def weighted_mean(X: Array, q: Array) -> Array:
    return X.T @ q




def weighted_covariance(X: Array, q: Array, center: Array | None = None) -> Array:
    mu = weighted_mean(X, q) if center is None else center
    Xc = X - mu
    return (Xc.T * q) @ Xc




def top_eigvec_symmetric(A: Array) -> tuple[float, Array]:
    w, V = eigh(A)
    return float(w[-1].real), V[:, -1].real




def F_cov_spectral(X: Array, q: Array) -> tuple[float, Array, Array]:
    mu = weighted_mean(X, q)
    Sigma = weighted_covariance(X, q, center=mu)
    lam, v = top_eigvec_symmetric(Sigma)
    return float(lam), mu, v