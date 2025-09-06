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
    "project_capped_simplex_kl",
]


def weights_uniform(n: int) -> Array:
    """Return the uniform weight vector of length :math:`n`.

    Parameters
    ----------
    n : int
        Number of samples.

    Returns
    -------
    ndarray of shape ``(n,)``
        The vector with all entries equal to :math:`1/n`.
    """
    return np.full(n, 1.0 / n)


def normalize_simplex(q: Array) -> Array:
    """Project to the full simplex by non‑negativity and renormalization.

    Sets negative entries to zero and divides by the sum; if the sum is
    non‑positive, returns the uniform vector.

    Parameters
    ----------
    q : ndarray of shape ``(n,)``
        Input weights.

    Returns
    -------
    ndarray of shape ``(n,)``
        Non‑negative weights summing to one.
    """
    q = np.maximum(q, 0.0)
    s = q.sum()
    if s <= 0:
        return np.full_like(q, 1.0 / len(q))
    return q / s


def weighted_mean(X: Array, q: Array) -> Array:
    r"""Weighted mean :math:`\mu_q = E_q[X]`.

    Parameters
    ----------
    X : ndarray of shape ``(n, d)``
        Data matrix.
    q : ndarray of shape ``(n,)``
        Weights summing to one.

    Returns
    -------
    ndarray of shape ``(d,)``
        The weighted mean.
    """
    return X.T @ q


def weighted_covariance(X: Array, q: Array, center: Array | None = None) -> Array:
    """Weighted covariance :math:`\\Sigma_q = E_q[(X-\\mu_q)(X-\\mu_q)^\\top]`.

    Parameters
    ----------
    X : ndarray of shape ``(n, d)``
        Data matrix.
    q : ndarray of shape ``(n,)``
        Weights summing to one.
    center : ndarray of shape ``(d,)``, optional
        If provided, use this center instead of the weighted mean.

    Returns
    -------
    ndarray of shape ``(d, d)``
        The weighted covariance matrix.
    """
    mu = weighted_mean(X, q) if center is None else center
    Xc = X - mu
    return (Xc.T * q) @ Xc


def top_eigvec_symmetric(A: Array) -> tuple[float, Array]:
    """Largest eigenvalue and eigenvector of a symmetric matrix.

    Parameters
    ----------
    A : ndarray of shape ``(d, d)``
        Symmetric matrix.

    Returns
    -------
    (float, ndarray)
        Top eigenvalue and the corresponding unit‑norm eigenvector.
    """
    w, V = eigh(A)
    return float(w[-1].real), V[:, -1].real


def F_cov_spectral(X: Array, q: Array) -> tuple[float, Array, Array]:
    """Computes :math:`(\\lambda_{max}(\\Sigma_q), \\mu_q, v)`.

    Returns the top eigenvalue of the weighted covariance, the weighted
    mean and the corresponding eigenvector.

    Parameters
    ----------
    X : ndarray of shape ``(n, d)``
        Data matrix.
    q : ndarray of shape ``(n,)``
        Weights summing to one.

    Returns
    -------
    (float, ndarray, ndarray)
        ``(lambda_max, mu, v)``.
    """
    mu = weighted_mean(X, q)
    Sigma = weighted_covariance(X, q, center=mu)
    lam, v = top_eigvec_symmetric(Sigma)
    return float(lam), mu, v


def project_capped_simplex_kl(z: Array, cap: float, total: float = 1.0) -> Array:
    """KL I‑projection onto the capped simplex.

    Solve

    .. math::

        \\min_{q \\ge 0} D_{KL}(q\\,\\|\\,z) \\quad
        \\text{s.t. } \\sum_i q_i = \\text{total}, \\; q_i \\le \\text{cap}.

    The KKT conditions imply :math:`q_i = \\min(\\text{cap}, t z_i)` for some scalar
    :math:`t > 0`.  The algorithm finds the active capped set by scanning the
    coordinates in descending order of :math:`z_i` and checking consistency
    of the implied :math:`t`.

    Parameters
    ----------
    z : ndarray of shape ``(n,)``
        Non‑negative reference measure.
    cap : float
        Upper bound for each coordinate.
    total : float, default=1.0
        Desired sum of the projected vector.

    Returns
    -------
    ndarray of shape ``(n,)``
        The projection of :math:`z` onto the capped simplex in KL geometry.
    """
    z = np.maximum(np.asarray(z, dtype=float), 1e-300)
    n = z.size
    if n == 0:
        return z
    if cap <= 0:
        # All mass must be zero if cap==0 and total>0 is infeasible; return uniform tiny vector
        return np.full_like(z, 0.0)
    # Sort z descending with original indices
    idx = np.argsort(-z)
    zs = z[idx]
    # Prefix sums of zs for fast tail sums
    prefix = np.cumsum(zs)
    # Maximum number of capped coordinates cannot exceed floor(total / cap)
    k_max = min(n, int(np.floor(total / cap)))

    def tail_sum_from(k: int) -> float:
        # sum_{i=k..n-1} zs[i] when using 0-based Python indices
        if k >= n:
            return 0.0
        return float(prefix[-1] - (prefix[k - 1] if k > 0 else 0.0))

    chosen_k = None
    chosen_t = None
    for k in range(0, k_max + 1):
        S_tail = tail_sum_from(k)
        if S_tail < 1e-300:
            # Only feasible if total == k*cap (i.e., all capped)
            if abs(total - k * cap) <= 1e-12:
                chosen_k, chosen_t = k, 0.0
                break
            else:
                continue
        t_k = (total - k * cap) / S_tail
        if t_k < 0:
            # too many capped; increasing k only decreases numerator, so break
            break
        theta = np.inf if t_k == 0 else (cap / t_k)
        cond_upper = True if k == 0 else (zs[k - 1] >= theta)
        cond_lower = True if k == n else (zs[k] <= theta)
        if cond_upper and cond_lower:
            chosen_k, chosen_t = k, t_k
            break

    if chosen_k is None:
        # Fallback to bisection method for robustness
        return project_capped_simplex_kl(z, cap=cap, total=total)

    k = chosen_k
    t = max(0.0, chosen_t)
    q_sorted = np.empty_like(zs)
    if k > 0:
        q_sorted[:k] = cap
    if k < n:
        q_sorted[k:] = t * zs[k:]
    q = np.empty_like(z)
    q[idx] = q_sorted
    # Renormalize for numerical safety
    s = float(q.sum())
    if s > 0:
        q *= total / s
    return q
