from __future__ import annotations
import numpy as np
from numpy.linalg import eigh, solve
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



def project_capped_simplex_kl(z: Array, cap: float, total: float = 1.0) -> Array:
    """KL (I-projection) using the **active-set water-filling** algorithm from the paper (pp. 48–49).

    Problem: minimize D_KL(q||z) subject to q ≥ 0, Σ q = total, and q_i ≤ cap.
    KKT ⇒ q_i = min(cap, t z_i). The algorithm determines the active capped set by
    searching over k (number of capped coordinates) in order of z_i size.

    Steps:
      1) Sort indices by z_i descending: z_(1) ≥ ... ≥ z_(n).
      2) For each k = 0..k_max, compute t_k = (total - k·cap) / Σ_{i>k} z_(i).
      3) Find k s.t. consistency holds: t_k z_(k) ≥ cap (for i ≤ k) and t_k z_(k+1) ≤ cap.
      4) Set q_(i) = cap for i ≤ k, and q_(i) = t_k z_(i) for i > k; undo the sort.
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
        return float(prefix[-1] - (prefix[k-1] if k > 0 else 0.0))

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
        cond_upper = True if k == 0 else (zs[k-1] >= theta)
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
        q *= (total / s)
    return q
