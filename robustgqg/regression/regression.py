# ruff: noqa: E501
"""
Robust linear regression via generalized quasi‑gradient filtering.

This module implements the filter algorithm (Algorithm 3) from
Zhu, Jiao and Steinhardt (2020), “Robust estimation via generalized
quasi‑gradients.” The algorithm maintains a probability distribution
over the :math:`n` training samples (a weight vector :math:`q` on the simplex)
and iteratively down‑weights outliers using multiplicative updates.  At
each iteration it checks two moment conditions:

* Hypercontractivity functional :math:`F_1(q)`

  .. math::

      F_1(q) = \\sup_{v \\in \\mathbb{R}^d}
        \\frac{\\mathbb{E}_q[(v^\top X)^4]}
             {\\big(\\mathbb{E}_q[(v^\top X)^2]\\big)^2},

  where :math:`X` is the design matrix and :math:`\\mathbb{E}_q` denotes
  expectation with respect to :math:`q`.  :math:`F_1(q)` measures heavy tails
  along some direction.  If :math:`F_1(q)` exceeds a threshold
  :math:`kappa_prime_sq` the algorithm forms a quasi‑gradient :math:`g1` and
  down‑weights the offending samples.

* Bounded noise functional :math:`F_2(q)`

  .. math::

      F_2(q) = \\sup_{v \\in \\mathbb{R}^d}
        \\frac{\\mathbb{E}_q\\big[(Y - X\\theta(q))^2\\,(v^\\top X)^2\\big]}
             {\\mathbb{E}_q[(v^\\top X)^2]},

  which quantifies the residual variance after a weighted least‑squares
  fit at the current :math:`q`.  If :math:`F_2(q)` exceeds a threshold
  :math:`sigma_prime_sq` the algorithm uses the quasi‑gradient :math:`g2` to
  perform another multiplicative update.

The hypercontractivity quasi‑gradient :math:`g1` is computed via a
Sum‑of‑Squares (SoS) relaxation on whitened data (see
``HyperContractivityOracle``). The noise quasi‑gradient :math:`g2` is derived
from a generalised eigenvalue problem.  Updates multiply each weight :math:`q[i]`
by :math:`1 − eta * g[i]` with :math:`eta` chosen as the reciprocal of
the largest gradient entry, and then re‑normalize :math:`q` to the simplex.
When both moment conditions are satisfied the algorithm returns the
final weight vector along with the weighted least‑squares estimator
:math:`\\theta(q)`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
from numpy.linalg import eigh, lstsq

from robustgqg.regression.hypercontractivity import HyperContractivityOracle
from robustgqg.types import Array


@dataclass
class FilterResult:
    """Container for the output of :meth:`FilterRegression.run`.

    Supports both attribute and dictionary-like access to fields.

    Parameters
    ----------
    q, weights : ndarray
        Final weight vector on the samples.  ``weights`` is provided as
        an alias of ``q`` for clarity.
    theta, coefficients : ndarray
        Weighted least–squares estimator.  ``coefficients`` is an
        alias of ``theta``.
    F1, F2 : float
        Values of the hypercontractivity and noise objectives at
        termination.
    iters, iterations : int
        Number of iterations performed.  ``iterations`` mirrors
        ``iters``.
    status : str
        Reason for termination: either ``'done'`` if both conditions
        were satisfied or ``'max_iter_reached'`` if the maximum number
        of iterations was hit.
    """

    q: Array
    weights: Array
    theta: Array
    coefficients: Array
    F1: float
    F2: float
    iters: int
    iterations: int
    status: str

    def __getitem__(self, key: str):
        # Allow dictionary-like access to attributes
        return getattr(self, key)


class FilterRegression:
    """Robust linear regression via the filter algorithm.

    The goal of robust regression is to recover the linear model
    :math:`y = X \\theta` in the presence of adversarially corrupted
    observations.  The filter algorithm maintains a probability
    distribution :math:`q` on the :math:`n` samples and iteratively updates
    :math:`q` to down‑weight points that violate certain moment conditions.

    Two moment functionals govern the updates:

    * **Hypercontractivity functional** :math:`F_1(q)`:

      .. math::

          F_1(q) = \\sup_{v \\in \\mathbb{R}^d} \\frac{\\mathbb{E}_q[(v^T X)^4]}{\\big(\\mathbb{E}_q[(v^T X)^2]\\big)^2}

      Intuitively, :math:`F_1(q)` measures how heavy‑tailed the weighted
      empirical distribution is along some direction :math:`v`.  If
      :math:`F_1(q)` exceeds a prescribed threshold :math:`\\kappa'^{2}` then the
      algorithm computes a quasi‑gradient :math:`g_1` that certifies this
      violation and performs a multiplicative update :math:`q \\leftarrow q * (1 - \\eta g_1)`.

    * **Noise functional** :math:`F_2(q)`:

      .. math::

          F_2(q) = \\sup_{v \\in \\mathbb{R}^d} \\frac{\\mathbb{E}_q[(Y - X^T\\theta(q))^2 (v^T X)^2]}{\\mathbb{E}_q[(v^T X)^2]}

      where :math:`\\theta(q)` is the weighted least‑squares estimator with
      respect to :math:`q`.  When :math:`F_2(q)` exceeds its threshold
      :math:`\\sigma'^{2}`, the algorithm uses the quasi‑gradient :math:`g_2` to
      down‑weight points contributing large residuals in directions of high
      leverage.

    The algorithm alternates between checking these two functionals.  If
    neither condition is violated, it terminates and returns the final
    weights and the weighted least‑squares estimator :math:`\\theta(q)`.  See
    algorithm 3 in the paper for details.

    Examples
    --------
    >>> import numpy as np
    >>> from robustgqg.regression import FilterRegression
    >>> rng = np.random.default_rng(0)
    >>> n, d = 200, 2
    >>> theta_true = np.array([1.0, -2.0])
    >>> X = rng.normal(size=(n, d))
    >>> y = X @ theta_true + rng.normal(scale=0.1, size=n)
    >>> fr = FilterRegression(eps=0.05, kappa=5.0, sigma=5.0)
    >>> result = fr.run(X, y)
    >>> result['status']
    'done'
    >>> np.allclose(result['theta'], theta_true, atol=0.2)
    True
    """

    def __init__(
        self,
        eps: float,
        kappa: float,
        sigma: float,
        solver: Optional[Literal["SCS", "MOSEK"]] = "SCS",
        max_iter: int = 500,
        tol_rel: float = 1e-5,
        verbose: bool = False,
        cap_kappa_prime_sq: Optional[float] = None,
        cap_sigma_prime_sq: Optional[float] = None,
    ) -> None:
        if not (0.0 < eps < 0.5):
            raise ValueError("eps must lie in (0, 0.5)")
        if kappa <= 0 or sigma <= 0:
            raise ValueError("kappa and sigma must be positive")
        self.eps = eps
        self.kappa = kappa
        self.sigma = sigma
        # Compute thresholds
        self.kappa_prime_sq, self.sigma_prime_sq = self._compute_thresholds(
            eps, kappa, sigma, cap_kappa_prime_sq, cap_sigma_prime_sq
        )
        # Hypercontractivity oracle
        self.oracle = HyperContractivityOracle(solver=solver, ridge=1e-8, verbose=verbose)
        self.max_iter = max_iter
        self.tol_rel = tol_rel  # currently unused but kept for API compatibility
        self.verbose = verbose

    def run(self, X: Array, y: Array, q0: Optional[Array] = None) -> FilterResult:
        """Execute the filter algorithm on the provided data.

        Parameters
        ----------
        X : ndarray of shape (n, d)
            Design matrix.
        y : ndarray of shape (n,)
            Response vector.
        q0 : Optional[ndarray], optional
            Initial weight vector.  If None, initializes to the uniform
            distribution over the ``n`` samples.

        Returns
        -------
        FilterResult
            A dataclass containing the final weights, coefficients,
            objective values and termination status.  The returned object
            supports both attribute access (``result.theta``) and
            dictionary‑style access (``result['theta']``).  See
            :class:`FilterResult` for a description of the available fields.

        Examples
        --------
        Basic usage on synthetic data:

        >>> import numpy as np
        >>> rng = np.random.default_rng(1)
        >>> X = rng.normal(size=(50, 3))
        >>> theta_star = np.array([0.5, -1.0, 2.0])
        >>> y = X @ theta_star + rng.normal(scale=0.05, size=50)
        >>> fr = FilterRegression(eps=0.1, kappa=5.0, sigma=5.0)
        >>> out = fr.run(X, y)
        >>> out['q'].shape, out['theta'].shape
        ((50,), (3,))
        """
        # Cast inputs to floating point and determine dimensions.  This
        # ensures all subsequent linear algebra is performed in double
        # precision.  ``n`` is the number of samples and ``d`` the
        # dimensionality of the covariates.
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        # Initialise the weight vector ``q`` on the probability simplex.
        # If ``q0`` is provided we normalize it to sum to one; otherwise
        # we start from the uniform distribution over the samples.  A
        # lower bound of ``1e-300`` prevents division by zero when the
        # weights are accidentally all zero.
        if q0 is None:
            q = np.full(n, 1.0 / n)
        else:
            q = np.asarray(q0, dtype=float)
            q = q / max(1e-300, q.sum())
        for it in range(self.max_iter):
            # Evaluate the hypercontractivity objective F1(q) and its quasi‑gradient g1.
            F1, _, g1 = self.oracle.run(X, q)
            if F1 >= self.kappa_prime_sq - 1e-12:
                # Hypercontractivity is violated: update using g1.
                g = g1
                stage = "hyper"
            else:
                # Otherwise evaluate the noise objective F2(q) and quasi‑gradient g2.
                F2, _, g2 = self._compute_noise_obj_and_grad(X, y, q)
                if F2 >= self.sigma_prime_sq - 1e-12:
                    g = g2
                    stage = "noise"
                else:
                    # Both moment conditions are satisfied; perform final fit and terminate.
                    theta = self._weighted_least_squares(X, y, q)
                    # Final objective values equal the current F1 and F2
                    F1_final = F1
                    F2_final = F2
                    return FilterResult(
                        q=q,
                        weights=q,
                        theta=theta,
                        coefficients=theta,
                        F1=F1_final,
                        F2=F2_final,
                        iters=it,
                        iterations=it,
                        status="done",
                    )
            # The quasi‑gradients are defined to be non‑negative; ensure this explicitly.
            g = np.maximum(0.0, g)
            # Determine the maximum gradient value.  If all entries are zero,
            # the update would leave q unchanged; in that case we terminate.
            g_max = float(np.max(g))
            if g_max <= 0.0:
                theta = self._weighted_least_squares(X, y, q)
                # Compute final objectives for consistency
                F1_final, _, _ = self.oracle.run(X, q)
                F2_final, _, _ = self._compute_noise_obj_and_grad(X, y, q)
                return FilterResult(
                    q=q,
                    weights=q,
                    theta=theta,
                    coefficients=theta,
                    F1=F1_final,
                    F2=F2_final,
                    iters=it,
                    iterations=it,
                    status="done",
                )
            # Use a step size eta = 1 / (g_max * (1 + tiny slack)).  This ensures
            # that the multiplicative update 1 - eta * g_i remains non‑negative
            # even in the degenerate case where all g_i are equal.
            g_max_slack = g_max * (1.0 + 1e-8) + 1e-18
            eta = 1.0 / g_max_slack
            # Multiplicative update: q_i <- q_i * (1 - eta * g_i).  Samples
            # with larger g_i are down‑weighted more strongly.
            q_tilde = q * (1.0 - eta * g)
            # Renormalize q to lie on the simplex
            q_sum = max(1e-300, q_tilde.sum())
            q = q_tilde / q_sum
            if self.verbose:
                print(f"[FilterRegression] iter={it} stage={stage} F1={F1:.6g}")
        # Maximum iterations reached
        # After reaching the maximum number of iterations, compute final objectives
        F1_final, _, _ = self.oracle.run(X, q)
        F2_final, _, _ = self._compute_noise_obj_and_grad(X, y, q)
        theta = self._weighted_least_squares(X, y, q)
        return FilterResult(
            q=q,
            weights=q,
            theta=theta,
            coefficients=theta,
            F1=F1_final,
            F2=F2_final,
            iters=self.max_iter,
            iterations=self.max_iter,
            status="max_iter_reached",
        )

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_thresholds(
        eps: float,
        kappa: float,
        sigma: float,
        cap_kappa_prime_sq: Optional[float],
        cap_sigma_prime_sq: Optional[float],
    ) -> Tuple[float, float]:
        """Compute the hypercontractivity and noise thresholds.

        The filter algorithm requires upper bounds on the hypercontractivity
        functional ``F_1(q)`` and the noise functional ``F_2(q)``.  These
        bounds depend on the contamination level ``eps`` as well as the
        assumed moment parameters ``kappa`` and ``sigma`` of the
        uncontaminated distribution.  Specifically,

        * ``eps`` (0 < eps < 0.5) is an upper bound on the fraction of
          adversarial samples.
        * ``kappa`` is such that the uncontaminated data satisfy a
          ``kappa``‑hypercontractive property: for any unit vector ``v``,

          ``E[(v^T X)^4] \\le kappa^2 (E[(v^T X)^2])^2``.
        * ``sigma`` bounds the noise standard deviation.

        Under these assumptions, Theorem 4.3 in the paper shows that
        restricting the filter to ``q`` with ``F_1(q) \\le kappa'^2`` and
        ``F_2(q) \\le sigma'^2`` suffices to isolate the uncontaminated
        distribution.  The thresholds are given by

        .. math::

            \\kappa'^2 = \frac{2 kappa^2}{1 - 2 kappa^2 \varepsilon},

        and

        .. math::

            \\sigma'^2 = \frac{4 \\sigma^2 \bigl(1 + 2 \\kappa' \\sqrt{\varepsilon(1-\varepsilon)}\bigr)}{
                (1 - 2 \varepsilon)^3 - 20 \\kappa'^3 \varepsilon (1-\varepsilon)}.

        Optional caps ``cap_kappa_prime_sq`` and ``cap_sigma_prime_sq`` may
        be provided to limit the thresholds in practice.

        Returns
        -------
        (float, float)
            The computed thresholds ``(kappa_prime_sq, sigma_prime_sq)``.
        """
        # Compute kappa'^2 using the closed‑form expression.  The term
        # ``denom1`` can approach zero when the contamination level is high,
        # so we protect against division by zero with ``max(1e-12, denom1)``.
        k2 = kappa**2
        denom1 = 1.0 - 2.0 * k2 * eps
        kappa_prime_sq = (2.0 * k2) / max(1e-12, denom1)
        # Compute sigma'^2.  We first derive kappa' and then evaluate the
        # denominator and numerator of the expression.  Again we guard
        # against small denominators for numerical stability.
        kprime = math.sqrt(max(0.0, kappa_prime_sq))
        denom2 = (1.0 - 2.0 * eps) ** 3 - 20.0 * (kprime**3) * eps * (1.0 - eps)
        sigma_prime_sq = (
            4.0 * sigma**2 * (1.0 + 2.0 * kprime * math.sqrt(eps * (1.0 - eps)))
        ) / max(1e-12, denom2)
        # Apply optional caps to avoid overly large thresholds
        if cap_kappa_prime_sq is not None:
            kappa_prime_sq = min(kappa_prime_sq, cap_kappa_prime_sq)
        if cap_sigma_prime_sq is not None:
            sigma_prime_sq = min(sigma_prime_sq, cap_sigma_prime_sq)
        return kappa_prime_sq, sigma_prime_sq

    @classmethod
    def _compute_noise_obj_and_grad(
        cls, X: Array, y: Array, q: Array
    ) -> Tuple[float, Array, Array]:
        """Evaluate the noise objective and its quasi‑gradient.

        This function computes the noise functional

        .. math::

            F_2(q) = \\sup_{v \neq 0} \frac{\\mathbb{E}_q[(y - x^T\theta)^2 (v^T x)^2]}{\\mathbb{E}_q[(v^T x)^2]},

        where ``\theta`` is the weighted least‑squares estimator with respect
        to the weight vector ``q``.  The maximizing direction ``v`` solves
        the generalized eigenproblem ``A v = \\lambda S_2 v`` with

        ``S_2 = E_q[X X^T]``, ``A = E_q[r^2 X X^T]`` and
        ``r = y - X \theta``.  The returned quasi‑gradient has entries
        ``g2[i] = r_i^2 (v^T x_i)^2``.

        Parameters
        ----------
        X : ndarray of shape ``(n, d)``
            Design matrix.
        y : ndarray of shape ``(n,)``
            Response vector.
        q : ndarray of shape ``(n,)``
            Non‑negative weights on the samples.

        Returns
        -------
        (float, ndarray, ndarray)
            A triple ``(F2_value, v, g2)`` consisting of the maximized value
            of ``F_2``, the direction ``v`` (unit normalized with respect
            to ``E_q[(v^T X)^2]``) and the non‑negative quasi‑gradient ``g2``.
        """
        # normalize q to ensure it sums to one
        q = q / max(1e-300, q.sum())
        n, d = X.shape
        # Solve for the weighted least‑squares parameter theta(q)
        theta = cls._weighted_least_squares(X, y, q)
        # Compute residuals r = y - X theta and squared residuals
        resid = y - X @ theta
        r2 = resid**2
        # Form S2 = E_q[X X^T]
        S2 = (X.T * q) @ X
        # Form A = E_q[r^2 X X^T] where r^2 weights each sample
        A = (X.T * (q * r2)) @ X
        # Symmetrize S2 and A to avoid numerical asymmetry
        S2 = (S2 + S2.T) / 2.0
        A = (A + A.T) / 2.0
        # Compute eigen-decomposition of S2 to whiten it
        lam_S2, U = eigh(S2)
        inv_sqrt = U @ np.diag(1.0 / np.sqrt(np.maximum(lam_S2, 1e-12))) @ U.T
        # Reduce the generalised eigenproblem to a standard one: M = S2^{-1/2} A S2^{-1/2}
        M = inv_sqrt @ A @ inv_sqrt
        M = (M + M.T) / 2.0
        lam_M, V_M = eigh(M)
        idx = int(np.argmax(lam_M))
        F2_value = float(lam_M[idx])
        # Obtain the direction in the original coordinates
        v = inv_sqrt @ V_M[:, idx]
        # normalize v so that E_q[(v^T x)^2] = 1.  This ensures the gradient
        # scale matches the objective value.
        proj2 = (X @ v) ** 2
        denom = float((proj2 * q).sum())
        if denom > 1e-12:
            v = v / math.sqrt(denom)
        # Compute quasi‑gradient: g2_i = r_i^2 (v^T x_i)^2
        g2 = r2 * (X @ v) ** 2
        g2 = np.maximum(0.0, g2)
        return F2_value, v, g2

    @staticmethod
    def _weighted_least_squares(X: Array, y: Array, q: Array) -> Array:
        """Solve a weighted least–squares problem.

        For a given weight vector ``q`` summing to one, this function
        computes the minimizer

        .. math::

            \theta(q) = \arg\\min_{\theta} \\sum_{i=1}^n q_i (y_i - x_i^T \theta)^2,

        where ``x_i`` is the ``i``‑th row of ``X``.  Equivalently, if
        ``D = diag(q)`` then ``\theta(q)`` solves the normal equation
        ``X^T D X theta = X^T D y``.  We implement this by scaling the
        design matrix and response vector by ``sqrt(q)`` and calling
        :func:`numpy.linalg.lstsq`.

        Parameters
        ----------
        X : ndarray of shape ``(n, d)``
            Design matrix.
        y : ndarray of shape ``(n,)``
            Response vector.
        q : ndarray of shape ``(n,)``
            Non‑negative weights summing to one.

        Returns
        -------
        ndarray of shape ``(d,)``
            The weighted least–squares solution.
        """
        # normalize weights in case they do not sum to one exactly
        q = q / max(1e-300, q.sum())
        # Form the square‑root scaled design matrix and response vector
        sqrt_q = np.sqrt(q)
        Xw = X * sqrt_q[:, None]
        yw = y * sqrt_q
        # Solve the unweighted least‑squares problem on the scaled data
        theta, *_ = lstsq(Xw, yw, rcond=None)
        return theta
