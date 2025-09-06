from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np
from numpy.linalg import eigh

from robustgqg.types import Array


class HyperContractivityOracle:
    """Sum-of-Squares oracle for the hypercontractivity functional.

    This class evaluates the degree‑4 hypercontractivity objective

    .. math::

        F_1(q) = \\sup_{v \\in \\mathbb{R}^d}
          \\frac{\\mathbb{E}_q[(v^\\top X)^4]}
               {\\big(\\mathbb{E}_q[(v^\\top X)^2]\\big)^2},

    using a degree‑2/degree‑4 Sum‑of‑Squares (SoS) relaxation on whitened
    data.  Whitening with respect to :math:`E_q[XX^\\top]` normalizes the
    denominator to one, so the relaxation maximises the quartic moment of
    the whitened samples.  The resulting SDP yields an upper bound on
    :math:`F_1(q)` together with a feasible pseudo‑moment vector :math:`y`.  From
    :math:`y` we extract a non‑negative quasi‑gradient over samples that the
    filter algorithm can use to down‑weight heavy‑tailed directions.

    Notes
    -----
    - Requires :mod:`cvxpy` and a compatible SDP solver (e.g., ``SCS`` or
      ``MOSEK``).

    Examples
    --------
    >>> import numpy as np
    >>> from robustgqg.regression.hypercontractivity import HyperContractivityOracle
    >>> X = np.random.default_rng(0).normal(size=(20, 3))
    >>> q = np.full(20, 1/20)
    >>> oracle = HyperContractivityOracle(solver='SCS')
    >>> F1, v, g = oracle.run(X, q)
    >>> F1 > 0 and g.shape == (20,)
    True

    The direction :math:`v` is a witness in the original coordinates; it is
    provided for diagnostics and is not used by the filter update itself.

    Whitening
    ---------

    Let :math:`B(q) = E_q[XX^\\top]` and let :math:`B^{-1/2}` denote its inverse
    square root. Define whitened samples :math:`z = B^{-1/2} x` (implemented
    via :math:`Z = X B^{-1/2}`). Then

    .. math::

        F_1(q)
        = \\sup_{v \\in \\mathbb{R}^d}
          \\frac{E_q[(v^\\top X)^4]}{(E_q[(v^\\top X)^2])^2}
        = \\sup_{\\|u\\|_2 = 1} E_q[(u^\\top z)^4],

    where the change of variables :math:`u = B^{1/2} v` uses
    :math:`E_q[(u^\\top z)^2] = \\|u\\|_2^2`. Writing the (weighted) empirical
    fourth-moment tensor of the whitened samples as

    .. math::

        T_{ijkl} = E_q[z_i z_j z_k z_l],

    the objective becomes :math:`\sum_{i,j,k,l} T_{ijkl} u_i u_j u_k u_l` with
    the constraint :math:`\|u\|_2 = 1`.

    Degree‑4 SoS Relaxation
    -----------------------

    Introduce pseudo‑moments :math:`y_\\alpha` indexed by multi‑indices
    :math:`\\alpha \\in \\mathbb{N}^d` with :math:`|\\alpha| \\le 4` and enforce:

    - Moment matrix PSD: :math:`M(y) \\succeq 0` where
      :math:`M_{\\alpha,\\beta} = y_{\\alpha+\\beta}` for :math:`|\\alpha|,|\\beta| \\le 2`.
    - Normalisation: :math:`y_{\\boldsymbol{0}} = 1`.
    - Sphere constraints (localisers):
      :math:`\\sum_{i=1}^d y_{\\alpha + 2 e_i} = y_{\\alpha}` for all :math:`|\\alpha| \\le 2`.

    Maximise the linear objective

    .. math::

        \\sum_{i,j,k,l} T_{ijkl}\, y_{e_i + e_j + e_k + e_l},

    which upper‑bounds :math:`\\sup_{\|u\|=1} E_q[(u^\\top z)^4]`. From the
    optimal pseudo‑moments :math:`y` we define a non‑negative quartic
    polynomial :math:`p(z) = \\sum_{|\\alpha|=4} c_{\\alpha} z^{\\alpha}` whose
    coefficients :math:`c_{\\alpha}` are the appropriately symmetrised entries
    of :math:`y`; the sample‑wise quasi‑gradient is :math:`g_1[i] = p(z_i)`.
    """

    def __init__(
        self,
        solver: Optional[Literal["SCS", "MOSEK"]] = "SCS",
        ridge: float = 1e-8,
        verbose: bool = False,
    ) -> None:
        self.solver = solver
        self.ridge = ridge
        self.verbose = verbose

    def run(self, X: Array, q: Array) -> Tuple[float, Optional[Array], Array]:
        """Compute :math:`F_1(q)` and a sample‑wise quasi‑gradient via SoS.

        Parameters
        ----------
        X : ndarray of shape ``(n, d)``
            Design matrix; rows are samples.
        q : ndarray of shape ``(n,)``
            Non‑negative weights summing to one.

        Returns
        -------
        (float, ndarray | None, ndarray)
            ``(value, direction, gradient)`` where ``value`` is the SoS
            upper bound on :math:`F_1(q)`, ``direction`` is a witness vector in
            :math:`R^d` (unit‑normalized; provided for diagnostics), and
            ``gradient`` is a non‑negative vector of length :math:`n` whose
            entries constitute the quasi‑gradient.

        Raises
        ------
        RuntimeError
            If :mod:`cvxpy` is not installed or the SDP solver does not
            return an optimal solution.
        """
        return self._hypercontractivity_oracle_sos_whitened(X, q)

    # ----------------------------------------------------------------------
    # Utility functions
    # ----------------------------------------------------------------------
    @staticmethod
    def _eigh_psd(A: Array, eps: float = 1e-18) -> Tuple[Array, Array]:
        """Eigen‑decomposition with PSD clipping.

        Symmetrises :math:`A` as :math:`(A + A^T)/2` and computes its eigenvalues
        and eigenvectors. Eigenvalues are then lower‑bounded by ``eps`` to
        ensure positive semi‑definiteness when building inverse square
        roots.

        Parameters
        ----------
        A : ndarray of shape ``(d, d)``
            Symmetric (or nearly symmetric) matrix.
        eps : float, default ``1e-18``
            Lower bound for eigenvalues.

        Returns
        -------
        (ndarray, ndarray)
            The clipped eigenvalues :math:`w` and eigenvectors :math:`V` such that
            :math:`A ≈ V @ diag(w) @ V^T`.
        """
        A = (A + A.T) / 2.0
        w, V = eigh(A)
        w = np.maximum(w, eps)
        return w, V

    @staticmethod
    def _whitener_Binvhalf(X: Array, q: Array, ridge: float = 1e-8) -> Array:
        """Compute the weighted covariance whitener :math:`B^{-1/2}`.

        Forms :math:`B = E_q[XX^T] = X^T diag(q) X` and returns its inverse
        square root. A small ``ridge`` multiple of the identity is added
        to stabilise ill‑conditioned problems.

        Parameters
        ----------
        X : ndarray of shape ``(n, d)``
            Data matrix (rows are samples).
        q : ndarray of shape ``(n,)``
            Non‑negative weights (will be normalized to sum to one).
        ridge : float, default ``1e-8``
            Tikhonov regularisation added to ``B`` before inversion.

        Returns
        -------
        ndarray of shape ``(d, d)``
            The symmetric matrix :math:`B^{-1/2}`.

        Notes
        -----
        If :math:`Z = X @ B^{-1/2}``, then :math:`E_q[ZZ^T] ≈ I_d` (whitening).
        """
        q = q / max(1e-300, float(q.sum()))
        B = (X.T * q) @ X
        w, V = HyperContractivityOracle._eigh_psd(B + ridge * np.eye(B.shape[0]), eps=1e-18)
        return (V / np.sqrt(w)) @ V.T

    @staticmethod
    def _coeff_vec_for_square(z: Array) -> Array:
        """Vectorise the symmetric square :math:`z z^T` with 2× off‑diagonals.

        Returns the unique degree‑2 monomials arranged as upper‑triangular
        entries of :math:`z z^T` with off‑diagonal terms scaled by :math:`2`:
        :math:`[z_1^2, 2 z_1 z_2, \\ldots, 2 z_1 z_d, z_2^2, 2 z_2 z_3, \\ldots, z_d^2]`.

        Parameters
        ----------
        z : ndarray of shape ``(d,)``
            Input vector.

        Returns
        -------
        ndarray of shape ``(d(d+1)/2,)``
            The symmetric monomial vectorisation of ``z``.
        """
        d = z.size
        vec = []
        for i in range(d):
            for j in range(i, d):
                if i == j:
                    vec.append(z[i] * z[i])
                else:
                    vec.append(2.0 * z[i] * z[j])
        return np.asarray(vec, dtype=float)

    def _hypercontractivity_oracle_sos_whitened(
        self, X: Array, q: Array
    ) -> Tuple[float, Optional[Array], Array]:
        try:
            import cvxpy as cp
        except Exception as e:
            raise RuntimeError("cvxpy is required for SOS, but is not installed") from e

        q = q / max(1e-300, float(q.sum()))
        n, d = X.shape
        Binvhalf = self._whitener_Binvhalf(X, q, ridge=self.ridge)
        Z = X @ Binvhalf
        # Build the empirical fourth moment tensor
        T4 = np.zeros((d, d, d, d), dtype=float)
        for i in range(n):
            zi = Z[i]
            wi = q[i]
            zz = np.outer(zi, zi)
            T4 += wi * np.einsum("ij,kl->ijkl", zz, zz)

        # Generate multi‑indices for monomials up to degree 2 and exactly degree 4
        def _multi_indices_upto_deg(d: int, deg: int):
            out = []
            cur = [0] * d

            def rec(pos: int, remaining: int):
                if pos == d:
                    out.append(tuple(cur))
                    return
                for k in range(remaining + 1):
                    cur[pos] = k
                    rec(pos + 1, remaining - k)

            rec(0, deg)
            return out

        def _alpha_e(d: int, i: int) -> Tuple[int, ...]:
            return tuple(1 if j == i else 0 for j in range(d))

        def _add_multi(a: Tuple[int, ...], b: Tuple[int, ...]) -> Tuple[int, ...]:
            return tuple(ai + bi for ai, bi in zip(a, b))

        mon_deg2 = _multi_indices_upto_deg(d, 2)
        all_deg4 = _multi_indices_upto_deg(d, 4)
        alpha_to_idx = {alpha: idx for idx, alpha in enumerate(all_deg4)}
        m4 = len(all_deg4)
        y = cp.Variable(m4)

        def y_of(alpha: Tuple[int, ...]) -> cp.Expression:
            return y[alpha_to_idx[alpha]]

        # Moment matrix M for monomials up to degree 2
        M_entries = [[y_of(_add_multi(a, b)) for b in mon_deg2] for a in mon_deg2]
        M = cp.bmat(M_entries)
        cons = [M >> 0]
        zero = tuple([0] * d)
        cons.append(y_of(zero) == 1.0)
        # Sphere localizers: sum_i y_{alpha + 2 e_i} = y_alpha for |alpha| <= 2
        for alpha in _multi_indices_upto_deg(d, 2):
            cons.append(
                cp.sum(
                    cp.hstack(
                        [
                            y_of(_add_multi(alpha, _add_multi(_alpha_e(d, i), _alpha_e(d, i))))
                            for i in range(d)
                        ]
                    )
                )
                == y_of(alpha)
            )
        # Objective: E~[ E_q[(v^T Z)^4] ] = sum_{ijkl} T4_{ijkl} y_{e_i+e_j+e_k+e_l}
        a = np.zeros(len(alpha_to_idx), dtype=float)

        # fill a by accumulating T4 onto the y[alpha] that matches e_i+e_j+e_k+e_l
        for i in range(d):
            ei = _alpha_e(d, i)
            for j in range(d):
                ej = _alpha_e(d, j)
                eij = _add_multi(ei, ej)
                for k in range(d):
                    ek = _alpha_e(d, k)
                    eijk = _add_multi(eij, ek)
                    for l in range(d):  # noqa: E741
                        el = _alpha_e(d, l)
                        e4 = _add_multi(eijk, el)
                        a[alpha_to_idx[e4]] += T4[i, j, k, l]

        # objective becomes a simple linear form:
        obj = a @ y
        prob = cp.Problem(cp.Maximize(obj), cons)
        solve_kwargs = {"solver": self.solver, "verbose": self.verbose}
        if self.solver == "SCS":
            solve_kwargs.update(dict(max_iters=20000, eps=1e-6))
        prob.solve(**solve_kwargs)
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"SoS solver failed with status: {prob.status}")
        F1_value = float(prob.value)
        y_val = y.value
        # Compute g1 by evaluating the quartic polynomial defined by y_val.
        # For a multi‑index alpha of total degree 4, the contribution to
        # g1 is multiplicity(alpha) * y_val[alpha] * prod(Z_i^alpha) over
        # rows i.  multiplicity(alpha) = 4! / prod_j alpha_j! counts how
        # many permutations of indices (i,j,k,l) sum to alpha.
        from math import factorial

        g1 = np.zeros(n)
        # Precompute multiplicities for each multi‑index
        multiplicities = {}
        for alpha, _idx_alpha in alpha_to_idx.items():
            # Compute multiplicity if not already done
            if alpha not in multiplicities:
                m = 24  # 4! = 24
                for a in alpha:
                    m //= factorial(a)
                multiplicities[alpha] = m
        # Evaluate contributions for each multi‑index
        for alpha, _idx_alpha in alpha_to_idx.items():
            mult = multiplicities[alpha]
            coeff = y_val[_idx_alpha] * mult
            if coeff == 0.0:
                continue
            # Compute product of Z^alpha along columns: prod_j Z[:,j]**alpha_j
            term = np.prod(Z ** np.array(alpha), axis=1)
            g1 += coeff * term
        # Clip small negatives due to numerical error
        g1 = np.maximum(0.0, g1)
        # Extract witness direction for information (not used by FilterRegression)
        M11 = np.array(
            [
                [y_val[alpha_to_idx[_add_multi(_alpha_e(d, i), _alpha_e(d, j))]] for j in range(d)]
                for i in range(d)
            ]
        )
        w_eig, V_eig = eigh((M11 + M11.T) / 2.0)
        v_white = V_eig[:, np.argmax(w_eig)]
        v_orig = Binvhalf @ v_white
        v_dir = v_orig / (np.linalg.norm(v_orig) + 1e-12)
        return F1_value, v_dir, g1
