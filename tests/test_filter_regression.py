import numpy as np
from numpy.linalg import eigh

from robustgqg.regression import FilterRegression


def test_weight_initialization_uniform():
    """If F1 and F2 constraints are easily satisfied, the algorithm should return immediately
    with a uniform weight vector.
    """
    rng = np.random.default_rng(0)
    n = 20
    # One‑dimensional data with small kurtosis and noise
    X = rng.normal(0.0, 1.0, size=(n, 1))
    y = 3.0 * X[:, 0] + rng.normal(0.0, 0.1, size=n)
    # Choose large kappa and sigma so thresholds are high and no update is needed
    fr = FilterRegression(eps=0.1, kappa=10.0, sigma=10.0, verbose=False)
    result = fr.run(X, y)
    q = result['q']
    # It should terminate in zero iterations (or at most one) without updates
    assert result['status'] == 'done'
    assert result['iters'] == 0
    # q should be approximately uniform
    assert np.allclose(q, np.full(n, 1.0 / n), atol=1e-6)
    # Final F1 and F2 should be below thresholds
    assert result['F1'] < fr.kappa_prime_sq + 1e-8
    assert result['F2'] < fr.sigma_prime_sq + 1e-8


def test_quasi_gradients_nonnegative():
    """The hypercontractivity and noise quasi‑gradients should be non‑negative."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(10, 2))
    y = rng.normal(size=10)
    fr = FilterRegression(eps=0.1, kappa=5.0, sigma=5.0)
    # compute g1
    F1, _, g1 = fr.oracle.run(X, np.full(10, 0.1))
    assert np.all(g1 >= -1e-12)
    # compute g2
    F2, _, g2 = fr._compute_noise_obj_and_grad(X, y, np.full(10, 0.1))
    assert np.all(g2 >= -1e-12)


def test_noise_objective_matches_manual():
    """Verify that _compute_noise_obj_and_grad returns the correct generalized eigenvalue."""
    rng = np.random.default_rng(2)
    n, d = 6, 3
    X = rng.normal(size=(n, d))
    # True theta
    theta_true = rng.normal(size=d)
    y = X @ theta_true + rng.normal(scale=0.05, size=n)
    q = np.full(n, 1.0 / n)
    fr = FilterRegression(eps=0.1, kappa=5.0, sigma=5.0)
    F2, v, g2 = fr._compute_noise_obj_and_grad(X, y, q)
    # Manual computation of F2: whiten the covariance and compute the top eigenvalue
    theta = fr._weighted_least_squares(X, y, q)
    resid = y - X @ theta
    r2 = resid ** 2
    S2 = (X.T * q) @ X
    A = (X.T * (q * r2)) @ X
    # Symmetrise
    S2 = (S2 + S2.T) / 2.0
    A = (A + A.T) / 2.0
    lam_S2, U = eigh(S2)
    inv_sqrt = U @ np.diag(1.0 / np.sqrt(np.maximum(lam_S2, 1e-12))) @ U.T
    M = inv_sqrt @ A @ inv_sqrt
    M = (M + M.T) / 2.0
    lam_manual, _ = eigh(M)
    lam_max = float(lam_manual[-1])
    assert np.isclose(F2, lam_max, atol=1e-6)
    # g2 has correct length
    assert g2.shape == (n,)


def test_outlier_downweighted():
    """An extreme outlier should receive a smaller weight after filtering."""
    rng = np.random.default_rng(3)
    n = 30
    # Inliers
    X_inliers = rng.normal(0.0, 1.0, size=(n - 1, 1))
    y_inliers = 2.0 * X_inliers[:, 0] + rng.normal(0.0, 0.1, size=n - 1)
    # Single outlier far away
    X_outlier = np.array([[10.0]])
    y_outlier = np.array([0.0])
    X = np.vstack([X_inliers, X_outlier])
    y = np.concatenate([y_inliers, y_outlier])
    # Choose moderate kappa so that the hypercontractivity condition is violated
    fr = FilterRegression(eps=0.1, kappa=1.0, sigma=10.0)
    result = fr.run(X, y)
    q = result['q']
    # The outlier weight should be smaller than its initial 1/n weight
    assert q[-1] < 1.0 / n * 0.5
    # All weights non‑negative and sum to one
    assert np.all(q >= -1e-12)
    assert np.isclose(q.sum(), 1.0, atol=1e-6)


def test_regression_coefficients_recovered():
    """For a clean linear dataset, the recovered coefficients should be close to the true coefficients."""
    rng = np.random.default_rng(4)
    n, d = 100, 2
    theta_true = np.array([1.5, -2.0])
    X = rng.normal(size=(n, d))
    y = X @ theta_true + rng.normal(scale=0.05, size=n)
    fr = FilterRegression(eps=0.05, kappa=5.0, sigma=5.0)
    result = fr.run(X, y)
    theta_hat = result['theta']
    # Should terminate quickly
    assert result['status'] == 'done'
    # Recovered coefficients close to true
    assert np.allclose(theta_hat, theta_true, atol=0.2)


def test_end_to_end_vs_ols():
    """Compare the filter regression against ordinary least squares on clean and contaminated data."""
    rng = np.random.default_rng(5)
    n_clean = 200
    d = 2
    theta_true = np.array([1.0, -3.0])
    # Clean data
    X_clean = rng.normal(size=(n_clean, d))
    y_clean = X_clean @ theta_true + rng.normal(scale=0.05, size=n_clean)
    # Ordinary least squares on clean data
    theta_ols_clean, *_ = np.linalg.lstsq(X_clean, y_clean, rcond=None)
    # Filter regression on clean data with large kappa and sigma so no points are dropped
    fr_clean = FilterRegression(eps=0.05, kappa=10.0, sigma=10.0)
    result_clean = fr_clean.run(X_clean, y_clean)
    theta_filter_clean = result_clean['theta']
    # On clean data with large kappa and sigma, filter regression should be equal to OLS and close to the true coefficients
    assert np.allclose(theta_filter_clean, theta_ols_clean, atol=0)
    assert np.allclose(theta_filter_clean, theta_true, atol=1e-2)
    # Now contaminate the data with outliers
    n_outliers = 20
    X_outliers = rng.normal(loc=10.0, scale=1.0, size=(n_outliers, d))
    # Assign adversarial responses that do not follow the linear model
    y_outliers = rng.normal(loc=50.0, scale=1.0, size=n_outliers)
    # Combine data
    X_combined = np.vstack([X_clean, X_outliers])
    y_combined = np.concatenate([y_clean, y_outliers])
    # OLS on contaminated data
    theta_ols_contaminated, *_ = np.linalg.lstsq(X_combined, y_combined, rcond=None)
    # Filter regression on contaminated data with moderate kappa to downweight outliers
    fr_contaminated = FilterRegression(eps=n_outliers / (n_clean + n_outliers), kappa=1.0, sigma=10.0)
    result_contaminated = fr_contaminated.run(X_combined, y_combined)
    theta_filter_contaminated = result_contaminated['theta']
    # The filter estimator should be closer to the true parameters than the OLS estimator on contaminated data
    err_ols = np.linalg.norm(theta_ols_contaminated - theta_true)
    err_filter = np.linalg.norm(theta_filter_contaminated - theta_true)
    assert err_filter <= err_ols + 1e-6
