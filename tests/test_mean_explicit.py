import numpy as np
import pytest
from numpy.linalg import norm
from robustgqg.mean import MeanOptions, ExplicitLowRegretMean
from robustgqg.mean.projections import project_capped_simplex_kl

SEED = 42
np.random.seed(SEED)


def make_contaminated_mean_data(n=600, d=5, eps=0.12, shift=12.0, scale=6.0):
    rng = np.random.default_rng()
    k = int((1 - eps) * n)
    mu = rng.normal(size=d)
    X_good = rng.normal(loc=mu, scale=1.0, size=(k, d))
    X_bad = rng.normal(loc=mu + shift, scale=scale, size=(n - k, d))
    X = np.vstack([X_good, X_bad])
    rng.shuffle(X, axis=0)
    return X, mu, eps


def run_algo(X, eps, **kw):
    opts = MeanOptions(eps=eps, max_iter=800, eta=1.0, verbose=False, **kw)
    est = ExplicitLowRegretMean(X, opts)
    return est.run()


def test_explicit_robust_beats_naive():
    X, mu_true, eps = make_contaminated_mean_data()
    out = run_algo(X, eps)
    err = norm(out["mu"] - mu_true)
    err_naive = norm(X.mean(axis=0) - mu_true)
    assert err <= 0.6 * err_naive + 1e-9


def test_weights_are_valid_simplex_and_capped():
    X, _, eps = make_contaminated_mean_data()
    out = run_algo(X, eps)
    q = out["q"]
    n = len(q)
    cap = 1.0 / ((1.0 - eps) * n)
    assert np.all(q >= -1e-12)
    assert abs(q.sum() - 1.0) <= 1e-8
    assert np.all(q <= cap + 1e-10)


def test_projection_matches_paper_structure():
    rng = np.random.default_rng()
    z = rng.random(100) + 1e-6
    eps = 0.1
    n = z.size
    cap = 1.0 / ((1.0 - eps) * n)
    q = project_capped_simplex_kl(z, cap=cap, total=1.0)
    assert abs(q.sum() - 1.0) <= 1e-10
    assert np.all(q >= -1e-12)
    assert np.all(q <= cap + 1e-12)


def test_monotone_certificate_decrease_in_practice():
    # Not guaranteed strictly monotone in theory, but in practice we expect decrease over iterations
    X, _, eps = make_contaminated_mean_data()
    opts = MeanOptions(eps=eps, max_iter=60, eta=1.0, verbose=False)
    est = ExplicitLowRegretMean(X, opts)
    # Instrument the internal loop by stepping manually
    from robustgqg.mean.utils import F_cov_spectral, weights_uniform
    q = weights_uniform(X.shape[0])
    Fs = []
    for _ in range(opts.max_iter):
        F, mu, v = F_cov_spectral(X, q)
        Fs.append(F)
        Xc_v = (X - mu) @ v
        g = Xc_v ** 2
        Bk = max(1e-12, float(np.max(g)))
        eta_k = opts.eta / (2.0 * Bk)
        q_tilde = q * (1.0 - eta_k * g)
        q = project_capped_simplex_kl(np.maximum(q_tilde, 0.0), cap=1.0/((1.0-opts.eps)*X.shape[0]), total=1.0)
    # Check that final F <= initial F (allow tiny tolerance)
    assert Fs[-1] <= Fs[0] + 1e-7


@pytest.mark.parametrize("eps", [0.05, 0.1, 0.2, 0.3])
def test_various_eps_values(eps):
    X, mu_true, _ = make_contaminated_mean_data(eps=eps)
    out = run_algo(X, eps)
    err = norm(out["mu"] - mu_true)
    err_naive = norm(X.mean(axis=0) - mu_true)
    assert err <= 0.75 * err_naive + 1e-9
