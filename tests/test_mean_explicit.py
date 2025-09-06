import numpy as np
import pytest
from numpy.linalg import norm
from robustgqg.mean import ExplicitLowRegretMean
from robustgqg.mean.explicit import project_capped_simplex_kl
from .utils import make_contaminated_mean_data

SEED = 42
np.random.seed(SEED)

def run_algo(X, eps, **kw):
    est = ExplicitLowRegretMean(X, eps, max_iter=800, eta=1.0, verbose=False, **kw)
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


@pytest.mark.parametrize("eps", [0.05, 0.1, 0.2, 0.3])
def test_various_eps_values(eps):
    X, mu_true, _ = make_contaminated_mean_data(eps=eps)
    out = run_algo(X, eps)
    err = norm(out["mu"] - mu_true)
    err_naive = norm(X.mean(axis=0) - mu_true)
    assert err <= 0.75 * err_naive + 1e-9
