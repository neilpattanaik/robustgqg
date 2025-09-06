# tests/test_mean_filter.py
import numpy as np
from numpy.linalg import norm
import pytest

from robustgqg.mean import FilterMean
from .utils import make_contaminated_mean_data

SEED = 42
np.random.seed(SEED)

def run_algo(X, eps, **kw):
    est = FilterMean(X, eps, max_iter=800, eta=1.0, verbose=False, **kw)
    return est.run()

def test_filter_beats_naive_mean():
    X, mu_true, eps = make_contaminated_mean_data()
    out = run_algo(X, eps)
    err = norm(out["mu"] - mu_true)
    err_naive = norm(X.mean(axis=0) - mu_true)
    assert err <= 0.7 * err_naive + 1e-9  # slightly looser than Explicit algo

def test_q_is_simplex_no_cap():
    X, _, eps = make_contaminated_mean_data()
    out = run_algo(X, eps)
    q = out["q"]
    assert np.all(q >= -1e-12)
    assert abs(q.sum() - 1.0) <= 1e-8
    # Unlike Explicit algo, there is no per-item cap to check.

@pytest.mark.parametrize("eps", [0.05, 0.1, 0.2, 0.3])
def test_filter_various_eps(eps):
    X, mu_true, _ = make_contaminated_mean_data(eps=eps)
    out = run_algo(X, eps)
    err = norm(out["mu"] - mu_true)
    err_naive = norm(X.mean(axis=0) - mu_true)
    assert err <= 0.8 * err_naive + 1e-9
