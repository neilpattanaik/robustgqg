import numpy as np

def make_contaminated_mean_data(n=600, d=5, eps=0.12, shift=12.0, scale=6.0):
    rng = np.random.default_rng()
    k = int((1 - eps) * n)
    mu = rng.normal(size=d)
    X_good = rng.normal(loc=mu, scale=1.0, size=(k, d))
    X_bad = rng.normal(loc=mu + shift, scale=scale, size=(n - k, d))
    X = np.vstack([X_good, X_bad])
    rng.shuffle(X, axis=0)
    return X, mu, eps

