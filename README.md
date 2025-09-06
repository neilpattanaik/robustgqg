robustgqg — Robust estimation via generalized quasi‑gradients
=============================================================

[![Docs build](https://github.com/neilpattanaik/robustgqg/actions/workflows/docs.yml/badge.svg)](https://github.com/neilpattanaik/robustgqg/actions/workflows/docs.yml)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://neilpattanaik.github.io/robustgqg/)

robustgqg implements the generalized quasi‑gradient filter algorithms from

  Zhu, Jiao, Steinhardt (2020). Robust estimation via generalized quasi‑gradients.
  https://arxiv.org/abs/2005.14073

It provides:

- Robust linear regression via the filter algorithm (hypercontractivity + noise checks)
- Robust mean estimation (filter and explicit low‑regret variants)
- A Sum‑of‑Squares (SoS) hypercontractivity oracle (whitened) for regression


Installation
------------

This repo is under active development. For local development:
```
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e '.[dev]'

# Optional (SoS oracle for hypercontractivity):
pip install -e '.[sos]'
```

Notes
- The hypercontractivity oracle uses cvxpy; the default solver in examples is SCS.
- You can also install MOSEK and set `solver='MOSEK'` if licensed.


Quickstart
----------

Robust regression
```
import numpy as np
from robustgqg.regression import FilterRegression

rng = np.random.default_rng(0)
n, d = 200, 2
theta_true = np.array([1.0, -2.0])
X = rng.normal(size=(n, d))
y = X @ theta_true + rng.normal(scale=0.1, size=n)

# eps: outlier fraction upper bound; kappa, sigma: clean moment parameters
fr = FilterRegression(eps=0.05, kappa=5.0, sigma=5.0, solver='SCS')
out = fr.run(X, y)
print(out['status'], out['F1'], out['F2'])
print(out['theta'])
```

Robust mean estimation
```
import numpy as np
from robustgqg.mean import FilterMean, ExplicitLowRegretMean

X = np.random.default_rng(1).normal(size=(100, 3))
mean1 = FilterMean(X, eps=0.1).run()
mean2 = ExplicitLowRegretMean(X, eps=0.1).run()
print(mean1['mu'], mean2['mu'])
```


Documentation
-------------

Live docs: https://neilpattanaik.github.io/robustgqg/

See docstrings in `robustgqg/` and the demo notebook `robustgqg_demo.ipynb`.
Sphinx sources are under `docs/`.


Development
-----------

Install dev deps and run tests:

```
pip install -e '.[dev]'
pytest
```

Linting and formatting:

```
ruff check .        # lint
ruff format .       # or: black .
```
