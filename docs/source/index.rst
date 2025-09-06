robustgqg Documentation
=======================

robustgqg provides robust estimators based on the generalized
quasi‑gradient filter of Zhu, Jiao & Steinhardt (2020). The regression
module combines hypercontractivity (degree‑4) and bounded noise checks;
the mean module implements filter and explicit low‑regret variants for
bounded covariance mean estimation.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Contents

   api

Getting Started
---------------

Install the package in editable mode and build docs:

.. code-block:: bash

   pip install -e '.[dev]'
   (cd docs && make html)

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
