Bayesian Hierarchical Beta Diversity Modelling
==============================================

**Demo release** — full package coming in 2026.

Python code for modelling beta diversity in the style of 
`GDM <https://onlinelibrary.wiley.com/doi/full/10.1111/geb.13459>`_
using its hierarchical Bayesian extension,
`spGDMM <https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.14259>`_.

Demo
----

``demo.ipynb`` fits spGDMM to plankton community data
(temperature, salinity, vorticity → 5 plankton groups)
from the SINMOD ocean model, using **50 training sites** and **100 test sites**
on day 105 (15 April 2019).

It covers model fitting, convergence diagnostics,
predictive scoring (CRPS), community mapping,
and I-spline response curves.

The pre-processed dataset (``demo_data/day105_publication.npz``) is included —
no data pipeline dependency is needed to run the notebook.

**Setup** (from repo root):

.. code-block:: bash

   mamba env create -f environment.yml
   mamba activate spgdmm-demo
   jupyter lab demo.ipynb


Based on
--------

Original model implementation in R
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

|spgdmm_doi| *Generative spatial generalized dissimilarity mixed modelling (spGDMM):
An enhanced approach to modelling beta diversity.* White, P.A., Frye, H.A., Slingsby, J.A., Silander, J.A. & Gelfand, A.E. (2024). *Methods in Ecology and Evolution*, 15(1), 214–226.

Bayesian inference framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

|DOIpaper| *PyMC: A Modern and Comprehensive Probabilistic Programming Framework in Python*, Abril-Pla O, Andreani V, Carroll C, Dong L, Fonnesbeck CJ, Kochurov M, Kumar R, Lao J, Luhmann CC, Martin OA, Osthege M, Vieira R, Wiecki T, Zinkov R. (2023)


Contact
-------

Anonymous


.. |spgdmm_doi| image:: https://img.shields.io/badge/DOI-10.1111%2F2041--210X.14259-blue.svg
   :target: https://doi.org/10.1111/2041-210X.14259

.. |DOIpaper| image:: https://img.shields.io/badge/DOI-10.7717%2Fpeerj--cs.1516-blue.svg
   :target: https://doi.org/10.7717/peerj-cs.1516