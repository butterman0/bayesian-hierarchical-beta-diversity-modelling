Bayesian Hierarchical Beta Diversity Modelling
==============================================

|status|

**Demo release** — full package coming in 2026.

Python code for modelling beta diversity with
`Generalized Dissimilarity Modelling (GDM) <https://onlinelibrary.wiley.com/doi/full/10.1111/geb.13459>`_
and its hierarchical Bayesian extension
`spGDMM <https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.14259>`_.

This repository forms part of PhD research on plankton beta-diversity
modelling from environmental and eDNA data.


Overview
--------

The demo notebook (``demo.ipynb``) fits **spGDMM** to plankton community data
(temperature, salinity, and vorticity → five plankton groups)
derived from the SINMOD ocean model at 80 spatial sites.

The workflow demonstrates:

- Model specification
- Bayesian inference
- Convergence diagnostics
- Predictive scoring (CRPS)
- Community turnover mapping
- I-spline response curves


Installation
------------

From the repository root:

.. code-block:: bash

   mamba env create -f environment.yml
   mamba activate spgdmm-demo
   jupyter lab demo.ipynb


Scientific Background
---------------------

Original model implementation in R:

White, P.A., Frye, H.A., Slingsby, J.A., Silander, J.A. & Gelfand, A.E. (2024).
*Generative spatial generalized dissimilarity mixed modelling (spGDMM):
An enhanced approach to modelling beta diversity.*
Methods in Ecology and Evolution, 15(1), 214–226.
https://doi.org/10.1111/2041-210X.14259


Citation
--------

If you use this code in academic work, please cite the spGDMM paper above.
A formal package citation file (``CITATION.cff``) will be added with the full release.


Project Status
--------------

This repository currently provides:

- A working demonstration notebook
- A reproducible modelling environment
- A research prototype implementation

The full software package and documentation are planned for release in 2026.


Contact
-------

Harold Horsley  
NTNU  
harold.horsley@ntnu.no


.. |status| image:: https://img.shields.io/badge/status-demo--release-orange