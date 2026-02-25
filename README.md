# Bayesian Hierarchical Beta Diversity Modelling

> **Demo release** — full package coming in 2026.

Python code for modelling beta diversity with [GDM](https://onlinelibrary.wiley.com/doi/full/10.1111/geb.13459) and its hierarchical Bayesian extension, [spGDMM](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.14259). Part of my PhD research on plankton beta-diversity modelling from environmental and eDNA data.

## Demo

`demo.ipynb` fits spGDMM to plankton community data (temperature, salinity, vorticity → 5 plankton groups) from the SINMOD ocean model at 80 sites. It covers model fitting, convergence diagnostics, predictive scoring (CRPS), community mapping, and I-spline response curves.

**Setup** (from repo root):

```bash
mamba env create -f environment.yml
mamba activate spgdmm-demo
jupyter lab demo.ipynb
```

## Based on

### Original model implementation in R:

White, P.A., Frye, H.A., Slingsby, J.A., Silander, J.A. & Gelfand, A.E. (2024). Generative spatial generalized dissimilarity mixed modelling (spGDMM): An enhanced approach to modelling beta diversity. *Methods in Ecology and Evolution*, 15(1), 214–226. https://doi.org/10.1111/2041-210X.14259

### Bayesian inference framework

Abril-Pla, O. et al. (2023). PyMC: A modern and comprehensive probabilistic programming framework in Python. *PeerJ Computer Science*, 9, e1516. https://doi.org/10.7717/peerj-cs.1516

## Contact

If any of this is interesting (or not), please let me know!

harold.horsley@ntnu.no

