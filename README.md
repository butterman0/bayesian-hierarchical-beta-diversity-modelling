# Bayesian Hierarchical Beta Diversity Modelling

A Python package for modelling **beta diversity** using **Generalised Dissimilarity Modelling** ([GDM](https://onlinelibrary.wiley.com/doi/full/10.1111/geb.13459)) and its **hierarchical Bayesian extension** ([spGDMM](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.14259)).

> **This is a demo release.** A fully documented, user-friendly package is planned for **2026**.

## What is this?

This package forms part of my PhD research in quantitative marine ecology, focused on modelling beta diversity from **environmental** and **eDNA** data.

It implements and extends GDM within a **hierarchical Bayesian framework (spGDMM)** to capture spatial structure, uncertainty, and compositional relationships in biodiversity data.

The project integrates:

- **Quantitative methods** – hierarchical Bayesian inference, spatial Gaussian processes, and spline-based dissimilarity modelling
- **Scientific Python tools** – `xarray`, `pandas`, `numpy`, `scikit-learn`, `pymc`, `matplotlib`
- **Geospatial and environmental data** – large-scale marine datasets from ocean models, satellites, and in-situ sampling
- **Reproducible code** – modular, well-documented code using `git`

## Demo

A self-contained demo notebook (`demo.ipynb`) shows how to fit spGDMM to plankton community data from the SINMOD ocean model. It walks through the full workflow on a single snapshot (15 April 2019, day 105) at 80 spatial locations, with temperature, salinity, and vorticity as predictors and the biomass of five plankton groups as the response.

### Requirements

Create and activate the conda/mamba environment from the repo root:

```bash
mamba env create -f environment.yml
mamba activate spgdmm-demo
```

Or with conda:

```bash
conda env create -f environment.yml
conda activate spgdmm-demo
```

This installs all dependencies and the local `spgdmm` package in one step.

### Running the demo

```bash
jupyter lab demo.ipynb
```

The notebook covers the following steps:

1. **Load data** – reads pre-processed environmental variables and biomass fields from `demo_data/day105.npz`
2. **Prepare train/test sets** – splits 80 sites into 60 training and 20 test locations; computes pairwise Bray–Curtis dissimilarity as the response variable
3. **Fit spGDMM** – fits the model with I-spline transformations and feature importance weights using HMC (via `nutpie`)
4. **Check convergence** – inspects R-hat values, effective sample sizes, and trace plots
5. **Evaluate predictive performance** – scores posterior predictive samples against held-out test pairs using CRPS
6. **Map biological space** – projects I-spline-transformed features to an RGB image via PCA to visualise inferred community structure
7. **Inspect I-spline response curves** – plots the fitted non-linear environmental response functions and importance weights (α) for each predictor

## Planned features

- GDM and spGDMM models with a consistent, scikit-learn-like API
- Spatial and compositional preprocessing utilities
- Model evaluation and uncertainty summaries
- Visualisation tools for dissimilarity, residuals, and spatial effects

## Status

**Demo / pre-release.** The full package release is planned for **2026**. Interfaces and structure are under active development.

If you're interested, please get in touch!

Email: harold.horsley@ntnu.no
