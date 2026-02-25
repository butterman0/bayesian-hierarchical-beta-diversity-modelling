# spGDMM

Spatial Generalised Dissimilarity Mixed Model (spGDMM) — a Bayesian model for predicting pairwise community dissimilarity as a function of environmental predictors and spatial distance.

## Installation

```bash
pip install spgdmm
```

## Quick start

```python
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from spgdmm import spGDMM

# X: site-level DataFrame with columns [xc, yc, time_idx, predictor1, ...]
# y: pre-computed condensed pairwise Bray-Curtis dissimilarities
y = pdist(biomass_matrix, "braycurtis")

model = spGDMM(
    model_str="model1",
    distance_measure="euclidean",
    alpha_importance=True,
)
idata = model.fit(X, y)
```

## Citation

[Paper citation to be added]
