# Preprocessing / data extraction

These scripts document how the analysis-ready data were produced from the raw SINMOD
ocean-model output. They are provided for transparency and reproducibility of the
*procedure*. The input paths point to the internal SINMOD archive (the full coupled
physical–biogeochemical model output), which is not redistributed here; the compact
extracted product used by the demo (`../demo_data/day105.npz`) is the public artefact in this
repository, and the full predictor and response datasets will be deposited on Dryad on
publication.

| Script | Purpose |
|---|---|
| `../demo_data/extract_day105.py` | Builds the compact `day105.npz` used by `demo.ipynb`: masks valid ocean cells (non-NaN predictors and biomass, biomass > 0), draws sample sites uniformly at random without replacement, and stores site-level predictors/biomass plus a subsampled grid. |
| `locationsampler.py` | The `LocationSampler` class used for the train/test location draws in the analysis: samples grid cells uniformly at random (without replacement) from the largest connected ocean region, with an optional boundary-buffer exclusion. |

## Train/test location selection (as used in the paper)

For each of six repeated spatial hold-out splits, a training set (N_train = 50) and a
non-overlapping test set (N_test = 100) of ocean grid cells are drawn uniformly at random
without replacement, and held fixed across days so that only the daily environmental and
community data vary. The six repeats use independent random splits.
