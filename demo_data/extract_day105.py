"""
Extract a compact data slice for day 105 (15 April) from the full
midnor NetCDF files and save to day105.npz for use in the demo notebook.

Run once:
    python extract_day105.py
"""

import numpy as np
import xarray as xr

FEATURE_PATH = "/cluster/home/haroldh/spGDMM/1_data/4_interim/midnor_features_with_FTLE.nc"
BIOSTATE_PATH = "/cluster/home/haroldh/spGDMM/1_data/4_interim/midnor_biostates_surface.nc"
OUT_PATH      = "/cluster/home/haroldh/spGDMM/demo_data/day105.npz"

TIME_IDX      = 105   # April 15 (0-indexed day-of-year)
N_SITES       = 80    # training + test sites to extract
GRID_STRIDE   = 4     # subsample grid every N pixels for RGB visualisation
SEED          = 42

BIO_VARS  = ["diatoms", "flagellates", "ciliates", "HNANO", "bacteria"]
PRED_VARS = ["temperature", "salinity", "vorticity"]

rng = np.random.default_rng(SEED)

print("Loading feature data ...")
feat = xr.open_dataset(FEATURE_PATH)
feat_t = feat[PRED_VARS].isel(time=TIME_IDX).load()
feat.close()

print("Loading biostate data ...")
bio  = xr.open_dataset(BIOSTATE_PATH)
bio_t = bio[BIO_VARS].isel(time=TIME_IDX).load()
bio.close()

# --- Build a valid-pixel mask (non-NaN in all pred. and bio. variables) ---
valid = np.ones((feat_t.sizes["yc"], feat_t.sizes["xc"]), dtype=bool)
for v in PRED_VARS:
    valid &= ~np.isnan(feat_t[v].values)
for v in BIO_VARS:
    valid &= ~np.isnan(bio_t[v].values)
    valid &= (bio_t[v].values > 0)   # no zero-abundance sites

yc_all, xc_all = np.where(valid)
print(f"Valid cells: {len(yc_all):,}")

# --- Sample N_SITES random valid locations ---
chosen = rng.choice(len(yc_all), size=N_SITES, replace=False)
yc_s = yc_all[chosen]
xc_s = xc_all[chosen]

T_s = np.array([feat_t["temperature"].values[yc_s[i], xc_s[i]] for i in range(N_SITES)])
S_s = np.array([feat_t["salinity"].values   [yc_s[i], xc_s[i]] for i in range(N_SITES)])
V_s = np.array([feat_t["vorticity"].values  [yc_s[i], xc_s[i]] for i in range(N_SITES)])
B_s = np.stack(
    [bio_t[v].values[yc_s, xc_s] for v in BIO_VARS], axis=1
)  # shape (N_SITES, 5)
# Normalise biomass to relative proportions
B_s = B_s / B_s.sum(axis=1, keepdims=True)

print(f"Extracted {N_SITES} sites")

# --- Build subsampled prediction grid ---
yc_idx = np.arange(0, feat_t.sizes["yc"], GRID_STRIDE)
xc_idx = np.arange(0, feat_t.sizes["xc"], GRID_STRIDE)
YC_g, XC_g = np.meshgrid(yc_idx, xc_idx, indexing="ij")  # (ny_g, nx_g)

T_g = feat_t["temperature"].values[YC_g, XC_g].astype(np.float32)
S_g = feat_t["salinity"].values   [YC_g, XC_g].astype(np.float32)
V_g = feat_t["vorticity"].values  [YC_g, XC_g].astype(np.float32)

print(f"Grid shape: {T_g.shape}")

# --- Save ---
np.savez_compressed(
    OUT_PATH,
    # site-level data
    xc          = xc_s.astype(np.float32),
    yc          = yc_s.astype(np.float32),
    temperature = T_s.astype(np.float32),
    salinity    = S_s.astype(np.float32),
    vorticity   = V_s.astype(np.float32),
    biomass     = B_s.astype(np.float32),   # (N_SITES, 5)
    # grid data (for RGB visualisation)
    grid_xc          = XC_g.astype(np.float32),
    grid_yc          = YC_g.astype(np.float32),
    grid_temperature = T_g,
    grid_salinity    = S_g,
    grid_vorticity   = V_g,
    # metadata
    biomass_names = np.array(BIO_VARS),
    time_idx      = np.array([TIME_IDX]),
)

import os
size_kb = os.path.getsize(OUT_PATH) / 1024
print(f"Saved {OUT_PATH}  ({size_kb:.0f} KB)")
