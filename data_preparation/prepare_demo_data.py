"""
prepare_demo_data.py
====================
Produce demo_data/day105_publication.npz with 50 train + 100 test sites,
matching the paper's experimental design (day 105, seed=1).

Samples valid ocean locations directly from the SINMOD grid — no pipeline
dependency. Run once on the cluster:

    cd bayesian-hierarchical-beta-diversity-modelling
    python prepare_demo_data.py
"""

import random
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from scipy.ndimage import label

from demo_connectivity import compute_connectivity

# ── Paths ─────────────────────────────────────────────────────────────────────
FEATURES_PATH = "/cluster/home/haroldh/spGDMM/1_data/4_interim/midnor_features_with_FTLE.nc"
TAVG_PATH     = "/cluster/home/haroldh/spGDMM/1_data/4_interim/midnor_time_averaged_features.nc"
BIO_PATH      = "/cluster/home/haroldh/spGDMM/1_data/4_interim/midnor_biostates_surface.nc"
NC_FILE       = "/cluster/projects/itk-SINMOD/coral-mapping/midnor/samp_2D_jun-aug.nc"
NC_FILE_2     = "/cluster/projects/itk-SINMOD/coral-mapping/midnor/samp_2D_jan_jun.nc"
NC_FILE_3     = "/cluster/projects/itk-SINMOD/coral-mapping/midnor/samp_2D_sep-dec.nc"
OUT_PATH      = Path("demo_data/day105_publication.npz")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Settings ──────────────────────────────────────────────────────────────────
SPLIT_SEED  = 1      # controls which locations are selected
TIME_IDX    = 105    # day 105 (15 April 2019)
N_TRAIN     = 50
N_TEST      = 100
N_PARTICLES = 1000
CONN_SEED   = 5      # Lagrangian simulation seed (matches original demo)

# ── Step 1: Find valid ocean locations from SINMOD grid ───────────────────────
print(f"Loading valid ocean mask from SINMOD (day {TIME_IDX})...")
with xr.open_dataset(FEATURES_PATH) as ds:
    vorticity_slice = ds["vorticity"].isel(time=TIME_IDX)
    valid_mask = vorticity_slice.notnull().values   # (yc, xc) bool

# Keep only the largest connected ocean region
structure = np.ones((3, 3), dtype=bool)
connected_labels, _ = label(valid_mask, structure=structure)
counts = np.bincount(connected_labels.ravel())
counts[0] = 0
largest_label = np.argmax(counts)
valid_mask = (connected_labels == largest_label)

# Stack to (N_valid, 2) array of [xc, yc] integer indices
ys, xs = np.where(valid_mask)
valid_coords = list(zip(xs.tolist(), ys.tolist()))   # [(xc, yc), ...]
print(f"  Valid ocean cells: {len(valid_coords):,}")

# ── Step 2: Sample train + test locations ────────────────────────────────────
print(f"Sampling {N_TRAIN} train + {N_TEST} test locations (seed={SPLIT_SEED})...")
random.seed(SPLIT_SEED)
all_sample = random.sample(valid_coords, N_TRAIN + N_TEST)
train_coords = all_sample[:N_TRAIN]
test_coords  = all_sample[N_TRAIN:]

train_xc = np.array([c[0] for c in train_coords], dtype=int)
train_yc = np.array([c[1] for c in train_coords], dtype=int)
test_xc  = np.array([c[0] for c in test_coords],  dtype=int)
test_yc  = np.array([c[1] for c in test_coords],  dtype=int)
print(f"  Train xc [{train_xc.min()}–{train_xc.max()}], yc [{train_yc.min()}–{train_yc.max()}]")
print(f"  Test  xc [{test_xc.min()}–{test_xc.max()}],  yc [{test_yc.min()}–{test_yc.max()}]")

# ── Step 3: Load TR predictors ────────────────────────────────────────────────
print("Loading TR predictors (temperature, salinity, vorticity) at day 105...")
with xr.open_dataset(FEATURES_PATH) as ds:
    tr_temp = ds["temperature"].isel(time=TIME_IDX).values   # (yc, xc)
    tr_sal  = ds["salinity"].isel(time=TIME_IDX).values
    tr_vort = ds["vorticity"].isel(time=TIME_IDX).values
    grid_temperature = tr_temp[::4, ::4].astype(np.float32)
    grid_salinity    = tr_sal[::4, ::4].astype(np.float32)

train_temperature = tr_temp[train_yc, train_xc].astype(np.float32)
train_salinity    = tr_sal[train_yc,  train_xc].astype(np.float32)
train_vorticity   = tr_vort[train_yc, train_xc].astype(np.float32)
test_temperature  = tr_temp[test_yc, test_xc].astype(np.float32)
test_salinity     = tr_sal[test_yc,  test_xc].astype(np.float32)
test_vorticity    = tr_vort[test_yc, test_xc].astype(np.float32)

ny_full, nx_full = tr_temp.shape
xs_sub = np.arange(0, nx_full, 4, dtype=np.float32)
ys_sub = np.arange(0, ny_full, 4, dtype=np.float32)
grid_xc, grid_yc = np.meshgrid(xs_sub, ys_sub)

# ── Step 4: Load TA predictors ────────────────────────────────────────────────
print("Loading TA predictors (time-averaged temperature, salinity, vorticity)...")
with xr.open_dataset(TAVG_PATH) as ds:
    ta_temp = ds["temperature"].isel(time=TIME_IDX).values
    ta_sal  = ds["salinity"].isel(time=TIME_IDX).values
    ta_vort = ds["vorticity"].isel(time=TIME_IDX).values
    grid_vorticity = ta_vort[::4, ::4].astype(np.float32)

train_temperature_ta = ta_temp[train_yc, train_xc].astype(np.float32)
train_salinity_ta    = ta_sal[train_yc,  train_xc].astype(np.float32)
train_vorticity_ta   = ta_vort[train_yc, train_xc].astype(np.float32)
test_temperature_ta  = ta_temp[test_yc, test_xc].astype(np.float32)
test_salinity_ta     = ta_sal[test_yc,  test_xc].astype(np.float32)
test_vorticity_ta    = ta_vort[test_yc, test_xc].astype(np.float32)

# ── Step 5: Load biomass ──────────────────────────────────────────────────────
print("Loading biomass at day 105...")
species = ["diatoms", "flagellates", "ciliates", "HNANO", "bacteria"]
with xr.open_dataset(BIO_PATH) as ds:
    train_bio_cols, test_bio_cols = [], []
    for sp in species:
        arr = ds[sp].isel(time=TIME_IDX).values
        train_bio_cols.append(arr[train_yc, train_xc].astype(np.float32))
        test_bio_cols.append(arr[test_yc,   test_xc].astype(np.float32))

train_biomass = np.column_stack(train_bio_cols)   # (50, 5)
test_biomass  = np.column_stack(test_bio_cols)    # (100, 5)
print(f"  Train biomass range [{train_biomass.min():.4f}, {train_biomass.max():.4f}]")
print(f"  Test  biomass range [{test_biomass.min():.4f},  {test_biomass.max():.4f}]")

# ── Step 6: Compute connectivity ─────────────────────────────────────────────
print(f"\nComputing train connectivity ({N_TRAIN}×{N_TRAIN})...")
_, conn_train_mat = compute_connectivity(
    locs        = pd.DataFrame({"xc": train_xc, "yc": train_yc}),
    t0_doy      = TIME_IDX,
    nc_file     = NC_FILE,
    nc_file_2   = NC_FILE_2,
    nc_file_3   = NC_FILE_3,
    n_particles = N_PARTICLES,
    dt          = 300.0,
    nsteps      = 20000,
    direction   = "backward",
    random_seed = CONN_SEED,
)
print(f"  range [{conn_train_mat.min():.3f}, {conn_train_mat.max():.3f}]")

print(f"\nComputing test connectivity ({N_TEST}×{N_TEST})...")
_, conn_test_mat = compute_connectivity(
    locs        = pd.DataFrame({"xc": test_xc, "yc": test_yc}),
    t0_doy      = TIME_IDX,
    nc_file     = NC_FILE,
    nc_file_2   = NC_FILE_2,
    nc_file_3   = NC_FILE_3,
    n_particles = N_PARTICLES,
    dt          = 300.0,
    nsteps      = 20000,
    direction   = "backward",
    random_seed = CONN_SEED,
)
print(f"  range [{conn_test_mat.min():.3f}, {conn_test_mat.max():.3f}]")

# ── Step 7: Save ──────────────────────────────────────────────────────────────
print(f"\nSaving to {OUT_PATH}...")
np.savez_compressed(
    OUT_PATH,
    train_xc=train_xc.astype(np.float32), train_yc=train_yc.astype(np.float32),
    test_xc=test_xc.astype(np.float32),   test_yc=test_yc.astype(np.float32),
    time_idx=np.array([TIME_IDX]),
    train_temperature=train_temperature,     train_salinity=train_salinity,     train_vorticity=train_vorticity,
    test_temperature=test_temperature,       test_salinity=test_salinity,       test_vorticity=test_vorticity,
    train_temperature_ta=train_temperature_ta, train_salinity_ta=train_salinity_ta, train_vorticity_ta=train_vorticity_ta,
    test_temperature_ta=test_temperature_ta,   test_salinity_ta=test_salinity_ta,   test_vorticity_ta=test_vorticity_ta,
    train_biomass=train_biomass, test_biomass=test_biomass,
    biomass_names=np.array(species),
    conn_train=conn_train_mat, conn_test=conn_test_mat,
    grid_xc=grid_xc.astype(np.float32), grid_yc=grid_yc.astype(np.float32),
    grid_temperature=grid_temperature, grid_salinity=grid_salinity, grid_vorticity=grid_vorticity,
)
print("Done.")

assert conn_train_mat.shape == (N_TRAIN, N_TRAIN)
assert conn_test_mat.shape  == (N_TEST,  N_TEST)
assert train_biomass.shape  == (N_TRAIN, 5)
assert test_biomass.shape   == (N_TEST,  5)
print("Sanity checks passed.")
print(f"Split: {N_TRAIN} train → {N_TRAIN*(N_TRAIN-1)//2} pairs | "
      f"{N_TEST} test → {N_TEST*(N_TEST-1)//2} pairs")
