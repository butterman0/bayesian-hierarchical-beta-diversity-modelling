"""
demo_connectivity.py
====================
Standalone Lagrangian particle-connectivity for the spGDMM demo.

Coordinate convention
---------------------
xc, yc  – grid-cell indices (integers), same as stored in demo_data/*.npz.
           Grid spacing is 800 m (SINMOD midnord grid).
           xc ∈ [0, nx-1],  yc ∈ [0, ny-1].

The velocity fields (u_velocity, v_velocity) in the NetCDF files are in m/s.
Each advection step converts to grid units via:  displacement = u * dt / dxy.

Returns
-------
hitsDMax         : (n, n) float  – max(hitsD, hitsD.T), diagonal = 0
connectivity_mat : (n, n) float  – -log(hitsDMax/n_particles + 1e-6), diagonal = 0
"""

import warnings
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from numba import njit


@njit
def _hits_mask(x_i, y_i, points_x, points_y, hitRng_sq):
    """Return (n_points, n_particles) bool: particle within hitRng of target."""
    n_particles = x_i.shape[0]
    n_points    = points_x.shape[0]
    hits = np.zeros((n_points, n_particles), dtype=np.bool_)
    for p in range(n_particles):
        xi = x_i[p]
        yi = y_i[p]
        if np.isnan(xi) or np.isnan(yi):
            continue
        for k in range(n_points):
            dx = xi - points_x[k]
            dy = yi - points_y[k]
            if dx * dx + dy * dy < hitRng_sq:
                hits[k, p] = True
    return hits


def compute_connectivity(
    locs,
    t0_doy,
    nc_file,
    nc_file_2,
    nc_file_3,
    n_particles=1000,
    dt=300.0,
    nsteps=20000,
    spread=0.75,
    q=0.75,
    hitRng=5.0,
    direction="backward",
    release_hour=12,
    allow_partial_window=True,
    random_seed=None,
):
    """
    Compute static window-mean Lagrangian particle connectivity.

    Parameters
    ----------
    locs : pd.DataFrame with columns 'xc', 'yc' (grid-cell indices, integers)
    t0_doy : int  – day-of-year (0-based) to select the velocity time window
    nc_file, nc_file_2, nc_file_3 : paths to SINMOD seasonal velocity NetCDF files
    n_particles : int – particles released per source site
    dt          : float – time step in seconds
    nsteps      : int – total number of advection steps
    spread      : float – initial particle spread in grid units
    q           : float – stochastic noise amplitude (grid units per step^0.5)
    hitRng      : float – hit detection radius in grid units (5 → 4 km)
    direction   : 'backward' or 'forward'
    release_hour : int – hour of day for release (used to anchor DOY to datetime)
    allow_partial_window : bool – warn instead of raise on partial time coverage
    random_seed : int or None

    Returns
    -------
    hitsDMax         : (n, n) ndarray – symmetric hit counts, diagonal = 0
    connectivity_mat : (n, n) ndarray – -log(hitsDMax/n_particles + 1e-6), diagonal = 0
    """
    dxy = 800.0  # metres per grid cell

    if random_seed is not None:
        np.random.seed(random_seed)

    # ------------------------------------------------------------------
    # 1. Load velocity NetCDFs (keep only needed variables + time dim)
    # ------------------------------------------------------------------
    ds_1 = xr.open_dataset(nc_file,   decode_times=True)[["u_velocity", "v_velocity", "time"]]
    ds_2 = xr.open_dataset(nc_file_2, decode_times=True)[["u_velocity", "v_velocity", "time"]]
    ds_3 = xr.open_dataset(nc_file_3, decode_times=True)[["u_velocity", "v_velocity", "time"]]

    for a, b, na, nb in [
        (ds_1, ds_2, "nc_file", "nc_file_2"),
        (ds_1, ds_3, "nc_file", "nc_file_3"),
    ]:
        if a.sizes.get("yc") != b.sizes.get("yc") or a.sizes.get("xc") != b.sizes.get("xc"):
            raise ValueError(f"{na} and {nb} have different spatial grids.")

    # ------------------------------------------------------------------
    # 2. Build release datetime and time window
    # ------------------------------------------------------------------
    year0 = np.datetime64(ds_1["time"].values[0], "Y")
    release_date = (
        year0
        + np.timedelta64(int(t0_doy), "D")
        + np.timedelta64(int(release_hour), "h")
    )

    window_seconds = int(round(nsteps * dt))
    if direction == "backward":
        window_start = release_date - np.timedelta64(window_seconds, "s")
        window_end   = release_date
    elif direction == "forward":
        window_start = release_date
        window_end   = release_date + np.timedelta64(window_seconds, "s")
    else:
        raise ValueError("direction must be 'backward' or 'forward'")

    # ------------------------------------------------------------------
    # 3. Select only needed time slices across the three seasonal files
    # ------------------------------------------------------------------
    def _sel_if_overlaps(ds, t0, t1):
        tmin = ds["time"].values.min()
        tmax = ds["time"].values.max()
        if t1 < tmin or t0 > tmax:
            return None
        return ds.sel(time=slice(t0, t1))

    ds_parts = []
    for ds in (ds_2, ds_1, ds_3):
        part = _sel_if_overlaps(ds, window_start, window_end)
        if part is not None and part.sizes.get("time", 0) > 0:
            ds_parts.append(part)

    if len(ds_parts) == 0:
        raise ValueError(
            f"No timesteps found for window [{window_start} .. {window_end}]. "
            "Provide NetCDF coverage for the requested t0_doy interval."
        )

    ds_win = xr.concat(ds_parts, dim="time").sortby("time")

    # Warn (or raise) if window coverage is partial
    time_vals = ds_win["time"].values
    if time_vals.size >= 2:
        diffs = np.diff(time_vals).astype("timedelta64[s]").astype(np.int64)
        diffs = diffs[diffs > 0]
        dt_native_s = int(np.median(diffs)) if diffs.size > 0 else int(max(1, dt))
    else:
        dt_native_s = int(max(1, dt))

    tol = np.timedelta64(max(int(dt * 2), dt_native_s), "s")
    if (time_vals.min() > window_start + tol) or (time_vals.max() < window_end - tol):
        msg = (
            "Partial time-window coverage; proceeding with available data. "
            f"Requested [{window_start} .. {window_end}], "
            f"selected [{time_vals.min()} .. {time_vals.max()}]."
        )
        if allow_partial_window:
            warnings.warn(msg)
        else:
            raise ValueError(msg)

    # ------------------------------------------------------------------
    # 4. Window-mean velocity field  →  (ny, nx) arrays in m/s
    # ------------------------------------------------------------------
    uv_mean = ds_win[["u_velocity", "v_velocity"]].mean("time", skipna=True)
    u_mean  = uv_mean["u_velocity"].values   # (ny, nx)
    v_mean  = uv_mean["v_velocity"].values
    ny, nx  = u_mean.shape

    # ------------------------------------------------------------------
    # 5. Build RegularGridInterpolators in grid-index coordinates
    # ------------------------------------------------------------------
    ys = np.arange(ny, dtype=float)
    xs = np.arange(nx, dtype=float)

    Fu = RegularGridInterpolator((ys, xs), u_mean, method="linear",
                                 bounds_error=False, fill_value=np.nan)
    Fv = RegularGridInterpolator((ys, xs), v_mean, method="linear",
                                 bounds_error=False, fill_value=np.nan)

    # ------------------------------------------------------------------
    # 6. Initialise particles at grid-index positions (NO /dxy conversion)
    # ------------------------------------------------------------------
    points_x = locs["xc"].values.astype(float)   # grid indices directly
    points_y = locs["yc"].values.astype(float)
    npoints  = len(points_x)

    xpos = np.array([
        points_x[i] + np.random.randn(n_particles) * spread
        for i in range(npoints)
    ])  # (npoints, n_particles)
    ypos = np.array([
        points_y[i] + np.random.randn(n_particles) * spread
        for i in range(npoints)
    ])

    hitRng_sq  = hitRng ** 2
    dt_sign    = -1.0 if direction == "backward" else 1.0
    ever_hit   = np.zeros((npoints, npoints, n_particles), dtype=bool)

    # ------------------------------------------------------------------
    # 7. Advect particles
    # ------------------------------------------------------------------
    print(f"[compute_connectivity] Advecting {npoints} × {n_particles} particles "
          f"for {nsteps} steps ({direction})...")

    for step in range(nsteps):
        pos_flat = np.column_stack([ypos.ravel(), xpos.ravel()])
        uLocal   = Fu(pos_flat).reshape(npoints, n_particles)
        vLocal   = Fv(pos_flat).reshape(npoints, n_particles)

        valid = (
            (xpos >= 0) & (xpos < nx) &
            (ypos >= 0) & (ypos < ny) &
            ~np.isnan(uLocal)
        )

        w_u = q * np.random.randn(npoints, n_particles)
        w_v = q * np.random.randn(npoints, n_particles)

        new_xpos = np.full_like(xpos, np.nan)
        new_ypos = np.full_like(ypos, np.nan)
        # u is m/s, dt is s, dxy is m/cell → displacement in grid units
        new_xpos[valid] = xpos[valid] + (uLocal[valid] + w_u[valid]) * dt_sign * dt / dxy
        new_ypos[valid] = ypos[valid] + (vLocal[valid] + w_v[valid]) * dt_sign * dt / dxy
        xpos, ypos = new_xpos, new_ypos

        # Hit detection using numba kernel
        for i in range(npoints):
            x_i = xpos[i]
            y_i = ypos[i]
            if not np.any(~np.isnan(x_i)):
                continue
            h = _hits_mask(x_i, y_i, points_x, points_y, hitRng_sq)
            ever_hit[i] |= h     # h[k,p]: particle p from site i within hitRng of site k

    # ------------------------------------------------------------------
    # 8. Aggregate to connectivity matrices
    # ------------------------------------------------------------------
    hitsD = ever_hit.sum(axis=2).astype(float)   # (npoints, npoints)
    np.fill_diagonal(hitsD, 0)
    hitsDMax = np.maximum(hitsD, hitsD.T)

    connectivity_mat = -np.log(hitsDMax / n_particles + 1e-6)
    np.fill_diagonal(connectivity_mat, 0)

    return hitsDMax, connectivity_mat
