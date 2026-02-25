"""
Plotting utilities for fitted spGDMM models.
"""

import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from dms_variants.ispline import Isplines


def plot_isplines(model, features=None, hdi_prob=0.9, figsize=(6.5, 4)):
    """
    Plot I-spline effect curves for each environmental predictor.

    For each predictor, shows the individual basis-function contributions
    (thin coloured lines) and their weighted sum (black line) with a
    credible-interval band.  Also reports the fitted alpha importance weight
    in the legend when ``alpha_importance=True``.

    Parameters
    ----------
    model : fitted spGDMM
        A model on which ``.fit()`` has been called.
    features : list of str, optional
        Subset of feature names to plot.  Defaults to all environmental
        features (distance excluded).
    hdi_prob : float
        Width of the highest-density interval band (default 0.90).
    figsize : tuple
        Figure size passed to matplotlib.

    Returns
    -------
    None  (displays plots inline)
    """
    idata    = model.idata
    config   = model.config
    meta     = model.training_metadata

    beta = idata.posterior.beta   # (chain, draw, feature, basis_function)
    n_bases   = beta.sizes["basis_function"]
    all_feats = list(beta.coords["feature"].values)

    # Drop distance / time predictors — they are not environmental effects
    skip = {"distance", "temporal_diff", "distance_euclidean", "binary_connectivity"}
    env_feats = [f for f in all_feats if f not in skip]
    if features is not None:
        env_feats = [f for f in env_feats if f in features]

    beta_mean = beta.mean(dim=["chain", "draw"])  # (feature, basis_function)
    hdi_vals  = az.hdi(beta, hdi_prob=hdi_prob)
    beta_lo   = hdi_vals.sel(hdi="lower").beta
    beta_hi   = hdi_vals.sel(hdi="higher").beta

    has_alpha = "alpha" in idata.posterior
    if has_alpha:
        alpha_mean = idata.posterior.alpha.mean(dim=["chain", "draw"])

    predictor_mesh = meta["predictor_mesh"]  # (knots+2, n_env_features)
    deg  = config["deg"]
    knots = config["knots"]

    for i, feat in enumerate(env_feats):
        mesh_col = predictor_mesh[:, i]
        x = np.linspace(mesh_col[0], mesh_col[-1], 300)

        spline   = Isplines(deg, mesh_col, x)
        basis    = np.column_stack([spline.I(j + 1) for j in range(n_bases)])

        bm  = beta_mean.sel(feature=feat).values   # (n_bases,)
        blo = beta_lo.sel(feature=feat).values
        bhi = beta_hi.sel(feature=feat).values

        comp    = basis * bm
        comp_lo = basis * blo
        comp_hi = basis * bhi
        total    = comp.sum(axis=1)
        total_lo = comp_lo.sum(axis=1)
        total_hi = comp_hi.sum(axis=1)

        fig, ax = plt.subplots(figsize=figsize)

        colors = plt.cm.tab10(np.linspace(0, 0.9, n_bases))
        for j in range(n_bases):
            ax.plot(x, comp[:, j], lw=1, color=colors[j],
                    label=f"I{j+1}  β={bm[j]:.2f}")

        ax.plot(x, total, color="black", lw=2, label="Total")
        ax.fill_between(x, total_lo, total_hi,
                        color="black", alpha=0.15,
                        label=f"{int(hdi_prob*100)}% HDI")

        if has_alpha:
            a_val = float(alpha_mean.sel(feature=feat).values)
            ax.plot([], [], " ", label=f"α = {a_val:.2f}")

        # Mark internal knots
        for ki, kv in enumerate(mesh_col[1:-1]):
            ax.axvline(kv, color="gray", linestyle="--", lw=0.8,
                       label="Knot" if ki == 0 else None)

        ax.set_title(feat)
        ax.set_xlabel("Predictor value")
        ax.set_ylabel("Effect")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7, ncol=2)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.show()


def plot_crps_comparison(y_test, y_pred, y_train, use_log=False, figsize=(7, 4)):
    """
    Plot CRPS comparison between model and null baseline, plus CRPS skill scores.
    
    Creates two side-by-side boxplots:
    1. CRPS comparison: model vs null baseline
    2. CRPS skill scores: 1 - (CRPS_model / CRPS_null)
    
    Parameters
    ----------
    y_test : array-like
        Test observations (e.g., Bray-Curtis dissimilarity values).
    y_pred : array-like (n_test, n_samples)
        Posterior predictive samples from the model.
    y_train : array-like
        Training observations used for null model baseline.
    use_log : bool
        If False (default), compute CRPS on original scale (0-1 for Bray-Curtis).
        If True, compute CRPS on log-transformed values.
    figsize : tuple
        Figure size for the two subplots (width, height).
    
    Returns
    -------
    fig, axes : matplotlib Figure and Axes
        The created figure and axes objects.
    """
    from properscoring import crps_ensemble
    
    # Prepare data on appropriate scale
    if use_log:
        y_test_scale = np.log(y_test)
        y_pred_scale = y_pred.values if hasattr(y_pred, 'values') else y_pred
        y_train_scale = np.log(y_train)
        ylabel = "CRPS (log scale)"
    else:
        y_test_scale = y_test
        y_pred_scale = y_pred.values if hasattr(y_pred, 'values') else y_pred
        y_train_scale = y_train
        ylabel = "CRPS (Bray–Curtis)"
    
    # Compute CRPS for model
    crps_model = crps_ensemble(y_test_scale, y_pred_scale)
    
    # Compute CRPS for null (use training data as forecast ensemble)
    crps_null = crps_ensemble(y_test_scale, np.tile(y_train_scale, (len(y_test_scale), 1)))
    
    # Compute CRPS skill score: 1 - (CRPS_model / CRPS_null)
    crps_skill = 1.0 - (crps_model / crps_null)
    
    # Create two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Left plot: CRPS comparison
    axes[0].boxplot(
        [crps_model, crps_null], 
        labels=["spGDMM", "Null"], 
        widths=0.5,
        patch_artist=True,
        boxprops=dict(facecolor="steelblue", alpha=0.6),
        medianprops=dict(color="black", linewidth=2),
    )
    axes[0].set_ylabel(ylabel)
    axes[0].set_title("CRPS Comparison")
    axes[0].spines[["top", "right"]].set_visible(False)
    
    # Right plot: CRPS skill scores
    axes[1].boxplot(
        [crps_skill], 
        labels=["spGDMM"], 
        widths=0.5,
        patch_artist=True,
        boxprops=dict(facecolor="forestgreen", alpha=0.6),
        medianprops=dict(color="black", linewidth=2),
    )
    axes[1].axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    axes[1].set_ylabel("CRPS Skill (1 − CRPS/CRPS_null)")
    axes[1].set_title("CRPS Skill Score")
    axes[1].spines[["top", "right"]].set_visible(False)
    
    plt.tight_layout()
    return fig, axes
