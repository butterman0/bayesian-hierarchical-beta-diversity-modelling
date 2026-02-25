import pymc as pm
from typing import Dict, Optional, Union
import numpy as np
import xarray as xr
from scipy.spatial.distance import pdist
from dms_variants.ispline import Isplines
import pandas as pd
import arviz as az
import json
import warnings
import json_tricks as jt
from ._base import ModelBuilder
from sklearn.decomposition import PCA


class spGDMM(ModelBuilder):
    _model_type = "spGDMM"
    version = "0.2"

    def __init__(
        self,
        deg: int = 3,
        knots: int = 2,
        mesh_choice: str = "percentile",
        distance_measure: str = "euclidean",
        alpha_importance: bool = True,
        custom_predictor_mesh: Optional[np.ndarray] = None,
        custom_dist_mesh: Optional[np.ndarray] = None,
        **kwargs,
    ):
        """Initialize spGDMM model.
        
        Parameters
        ----------
        deg : int
            Degree of I-spline basis functions (default: 3).
        knots : int
            Number of internal knots for I-splines (default: 2).
        mesh_choice : str
            Method for placing knots: 'percentile', 'even', or 'custom' (default: 'percentile').
        distance_measure : str
            Distance metric for spatial coordinates (default: 'euclidean').
        alpha_importance : bool
            Whether to estimate feature importance weights (default: True).
        custom_predictor_mesh : np.ndarray, optional
            Custom knot locations for environmental predictors.
        custom_dist_mesh : np.ndarray, optional
            Custom knot locations for spatial distance.
        **kwargs
            Additional arguments passed to ModelBuilder.
        """
        self._config = {
            "deg": deg,
            "knots": knots,
            "mesh_choice": mesh_choice,
            "distance_measure": distance_measure,
            "alpha_importance": alpha_importance,
            "custom_predictor_mesh": custom_predictor_mesh,
            "custom_dist_mesh": custom_dist_mesh,
        }
        self.metadata = None
        super().__init__(**kwargs)

    def build_model(self, X, log_y, prediction=False, **kwargs):
        if prediction and not hasattr(self, "idata"):
            raise ValueError("Cannot build model with prediction=True before model is trained.")

        if self.metadata is None:
            self._config = json.loads(self.idata.attrs["config"])
            if "metadata" in self.idata.attrs and self.idata.attrs["metadata"] is not None:
                self.metadata = jt.loads(self.idata.attrs["metadata"])
                self.training_metadata = jt.loads(self.idata.attrs["training_metadata"])
            self.X = self.idata.constant_data.X_data
            self.y = self.idata.constant_data.log_y_data.values
            return self.build_model(self.X, self.y, **kwargs)

        self.model_coords = {
            "obs_pair": np.arange(X.shape[0]),
            "predictor": self.metadata["column_names"],
            "site_pair": np.arange(X.shape[0]),
            "site_train": np.arange(self.metadata["no_sites_train"]),
            "feature": self.metadata["predictors"],
            "basis_function": np.arange(1, self.config["deg"] + self.config["knots"] + 1),
        }

        X_values = X.values if isinstance(X, pd.DataFrame) else X
        log_y_values = log_y.values if isinstance(log_y, pd.Series) else log_y

        with pm.Model(coords=self.model_coords) as model:
            X_data = pm.Data("X_data", X_values, dims=("obs_pair", "predictor"))
            log_y_data = pm.Data("log_y_data", log_y_values, dims=("obs_pair",))

            beta_0 = pm.Normal("beta_0", mu=0, sigma=10)

            J = self.config["deg"] + self.config["knots"]
            F = len(self.metadata["predictors"])
            beta = pm.Dirichlet("beta", a=np.ones(J), shape=(F, J), dims=("feature", "basis_function"))

            X_reshaped = X_data.reshape((-1, F, J))
            warped = (X_reshaped * beta[None, :, :]).sum(axis=2)

            alpha = pm.HalfNormal("alpha", sigma=10, shape=F, dims=("feature",))
            mu = beta_0 + pm.math.dot(warped, alpha)

            sigma2 = pm.InverseGamma("sigma2", alpha=1, beta=1)

            if prediction:
                pm.Censored("log_y", pm.Normal.dist(mu=mu, sigma=pm.math.sqrt(sigma2)), lower=None, upper=0)
                self.prediction_model = model
            else:
                pm.Censored("log_y", pm.Normal.dist(mu=mu, sigma=pm.math.sqrt(sigma2)), lower=None, upper=0, observed=log_y_data)
                self.model = model

    def _data_setter(self, X, y=None):
        X_values = self._transform_for_prediction(X)
        y_values = np.log(y.values) if isinstance(y, pd.Series) else (np.log(y) if y is not None else np.zeros(X_values.shape[0]))
        self.build_model(X_values, y_values, prediction=True)

    def get_default_model_config(self) -> Dict:
        # Priors are hardcoded in build_model; this satisfies the abstract interface.
        return {}

    def get_default_sampler_config(self) -> Dict:
        """Default MCMC sampler configuration.
        
        Returns
        -------
        dict
            Default sampler settings optimized for HMC sampling.
        """
        return {
            "draws": 1000,
            "tune": 1000,
            "chains": 4,
            "target_accept": 0.95,
            "nuts_sampler": "nutpie",
        }

    def pw_distance(self, location_values: np.ndarray) -> np.ndarray:
        return pdist(location_values, metric="euclidean") / 1000

    def _transform_for_training(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        print("[spGDMM] Starting _transform_for_training...")

        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError("X must be Pandas DataFrame or numpy array")

        if self.config["deg"] <= 0 or self.config["knots"] <= 0:
            raise ValueError("Degree and knots must be positive integers.")

        deg_freedom = self.config["deg"] + self.config["knots"]

        location_values = X.iloc[:, :2].values if isinstance(X, pd.DataFrame) else X[:, :2]
        X_values = X.iloc[:, 3:].values if isinstance(X, pd.DataFrame) else X[:, 3:]

        print(f"[spGDMM] Calculating pairwise distances for {location_values.shape[0]} locations...")
        pw_distance = self.pw_distance(location_values)

        if self.config["mesh_choice"] == "percentile":
            dist_mesh = np.percentile(pw_distance, np.linspace(0, 100, self.config["knots"] + 2))
            if np.any(np.diff(dist_mesh) == 0):
                warnings.warn("[spGDMM] Distance mesh has repeated values. Falling back to even spacing.")
                dist_mesh = np.linspace(pw_distance.min(), pw_distance.max(), self.config["knots"] + 2)

            predictor_mesh = np.empty((self.config["knots"] + 2, X_values.shape[1]))
            for col in range(X_values.shape[1]):
                x = X_values[:, col]
                q = np.percentile(x, np.linspace(0, 100, self.config["knots"] + 2))
                if np.any(np.diff(q) == 0):
                    warnings.warn(f"[spGDMM] Predictor column {col} has repeated values. Falling back to even spacing.")
                    q = np.linspace(x.min(), x.max(), self.config["knots"] + 2)
                predictor_mesh[:, col] = q

        elif self.config["mesh_choice"] == "even":
            dist_mesh = np.linspace(pw_distance.min(), pw_distance.max(), self.config["knots"] + 2)
            predictor_mesh = np.column_stack([
                np.linspace(X_values[:, col].min(), X_values[:, col].max(), self.config["knots"] + 2)
                for col in range(X_values.shape[1])
            ])

        elif self.config["mesh_choice"] == "custom":
            if self.config["custom_dist_mesh"] is None or self.config["custom_predictor_mesh"] is None:
                raise ValueError("Custom mesh requires custom_dist_mesh and custom_predictor_mesh.")
            dist_mesh = self.config["custom_dist_mesh"]
            predictor_mesh = self.config["custom_predictor_mesh"]

        else:
            raise ValueError(f"Unknown mesh_choice: {self.config['mesh_choice']}")

        print("[spGDMM] Calculating I-spline bases for predictors...")
        I_spline_bases = np.column_stack([
            Isplines(self.config["deg"], predictor_mesh[:, i], X_values[:, i]).I(j)
            for i in range(X_values.shape[1])
            for j in range(1, deg_freedom + 1)
        ])

        I_spline_bases_diffs = np.array([
            pdist(I_spline_bases[:, i].reshape(-1, 1), metric="euclidean")
            for i in range(I_spline_bases.shape[1])
        ]).T

        dist_predictors = np.column_stack([
            Isplines(self.config["deg"], dist_mesh, pw_distance).I(i)
            for i in range(1, deg_freedom + 1)
        ])

        X_GDM = np.column_stack([I_spline_bases_diffs, dist_predictors])

        self.training_metadata = {
            "I_spline_bases": I_spline_bases,
            "I_spline_bases_diffs": I_spline_bases_diffs,
            "dist_predictors": dist_predictors,
            "pw_distance": pw_distance,
            "dist_mesh": dist_mesh,
            "predictor_mesh": predictor_mesh,
            "location_values_train": location_values,
        }

        print("[spGDMM] _transform_for_training complete.")
        return X_GDM

    def _generate_and_preprocess_model_data(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a Pandas DataFrame")
        if not isinstance(y, (pd.Series, np.ndarray)):
            raise TypeError("y must be Pandas Series or numpy array")

        self.prediction_metadata = None

        X_GDM = self._transform_for_training(X)

        predictor_names = list(X.iloc[:, 3:].columns)
        column_names = [
            f"{col}_I{j}"
            for col in predictor_names
            for j in range(1, self.config["deg"] + self.config["knots"] + 1)
        ]
        column_names += [f"Dist_I{j}" for j in range(1, self.config["deg"] + self.config["knots"] + 1)]
        predictor_names.append("distance")

        self.X = pd.DataFrame(X_GDM, columns=column_names)
        print(f"[spGDMM] Generated X_GDM with shape {X_GDM.shape} and columns: {column_names}")

        y_values = y.values if isinstance(y, pd.Series) else y
        self.y = np.log(y_values)

        no_sites_train = X.shape[0]
        row_ind, col_ind = np.triu_indices(no_sites_train, k=1)

        self.metadata = {
            "row_ind": row_ind,
            "col_ind": col_ind,
            "no_cols": X_GDM.shape[1],
            "no_sites_train": no_sites_train,
            "proportion_zeros": np.mean(y_values == 0),
            "proportion_ones": np.mean(y_values == 1),
            "predictors": predictor_names,
            "column_names": column_names,
            "var_names": ["log_y"],
        }

    def _transform_for_prediction(self, X_pred: Union[pd.DataFrame, np.ndarray], biological_space: bool = False) -> np.ndarray:
        if not isinstance(X_pred, (pd.DataFrame, np.ndarray)):
            raise TypeError("X must be Pandas DataFrame or numpy array")

        if self.idata is None or "posterior" not in self.idata:
            raise ValueError("Model must be fitted before transforming data.")

        location_values = X_pred.iloc[:, :2].values if isinstance(X_pred, pd.DataFrame) else X_pred[:, :2]
        X_values = X_pred.iloc[:, 3:].values if isinstance(X_pred, pd.DataFrame) else X_pred[:, 3:]

        X_values_clipped = np.clip(
            X_values,
            self.training_metadata["predictor_mesh"][0, :],
            self.training_metadata["predictor_mesh"][-1, :],
        )

        n_clipped = np.sum(X_values_clipped != X_values)
        if n_clipped > 0:
            warnings.warn(f"{n_clipped} env values were clipped to predictor_mesh bounds.")

        NaN_mask = np.isnan(X_values_clipped)
        X_clipped_nonan = X_values_clipped[~NaN_mask.any(axis=1)]

        I_spline_bases = np.column_stack([
            Isplines(self.config["deg"], self.training_metadata["predictor_mesh"][:, i], X_clipped_nonan[:, i]).I(j)
            for i in range(X_clipped_nonan.shape[1])
            for j in range(1, self.config["deg"] + self.config["knots"] + 1)
        ])

        I_spline_bases_full = np.full((X_values_clipped.shape[0], I_spline_bases.shape[1]), np.nan)
        I_spline_bases_full[~NaN_mask.any(axis=1)] = I_spline_bases

        if biological_space:
            return I_spline_bases_full

        I_spline_bases_diffs = np.array([
            pdist(I_spline_bases_full[:, i].reshape(-1, 1), metric="euclidean")
            for i in range(I_spline_bases_full.shape[1])
        ]).T

        pw_distance = self.pw_distance(location_values)
        pw_distance_clipped = np.clip(
            pw_distance,
            self.training_metadata["dist_mesh"][0],
            self.training_metadata["dist_mesh"][-1],
        )

        dist_predictors = np.column_stack([
            Isplines(self.config["deg"], self.training_metadata["dist_mesh"], pw_distance_clipped).I(j)
            for j in range(1, self.config["deg"] + self.config["knots"] + 1)
        ])

        X_GDM_pred = np.column_stack([I_spline_bases_diffs, dist_predictors])

        no_sites_test = X_values.shape[0]
        row_ind_test, col_ind_test = np.triu_indices(no_sites_test, k=1)

        self.prediction_metadata = {
            "I_spline_bases": I_spline_bases_full,
            "I_spline_bases_diffs": I_spline_bases_diffs,
            "dist_predictors": dist_predictors,
            "pw_distance": pw_distance,
            "location_values_test": location_values,
            "pw_distance_clipped": pw_distance_clipped,
            "row_ind_test": row_ind_test,
            "col_ind_test": col_ind_test,
            "no_sites_test": no_sites_test,
        }

        self.idata.attrs["prediction_metadata"] = jt.dumps(self.prediction_metadata)
        return X_GDM_pred

    def _predict_biological_space(self, X_pred: pd.DataFrame, metric: str = "median", add_idata: bool = False):
        if metric == "mean":
            beta_posterior_summary = self.idata.posterior.beta.mean(dim=["chain", "draw"])
        else:
            beta_posterior_summary = self.idata.posterior.beta.median(dim=["chain", "draw"])

        # Drop spatial distance feature for biological space representation
        beta_posterior_summary = beta_posterior_summary.drop_sel(feature="distance")

        X_pred_splined = self._transform_for_prediction(X_pred, biological_space=True)
        X_pred_splined = X_pred_splined.reshape(
            1, -1,
            beta_posterior_summary.sizes["feature"],
            beta_posterior_summary.sizes["basis_function"],
        )
        X_pred_splined = xr.DataArray(
            X_pred_splined,
            dims=("time", "grid_cell", "feature", "basis_function"),
            coords={
                "time": [0],
                "grid_cell": X_pred.index,
                "feature": beta_posterior_summary["feature"].values,
                "basis_function": beta_posterior_summary["basis_function"].values,
            },
        )

        out_da = (X_pred_splined * beta_posterior_summary).sum(dim="basis_function", skipna=False)
        return out_da

    def rgb_biological_space(self, X_pred: pd.DataFrame, metric: str = "median", add_idata: bool = False):
        transformed_features = self._predict_biological_space(X_pred, metric=metric, add_idata=add_idata)

        tf = transformed_features.unstack("grid_cell")
        valid = ~tf.isnull().any(dim="feature")
        X_all = tf.where(valid).stack(sample=("time", "yc", "xc")).dropna(dim="sample")
        if X_all.sizes["sample"] == 0:
            raise ValueError("No fully observed rows available for PCA.")
        X_mat = X_all.transpose("sample", "feature").values

        pca = PCA(n_components=3).fit(X_mat)
        PC_ref = pca.transform(X_mat)
        pc_min, pc_max = PC_ref.min(axis=0), PC_ref.max(axis=0)
        pc_rng = np.where(pc_max > pc_min, pc_max - pc_min, 1.0)

        X_full = tf.transpose("time", "yc", "xc", "feature").stack(sample=("time", "yc", "xc"))
        X_valid = X_full.sel(sample=X_all["sample"]).transpose("sample", "feature").values
        PC_curr = pca.transform(X_valid)

        pcs_full = xr.DataArray(
            np.full((tf.sizes["time"], tf.sizes["yc"], tf.sizes["xc"], 3), np.nan, dtype=float),
            dims=("time", "yc", "xc", "pc"),
            coords={"time": tf["time"], "yc": tf["yc"], "xc": tf["xc"], "pc": range(3)},
        ).stack(sample=("time", "yc", "xc"))

        pcs_full.loc[dict(sample=X_all["sample"])] = xr.DataArray(
            PC_curr, dims=("sample", "pc"), coords={"sample": X_all["sample"], "pc": pcs_full["pc"]}
        )
        pcs_full = pcs_full.unstack("sample")

        pcs_norm = (pcs_full - xr.DataArray(pc_min, dims="pc", coords={"pc": pcs_full["pc"]})) / \
                   xr.DataArray(pc_rng, dims="pc", coords={"pc": pcs_full["pc"]})
        pcs_norm = xr.where(xr.DataArray(pc_max == pc_min, dims="pc", coords={"pc": pcs_full["pc"]}), 0.5, pcs_norm)

        rgb_da = pcs_norm.assign_coords(pc=["R", "G", "B"]).rename(pc="rgb")
        return rgb_da.transpose("time", "xc", "yc", "rgb")

    @property
    def output_var(self):
        return "log_y"

    @property
    def _serializable_model_config(self) -> Dict:
        return self.model_config

    def _save_input_params(self, idata) -> None:
        idata.attrs["config"] = jt.dumps(self._config)
        idata.attrs["metadata"] = jt.dumps(self.metadata)
        idata.attrs["training_metadata"] = jt.dumps(self.training_metadata)
        if self.prediction_metadata is not None:
            idata.attrs["prediction_metadata"] = jt.dumps(self.prediction_metadata)

    def sample_posterior_predictive(self, X_pred, extend_idata, combined, y_pred_obs=None, **kwargs):
        print("\nUsing overridden sample_posterior_predictive!")
        self._data_setter(X_pred, y_pred_obs)
        with self.prediction_model:
            post_pred = pm.sample_posterior_predictive(self.idata, var_names=self.metadata["var_names"], **kwargs)
            if extend_idata:
                self.idata.extend(post_pred, join="right")
        group_name = "predictions" if kwargs.get("predictions", False) else "posterior_predictive"
        posterior_predictive_samples = az.extract(post_pred, group_name, combined=combined)
        print("Saving idata")
        return posterior_predictive_samples

    @property
    def config(self):
        return self._config.copy()
