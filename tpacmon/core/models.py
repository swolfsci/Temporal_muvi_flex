"""Core model classes for tpacmon: TemporalModel, TemporalGuide, TemporalPACMON."""

import logging
import math
from functools import partial
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.distributions.constraints as constraints

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroParam
from pyro.optim import Adam, ClippedAdam
from tqdm import trange

from tpacmon.core.kernels import build_kernel, KERNEL_REGISTRY
from tpacmon.core.early_stopping import EarlyStoppingCallback
from tpacmon.core.callbacks import CheckpointCallback

logger = logging.getLogger(__name__)

# Type aliases
SingleView = Union[np.ndarray, pd.DataFrame]
MultiView = Union[Dict[str, SingleView], List[SingleView]]


# ---------------------------------------------------------------------------
# TemporalModel — generative model
# ---------------------------------------------------------------------------
class TemporalModel(PyroModule):
    def __init__(
        self,
        n_patients: int,
        n_features: List[int],
        n_sparse_factors: int,
        n_dense_factors: int,
        prior_scales: Optional[torch.Tensor],
        n_covariates: int,
        time_points: torch.Tensor,
        patient_masks: torch.Tensor,
        kernel: str = "matern32",
        likelihoods: Optional[List[str]] = None,
        global_prior_scale: float = 1.0,
        reg_hs: bool = True,
        scale_elbo: bool = True,
        gp_scale: float = 1.0,
        device=None,
    ):
        super().__init__(name="TemporalModel")
        self.n_patients = n_patients
        self.n_features = n_features
        self.feature_offsets = [0, *np.cumsum(self.n_features).tolist()]
        self.n_views = len(self.n_features)
        self.n_sparse_factors = n_sparse_factors
        self.n_dense_factors = n_dense_factors
        self.n_covariates = n_covariates
        self.kernel_name = kernel
        self.gp_scale = gp_scale
        self.device = device

        # Time structure
        self.time_points = time_points.to(device)
        self.patient_masks = patient_masks.to(device)
        self.n_timepoints = time_points.shape[0]
        # Per-patient observed counts
        self.n_obs_per_patient = patient_masks.sum(dim=1).long()
        self.max_obs = int(self.n_obs_per_patient.max().item())

        # Prior scales
        self.prior_scales = prior_scales
        if self.prior_scales is not None:
            self.prior_scales = self.prior_scales.to(device)
            self.n_informed_factors = self.prior_scales.shape[0]
        else:
            self.n_informed_factors = 0
        self.n_factors = self.n_informed_factors + self.n_dense_factors

        # Likelihoods
        if likelihoods is None:
            likelihoods = ["normal"] * self.n_views
        self.likelihoods = likelihoods

        self.global_prior_scale = global_prior_scale
        self.reg_hs = reg_hs

        # View scaling for ELBO
        self.scale_elbo = scale_elbo
        self.view_scales = np.ones(self.n_views)
        if self.scale_elbo and self.n_views > 1:
            alpha = 0.25
            raw_weights = np.array([1.0 / (nf ** alpha) for nf in n_features])
            self.view_scales = self.n_views * raw_weights / raw_weights.sum()

        # Total number of observation rows (patients * their observed timepoints)
        self.n_obs_total = int(self.n_obs_per_patient.sum().item())

    def _zeros(self, size):
        return torch.zeros(size, device=self.device)

    def _ones(self, size):
        return torch.ones(size, device=self.device)

    def forward(self, obs: torch.Tensor, obs_mask: torch.Tensor,
                patient_idx: torch.Tensor, time_idx: torch.Tensor,
                covs: Optional[torch.Tensor] = None):
        output_dict = {}
        n_obs = obs.shape[0]

        # --- Factor loading plates ---
        if self.n_informed_factors > 0:
            factor_ls_plate = pyro.plate("factor_left_sparse", self.n_informed_factors, dim=-2, device=self.device)
        if self.n_dense_factors > 0:
            factor_ld_plate = pyro.plate("factor_left_dense", self.n_dense_factors, dim=-2, device=self.device)
        feature_plates = [
            pyro.plate(f"feature_{m}", self.n_features[m], dim=-1, device=self.device)
            for m in range(self.n_views)
        ]

        # --- Horseshoe prior on loadings (from PACMon, unchanged) ---
        if self.n_informed_factors > 0:
            view_plate = pyro.plate("view", self.n_views, dim=-1, device=self.device)
            with view_plate:
                output_dict["view_scale"] = pyro.sample(
                    "view_scale", dist.HalfCauchy(self._ones((1,)))
                )
                with factor_ls_plate:
                    output_dict["factor_scale"] = pyro.sample(
                        "factor_scale", dist.HalfCauchy(self._ones((1,)))
                    )

        for m in range(self.n_views):
            with feature_plates[m]:
                if self.n_informed_factors > 0:
                    with factor_ls_plate:
                        output_dict[f"local_scale_{m}"] = pyro.sample(
                            f"local_scale_{m}", dist.HalfCauchy(self._ones((1,)))
                        )
                        w_scale = (
                            output_dict[f"local_scale_{m}"]
                            * output_dict["factor_scale"][..., m:m+1]
                            * output_dict["view_scale"][..., m:m+1]
                        )
                        if self.reg_hs:
                            output_dict[f"caux_{m}"] = pyro.sample(
                                f"caux_{m}",
                                dist.InverseGamma(0.5 * self._ones((1,)), 0.5 * self._ones((1,)))
                            )
                            c = torch.sqrt(output_dict[f"caux_{m}"])
                            if self.prior_scales is not None:
                                c = c * self.prior_scales[:, self.feature_offsets[m]:self.feature_offsets[m+1]]
                            w_scale = (self.global_prior_scale * c * w_scale) / torch.sqrt(c**2 + w_scale**2)

                        output_dict[f"w_sparse_{m}"] = pyro.sample(
                            f"w_sparse_{m}", dist.Normal(self._zeros((1,)), w_scale)
                        )
                        output_dict[f"w_{m}"] = output_dict[f"w_sparse_{m}"]

                if self.n_dense_factors > 0:
                    with factor_ld_plate:
                        output_dict[f"w_dense_{m}"] = pyro.sample(
                            f"w_dense_{m}", dist.Normal(self._zeros((1,)), self._ones((1,)))
                        )
                    if f"w_sparse_{m}" in output_dict:
                        output_dict[f"w_{m}"] = torch.cat(
                            [output_dict[f"w_sparse_{m}"], output_dict[f"w_dense_{m}"]], dim=-2
                        )
                    else:
                        output_dict[f"w_{m}"] = output_dict[f"w_dense_{m}"]

                # Beta (covariate regression on features)
                if self.n_covariates > 0:
                    covariate_plate = pyro.plate(f"covariate_{m}", self.n_covariates, dim=-2, device=self.device)
                    with covariate_plate:
                        output_dict[f"beta_{m}"] = pyro.sample(
                            f"beta_{m}", dist.Normal(self._zeros((1,)), self._ones((1,)))
                        )

                output_dict[f"sigma_{m}"] = pyro.sample(
                    f"sigma_{m}", dist.LogNormal(self._zeros((1,)), self._ones((1,)))
                )

        # --- GP kernel hyperparameters (per factor) ---
        factor_plate = pyro.plate("factor", self.n_factors, device=self.device)
        with factor_plate:
            output_dict["lengthscale"] = pyro.sample(
                "lengthscale", dist.LogNormal(self._zeros((1,)), self._ones((1,)))
            )
            output_dict["amplitude"] = pyro.sample(
                "amplitude", dist.LogNormal(self._zeros((1,)), 0.5 * self._ones((1,)))
            )
            output_dict["zeta"] = pyro.sample(
                "zeta", dist.Beta(self._ones((1,)), self._ones((1,)))
            )

        # --- Gamma (covariate effect on factor-level GP mean) ---
        if self.n_covariates > 0:
            gamma_plate_cov = pyro.plate("gamma_cov", self.n_covariates, dim=-2, device=self.device)
            gamma_plate_fac = pyro.plate("gamma_fac", self.n_factors, dim=-1, device=self.device)
            with gamma_plate_cov:
                with gamma_plate_fac:
                    output_dict["gamma"] = pyro.sample(
                        "gamma", dist.Normal(self._zeros((1,)), self._ones((1,)))
                    )

        # --- Sample z per patient via GP prior ---
        # Build z for all observations: shape (n_obs,) indexes into patients/times
        z_all = self._zeros((n_obs, self.n_factors))

        lengthscales = output_dict["lengthscale"].squeeze()
        amplitudes = output_dict["amplitude"].squeeze()
        zetas = output_dict["zeta"].squeeze()

        # Unique patients in this batch
        unique_patients = torch.unique(patient_idx)

        for p in unique_patients:
            p = p.item()
            mask_p = self.patient_masks[p].bool()
            t_obs = self.time_points[mask_p]
            n_t = t_obs.shape[0]

            # Which rows in the obs tensor belong to this patient
            obs_rows = (patient_idx == p).nonzero(as_tuple=True)[0]

            for k in range(self.n_factors):
                ls_k = lengthscales[k] if self.n_factors > 1 else lengthscales
                amp_k = amplitudes[k] if self.n_factors > 1 else amplitudes
                zeta_k = zetas[k] if self.n_factors > 1 else zetas

                # GP mean (covariate-shifted)
                if self.n_covariates > 0 and covs is not None:
                    gp_mean = (covs[p] @ output_dict["gamma"][:, k]).expand(n_t)
                else:
                    gp_mean = self._zeros((n_t,))

                # Build kernel
                K = build_kernel(self.kernel_name, t_obs, ls_k, amp_k, jitter=1e-5)

                # GP component
                with pyro.poutine.scale(scale=self.gp_scale):
                    f_pk = pyro.sample(
                        f"f_{p}_{k}",
                        dist.MultivariateNormal(gp_mean, covariance_matrix=K)
                    )

                # i.i.d. component
                eta_pk = pyro.sample(
                    f"eta_{p}_{k}",
                    dist.Normal(self._zeros((n_t,)), self._ones((n_t,))).to_event(1)
                )

                # Mix: z = sqrt(1-zeta)*f + sqrt(zeta)*eta
                z_pk = torch.sqrt(1 - zeta_k) * f_pk + torch.sqrt(zeta_k) * eta_pk
                z_all[obs_rows, k] = z_pk

        output_dict["z"] = z_all

        # --- Observation likelihood ---
        obs_plate = pyro.plate("obs", n_obs, dim=-2, device=self.device)
        with obs_plate:
            for m in range(self.n_views):
                with feature_plates[m]:
                    y_loc = torch.matmul(z_all, output_dict[f"w_{m}"])
                    if self.n_covariates > 0 and covs is not None:
                        # Expand covs to match obs rows (covs indexed by patient)
                        covs_expanded = covs[patient_idx]
                        y_loc = y_loc + torch.matmul(covs_expanded, output_dict[f"beta_{m}"])

                    feature_idx = slice(self.feature_offsets[m], self.feature_offsets[m+1])
                    if self.likelihoods[m] == "normal":
                        y_dist = dist.Normal(y_loc, output_dict[f"sigma_{m}"])
                    else:
                        y_dist = dist.Bernoulli(logits=y_loc)

                    with pyro.poutine.scale(scale=self.view_scales[m]):
                        with pyro.poutine.mask(mask=obs_mask[..., feature_idx]):
                            output_dict[f"y_{m}"] = pyro.sample(
                                f"y_{m}", y_dist,
                                obs=obs[..., feature_idx],
                                infer={"is_auxiliary": True},
                            )

        return output_dict


# ---------------------------------------------------------------------------
# TemporalGuide — variational posterior
# ---------------------------------------------------------------------------
class TemporalGuide(PyroModule):
    def __init__(self, model: TemporalModel, init_loc: float = 0.0, init_scale: float = 0.1):
        super().__init__(name="TemporalGuide")
        self.model = model
        self.locs = PyroModule()
        self.scales = PyroModule()
        self.init_loc = init_loc
        self.init_scale = init_scale
        self.site_to_dist = self._setup()

    def _setup(self):
        n_views = self.model.n_views
        n_factors = self.model.n_factors
        n_informed = self.model.n_informed_factors
        n_dense = self.model.n_dense_factors
        n_features = self.model.n_features
        n_covariates = self.model.n_covariates
        n_patients = self.model.n_patients

        # --- Standard sites (same as PACMon) ---
        site_to_shape = {
            "view_scale": (n_views,),
            "factor_scale": (n_informed, n_views),
            "lengthscale": (n_factors,),
            "amplitude": (n_factors,),
            "zeta": (n_factors,),
        }

        normal_sites = []
        for m in range(n_views):
            site_to_shape[f"local_scale_{m}"] = (n_informed, n_features[m])
            site_to_shape[f"caux_{m}"] = (n_informed, n_features[m])
            if n_informed > 0:
                site_to_shape[f"w_sparse_{m}"] = (n_informed, n_features[m])
                normal_sites.append(f"w_sparse_{m}")
            if n_dense > 0:
                site_to_shape[f"w_dense_{m}"] = (n_dense, n_features[m])
                normal_sites.append(f"w_dense_{m}")
            site_to_shape[f"beta_{m}"] = (n_covariates, n_features[m])
            site_to_shape[f"sigma_{m}"] = (n_features[m],)
            normal_sites.append(f"beta_{m}")

        if n_covariates > 0:
            site_to_shape["gamma"] = (n_covariates, n_factors)
            normal_sites.append("gamma")

        site_to_dist = {
            k: "Normal" if k in normal_sites else "LogNormal"
            for k in site_to_shape
        }
        # zeta uses Beta — handle separately
        site_to_dist["zeta"] = "Beta"

        # Register standard params
        for name, shape in site_to_shape.items():
            if name == "zeta":
                # Beta guide: parameterize via concentration
                setattr(self.locs, name, PyroParam(
                    2.0 * self.model._ones(shape), constraints.positive
                ))
                setattr(self.scales, name, PyroParam(
                    2.0 * self.model._ones(shape), constraints.positive
                ))
            else:
                setattr(self.locs, name, PyroParam(
                    self.init_loc * self.model._ones(shape), constraints.real
                ))
                setattr(self.scales, name, PyroParam(
                    self.init_scale * self.model._ones(shape), constraints.positive
                ))

        # --- Per-patient per-factor GP guide parameters ---
        # f_{p}_{k} gets MVN guide with learned mean + Cholesky
        # We store padded tensors: (n_patients, max_obs, n_factors) for means
        # and (n_patients, max_obs, max_obs, n_factors) for Cholesky factors
        max_obs = self.model.max_obs
        self.z_mean = PyroParam(
            torch.zeros(n_patients, max_obs, n_factors, device=self.model.device),
            constraints.real,
        )
        self.z_scale_tril = PyroParam(
            0.1 * torch.eye(max_obs, device=self.model.device).unsqueeze(0).unsqueeze(-1).expand(
                n_patients, max_obs, max_obs, n_factors
            ).clone(),
            constraints.real,  # We'll construct L @ L^T manually
        )
        # eta_{p}_{k} gets diagonal Normal guide
        self.eta_mean = PyroParam(
            torch.zeros(n_patients, max_obs, n_factors, device=self.model.device),
            constraints.real,
        )
        self.eta_scale = PyroParam(
            0.1 * torch.ones(n_patients, max_obs, n_factors, device=self.model.device),
            constraints.positive,
        )

        self.site_to_shape = site_to_shape
        return site_to_dist

    def _sample_standard(self, name: str):
        loc = getattr(self.locs, name)
        scale = getattr(self.scales, name)
        dist_type = self.site_to_dist[name]
        if dist_type == "Normal":
            return pyro.sample(name, dist.Normal(loc, scale))
        elif dist_type == "LogNormal":
            return pyro.sample(name, dist.LogNormal(loc, scale))
        elif dist_type == "Beta":
            return pyro.sample(name, dist.Beta(loc, scale))

    @torch.no_grad()
    def mode(self, name: str):
        loc = getattr(self.locs, name)
        scale = getattr(self.scales, name)
        if self.site_to_dist[name] == "LogNormal":
            return (loc - scale.square()).exp().cpu().numpy()
        elif self.site_to_dist[name] == "Beta":
            a, b = loc, scale
            return ((a - 1) / (a + b - 2)).clamp(0, 1).cpu().numpy()
        return loc.clone().cpu().numpy()

    def get_w(self, as_list: bool = False):
        ws = []
        for m in range(self.model.n_views):
            parts = []
            if self.model.n_informed_factors > 0:
                parts.append(self.mode(f"w_sparse_{m}"))
            if self.model.n_dense_factors > 0:
                parts.append(self.mode(f"w_dense_{m}"))
            ws.append(np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0])
        if as_list:
            return ws
        return np.concatenate(ws, axis=1)

    def get_beta(self, as_list: bool = False):
        betas = [self.mode(f"beta_{m}") for m in range(self.model.n_views)]
        if as_list:
            return betas
        return np.concatenate(betas, axis=1)

    def get_sigma(self, as_list: bool = False):
        sigmas = [self.mode(f"sigma_{m}") for m in range(self.model.n_views)]
        if as_list:
            return sigmas
        return np.concatenate(sigmas, axis=0)

    def get_lengthscales(self):
        return self.mode("lengthscale").squeeze()

    def get_amplitudes(self):
        return self.mode("amplitude").squeeze()

    def get_zeta(self):
        return self.mode("zeta").squeeze()

    def get_z(self):
        """Return factor scores as (n_patients, max_obs, n_factors) numpy array."""
        return self.z_mean.detach().cpu().numpy()

    def forward(self, obs, obs_mask, patient_idx, time_idx, covs=None):
        output_dict = {}

        # --- Standard sites ---
        if self.model.n_informed_factors > 0:
            view_plate = pyro.plate("view", self.model.n_views, dim=-1, device=self.model.device)
            factor_ls_plate = pyro.plate("factor_left_sparse", self.model.n_informed_factors, dim=-2, device=self.model.device)
            with view_plate:
                output_dict["view_scale"] = self._sample_standard("view_scale")
                with factor_ls_plate:
                    output_dict["factor_scale"] = self._sample_standard("factor_scale")

        feature_plates = [
            pyro.plate(f"feature_{m}", self.model.n_features[m], dim=-1, device=self.model.device)
            for m in range(self.model.n_views)
        ]
        if self.model.n_dense_factors > 0:
            factor_ld_plate = pyro.plate("factor_left_dense", self.model.n_dense_factors, dim=-2, device=self.model.device)

        for m in range(self.model.n_views):
            with feature_plates[m]:
                if self.model.n_informed_factors > 0:
                    with factor_ls_plate:
                        output_dict[f"local_scale_{m}"] = self._sample_standard(f"local_scale_{m}")
                        if self.model.reg_hs:
                            output_dict[f"caux_{m}"] = self._sample_standard(f"caux_{m}")
                        output_dict[f"w_sparse_{m}"] = self._sample_standard(f"w_sparse_{m}")
                if self.model.n_dense_factors > 0:
                    with factor_ld_plate:
                        output_dict[f"w_dense_{m}"] = self._sample_standard(f"w_dense_{m}")
                if self.model.n_covariates > 0:
                    cov_plate = pyro.plate(f"covariate_{m}", self.model.n_covariates, dim=-2, device=self.model.device)
                    with cov_plate:
                        output_dict[f"beta_{m}"] = self._sample_standard(f"beta_{m}")
                output_dict[f"sigma_{m}"] = self._sample_standard(f"sigma_{m}")

        # --- GP hyperparameters ---
        factor_plate = pyro.plate("factor", self.model.n_factors, device=self.model.device)
        with factor_plate:
            output_dict["lengthscale"] = self._sample_standard("lengthscale")
            output_dict["amplitude"] = self._sample_standard("amplitude")
            output_dict["zeta"] = self._sample_standard("zeta")

        # --- Gamma ---
        if self.model.n_covariates > 0:
            gamma_plate_cov = pyro.plate("gamma_cov", self.model.n_covariates, dim=-2, device=self.model.device)
            gamma_plate_fac = pyro.plate("gamma_fac", self.model.n_factors, dim=-1, device=self.model.device)
            with gamma_plate_cov:
                with gamma_plate_fac:
                    output_dict["gamma"] = self._sample_standard("gamma")

        # --- Per-patient GP factors (structured MVN guide) ---
        unique_patients = torch.unique(patient_idx)
        for p in unique_patients:
            p_int = p.item()
            n_t = self.model.n_obs_per_patient[p_int].item()

            for k in range(self.model.n_factors):
                # f_{p}_{k}: MVN with learned Cholesky
                mu = self.z_mean[p_int, :n_t, k]
                L_raw = self.z_scale_tril[p_int, :n_t, :n_t, k]
                L = torch.tril(L_raw)
                # Ensure positive diagonal
                L = L - torch.diag(torch.diag(L)) + torch.diag(torch.diag(L).abs().clamp(min=1e-6))
                pyro.sample(
                    f"f_{p_int}_{k}",
                    dist.MultivariateNormal(mu, scale_tril=L)
                )

                # eta_{p}_{k}: diagonal Normal
                eta_mu = self.eta_mean[p_int, :n_t, k]
                eta_s = self.eta_scale[p_int, :n_t, k]
                pyro.sample(
                    f"eta_{p_int}_{k}",
                    dist.Normal(eta_mu, eta_s).to_event(1)
                )

        return output_dict


# ---------------------------------------------------------------------------
# TemporalPACMON — main user-facing class
# ---------------------------------------------------------------------------
class TemporalPACMON:
    def __init__(
        self,
        observations: Dict[str, np.ndarray],
        time_points: np.ndarray,
        patient_masks: np.ndarray,
        prior_masks: Optional[dict] = None,
        covariates: Optional[np.ndarray] = None,
        prior_confidence: Union[float, str] = "low",
        n_sparse_factors: Optional[int] = None,
        n_dense_factors: Optional[int] = None,
        kernel: str = "matern32",
        likelihoods: Optional[Dict[str, str]] = None,
        normalize: bool = True,
        shared_lengthscale: bool = False,
        guide_type: str = "cholesky",
        gp_scale: float = 1.0,
        double_precision: bool = False,
        device: str = "auto",
    ):
        # Device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.kernel = kernel
        self.guide_type = guide_type
        self.gp_scale = gp_scale
        self.double_precision = double_precision
        self.normalize = normalize
        self._trained = False

        # Store raw inputs
        self.time_points_np = np.asarray(time_points, dtype=np.float32)
        self.patient_masks_np = np.asarray(patient_masks, dtype=bool)
        self.n_patients = self.patient_masks_np.shape[0]
        self.n_timepoints = self.time_points_np.shape[0]

        # Setup observations
        self.view_names = list(observations.keys())
        self.observations, self.feature_names = self._setup_observations(observations)
        self.n_features = {vn: obs.shape[-1] for vn, obs in self.observations.items()}

        # Setup covariates
        self.covariates = None
        self.n_covariates = 0
        if covariates is not None:
            self.covariates = np.asarray(covariates, dtype=np.float32)
            self.n_covariates = self.covariates.shape[1]

        # Setup priors
        self.prior_confidence = self._setup_prior_confidence(prior_confidence)
        self.prior_masks, self.prior_scales = self._setup_prior_masks(
            prior_masks, n_sparse_factors
        )
        self.n_sparse_factors = 0 if self.prior_scales is None else self.prior_scales.shape[0]
        self.n_dense_factors = n_dense_factors or 0
        self.n_factors = self.n_sparse_factors + self.n_dense_factors
        if self.n_factors == 0:
            raise ValueError("Must specify at least one factor via prior_masks or n_dense_factors.")

        # Factor naming
        factor_names = []
        if self.prior_masks is not None:
            vn0 = self.view_names[0]
            if isinstance(self.prior_masks[vn0], pd.DataFrame):
                factor_names.extend(self.prior_masks[vn0].index.tolist())
            else:
                factor_names.extend([f"sparse_{k}" for k in range(self.n_sparse_factors)])
        factor_names.extend([f"dense_{k}" for k in range(self.n_dense_factors)])
        self.factor_names = factor_names

        # Likelihoods
        if likelihoods is None:
            self.likelihoods = {vn: "normal" for vn in self.view_names}
        else:
            self.likelihoods = likelihoods

        # Internal state
        self._model = None
        self._guide = None
        self._training_history = []

    def _setup_observations(self, observations):
        """Parse observations dict, extract feature names, optionally normalize."""
        parsed = {}
        feature_names = {}
        for vn, obs in observations.items():
            if isinstance(obs, pd.DataFrame):
                feature_names[vn] = obs.columns.tolist()
                arr = obs.to_numpy(dtype=np.float32)
            else:
                arr = np.asarray(obs, dtype=np.float32)
                feature_names[vn] = [f"{vn}_{j}" for j in range(arr.shape[-1])]
            # Expect shape (P, T, D) or (P, D) for non-temporal
            if arr.ndim == 2:
                # Expand: (P, D) -> (P, 1, D) assuming single timepoint
                arr = arr[:, np.newaxis, :]
            if self.normalize:
                # Normalize per feature across all non-NaN observations
                reshaped = arr.reshape(-1, arr.shape[-1])
                col_mean = np.nanmean(reshaped, axis=0)
                col_std = np.nanstd(reshaped, axis=0)
                col_std[col_std == 0] = 1.0
                arr = (arr - col_mean) / col_std
            parsed[vn] = arr
        return parsed, feature_names

    def _setup_prior_confidence(self, prior_confidence):
        mapping = {"low": 0.99, "med": 0.995, "high": 0.999}
        if isinstance(prior_confidence, str):
            if prior_confidence.lower() not in mapping:
                raise KeyError("Invalid prior confidence. Use 'low', 'med', 'high' or a float.")
            return mapping[prior_confidence.lower()]
        if not (0 < prior_confidence < 1.0):
            raise ValueError("prior_confidence must be between 0 and 1.")
        return prior_confidence

    def _setup_prior_masks(self, masks, n_sparse_factors):
        if masks is None:
            return None, None

        # Convert to numpy arrays
        processed = {}
        for vn in self.view_names:
            if vn in masks:
                m = masks[vn]
                if isinstance(m, pd.DataFrame):
                    processed[vn] = m.to_numpy(dtype=np.float32)
                else:
                    processed[vn] = np.asarray(m, dtype=np.float32)
            else:
                # Uninformed view
                n_factors = next(iter(masks.values())).shape[0] if isinstance(next(iter(masks.values())), (np.ndarray, pd.DataFrame)) else 0
                processed[vn] = np.zeros((n_factors, self.n_features[vn]), dtype=np.float32)

        # Compute prior scales: clip(mask + (1 - confidence), 1e-8, 1.0)
        prior_scales = {}
        for vn, vm in processed.items():
            prior_scales[vn] = np.clip(
                vm.astype(np.float32) + (1.0 - self.prior_confidence), 1e-8, 1.0
            )

        # Concatenate across views for the model
        scales_list = [torch.tensor(prior_scales[vn]) for vn in self.view_names]
        scales_cat = torch.cat(scales_list, dim=1)

        return processed, scales_cat

    def _flatten_observations(self):
        """Flatten (P, T, D) observations to (N_obs, D_total) with tracking indices."""
        obs_list = []
        mask_list = []
        patient_idx_list = []
        time_idx_list = []

        for p in range(self.n_patients):
            for t in range(self.n_timepoints):
                if self.patient_masks_np[p, t]:
                    row = []
                    row_mask = []
                    for vn in self.view_names:
                        data_pt = self.observations[vn][p, t, :]
                        valid = ~np.isnan(data_pt)
                        data_pt = np.nan_to_num(data_pt, nan=0.0)
                        row.append(data_pt)
                        row_mask.append(valid)
                    obs_list.append(np.concatenate(row))
                    mask_list.append(np.concatenate(row_mask))
                    patient_idx_list.append(p)
                    time_idx_list.append(t)

        obs = torch.tensor(np.array(obs_list), dtype=torch.float32, device=self.device)
        obs_mask = torch.tensor(np.array(mask_list), dtype=torch.bool, device=self.device)
        patient_idx = torch.tensor(patient_idx_list, dtype=torch.long, device=self.device)
        time_idx = torch.tensor(time_idx_list, dtype=torch.long, device=self.device)
        return obs, obs_mask, patient_idx, time_idx

    def _setup_model_guide(self):
        time_points_t = torch.tensor(self.time_points_np, dtype=torch.float32, device=self.device)
        patient_masks_t = torch.tensor(self.patient_masks_np, dtype=torch.float32, device=self.device)

        n_features_list = [self.n_features[vn] for vn in self.view_names]
        likelihoods_list = [self.likelihoods[vn] for vn in self.view_names]

        self._model = TemporalModel(
            n_patients=self.n_patients,
            n_features=n_features_list,
            n_sparse_factors=self.n_sparse_factors,
            n_dense_factors=self.n_dense_factors,
            prior_scales=self.prior_scales,
            n_covariates=self.n_covariates,
            time_points=time_points_t,
            patient_masks=patient_masks_t,
            kernel=self.kernel,
            likelihoods=likelihoods_list,
            gp_scale=self.gp_scale,
            device=self.device,
        )
        self._guide = TemporalGuide(self._model)

    def fit(
        self,
        n_epochs: int = 1000,
        learning_rate: float = 0.01,
        optimizer: str = "clipped",
        early_stopping: bool = True,
        min_epochs: int = 100,
        tolerance: float = 1e-5,
        patience: int = 10,
        clip_norm: float = 10.0,
        seed: int = 0,
    ):
        """Train the model via SVI."""
        pyro.clear_param_store()
        pyro.set_rng_seed(seed)

        self._setup_model_guide()

        # Flatten observations
        obs, obs_mask, patient_idx, time_idx = self._flatten_observations()
        covs = None
        if self.covariates is not None:
            covs = torch.tensor(self.covariates, dtype=torch.float32, device=self.device)

        # Optimizer
        if optimizer == "clipped":
            n_iterations = n_epochs
            gamma = 0.1
            lrd = gamma ** (1 / max(n_iterations, 1))
            opt = ClippedAdam({"lr": learning_rate, "lrd": lrd})
        else:
            opt = Adam({"lr": learning_rate, "betas": (0.95, 0.999)})

        # SVI
        loss_fn = pyro.infer.Trace_ELBO(num_particles=1, vectorize_particles=False)
        svi = pyro.infer.SVI(
            model=self._model,
            guide=self._guide,
            optim=opt,
            loss=loss_fn,
        )

        # Early stopping
        es_callback = None
        if early_stopping:
            es_callback = EarlyStoppingCallback(
                min_epochs=min_epochs, tolerance=tolerance, patience=patience
            )

        # Training loop
        history = []
        pbar = trange(n_epochs, desc="Training")
        for epoch in pbar:
            loss = svi.step(obs, obs_mask, patient_idx, time_idx, covs)

            # Gradient clipping
            if clip_norm > 0:
                params = [p for p in pyro.get_param_store().values() if p.requires_grad]
                if params:
                    torch.nn.utils.clip_grad_norm_(params, clip_norm)

            history.append(loss)
            pbar.set_postfix({"ELBO": f"{loss:.2f}"})

            if es_callback and es_callback(history):
                break

        self._training_history = history
        self._trained = True
        logger.info(f"Training complete. Final ELBO: {history[-1]:.2f}")

    def get_factors(self, as_df: bool = False):
        """Return factor scores as (P, T_max, K) array. NaN for unobserved."""
        z = self._guide.get_z()  # (P, max_obs, K)
        # Expand to full grid
        result = np.full((self.n_patients, self.n_timepoints, self.n_factors), np.nan)
        for p in range(self.n_patients):
            obs_idx = np.where(self.patient_masks_np[p])[0]
            n_obs = len(obs_idx)
            result[p, obs_idx, :] = z[p, :n_obs, :]
        if as_df:
            dfs = {}
            for k, fn in enumerate(self.factor_names):
                dfs[fn] = pd.DataFrame(
                    result[:, :, k],
                    columns=[f"t={t:.2f}" for t in self.time_points_np],
                )
            return dfs
        return result

    def get_loadings(self, as_df: bool = False):
        """Return factor loadings per view."""
        ws = self._guide.get_w(as_list=True)
        if as_df:
            return {
                vn: pd.DataFrame(ws[i], index=self.factor_names, columns=self.feature_names[vn])
                for i, vn in enumerate(self.view_names)
            }
        return {vn: ws[i] for i, vn in enumerate(self.view_names)}

    def get_lengthscales(self):
        return self._guide.get_lengthscales()

    def get_smoothness(self):
        """Return zeta_k per factor (0=temporal, 1=i.i.d.)."""
        return self._guide.get_zeta()

    def get_covariate_coefficients(self, as_df: bool = False):
        betas = self._guide.get_beta(as_list=True)
        if as_df:
            return {
                vn: pd.DataFrame(betas[i], columns=self.feature_names[vn])
                for i, vn in enumerate(self.view_names)
            }
        return {vn: betas[i] for i, vn in enumerate(self.view_names)}

    def predict(self, new_time_points: np.ndarray):
        """GP posterior prediction at unobserved times.

        Returns dict with 'mean' (P, T_new, K) and 'variance' (P, T_new, K).
        """
        if not self._trained:
            raise RuntimeError("Model must be trained before prediction.")

        new_t = torch.tensor(new_time_points, dtype=torch.float32, device=self.device)
        lengthscales = torch.tensor(self._guide.get_lengthscales(), dtype=torch.float32, device=self.device)
        amplitudes = torch.tensor(self._guide.get_amplitudes(), dtype=torch.float32, device=self.device)
        z_means = self._guide.z_mean.detach()  # (P, max_obs, K)

        time_t = torch.tensor(self.time_points_np, dtype=torch.float32, device=self.device)

        means = np.zeros((self.n_patients, len(new_time_points), self.n_factors))
        variances = np.zeros_like(means)

        with torch.no_grad():
            for p in range(self.n_patients):
                mask_p = self.patient_masks_np[p].astype(bool)
                t_obs = time_t[mask_p]
                n_obs = t_obs.shape[0]

                for k in range(self.n_factors):
                    ls_k = lengthscales[k] if self.n_factors > 1 else lengthscales
                    amp_k = amplitudes[k] if self.n_factors > 1 else amplitudes

                    K_obs = build_kernel(self.kernel, t_obs, ls_k, amp_k, jitter=1e-5)
                    K_new_obs = build_kernel(self.kernel, new_t, ls_k, amp_k, jitter=0.0, t_obs2=t_obs)
                    K_new = build_kernel(self.kernel, new_t, ls_k, amp_k, jitter=1e-5)

                    K_inv = torch.linalg.solve(K_obs, torch.eye(n_obs, device=self.device))
                    z_obs = z_means[p, :n_obs, k]

                    mu_pred = K_new_obs @ K_inv @ z_obs
                    var_pred = torch.diag(K_new - K_new_obs @ K_inv @ K_new_obs.T).clamp(min=0)

                    means[p, :, k] = mu_pred.cpu().numpy()
                    variances[p, :, k] = var_pred.cpu().numpy()

        return {"mean": means, "variance": variances}

    @property
    def training_history(self):
        return self._training_history
