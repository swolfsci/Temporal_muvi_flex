"""Core model classes for tpacmon: TemporalModel, TemporalGuide, TemporalPACMON."""

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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


def _group_patients_by_mask(patient_masks: torch.Tensor):
    """Group patients that share the same observation mask.

    Returns list of (mask_bool, patient_indices) tuples.
    With 4 timepoints there are at most 16 possible patterns.
    """
    groups = {}
    for p in range(patient_masks.shape[0]):
        key = tuple(patient_masks[p].bool().tolist())
        if key not in groups:
            groups[key] = []
        groups[key].append(p)
    return [(torch.tensor(list(k), dtype=torch.bool),
             torch.tensor(v, dtype=torch.long))
            for k, v in groups.items()]


# ---------------------------------------------------------------------------
# TemporalModel — generative model (vectorized)
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
        self.n_obs_per_patient = patient_masks.sum(dim=1).long()
        self.max_obs = int(self.n_obs_per_patient.max().item())

        # Pre-compute patient groups by mask pattern
        self.mask_groups = _group_patients_by_mask(patient_masks)

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

        # --- Vectorized GP sampling per mask group ---
        # Instead of P*K individual samples, we sample per group:
        # one MVN sample per group with shape (n_group_patients, K, T_obs)
        z_all = self._zeros((n_obs, self.n_factors))

        lengthscales = output_dict["lengthscale"].squeeze()
        amplitudes = output_dict["amplitude"].squeeze()
        zetas = output_dict["zeta"].squeeze()

        for g_idx, (mask_bool, group_pats) in enumerate(self.mask_groups):
            mask_bool = mask_bool.to(self.device)
            group_pats = group_pats.to(self.device)
            t_obs = self.time_points[mask_bool]
            n_t = t_obs.shape[0]
            n_group = group_pats.shape[0]

            # Build kernel matrices for all factors: (K, T, T)
            if self.n_factors > 1:
                K_all = torch.stack([
                    build_kernel(self.kernel_name, t_obs, lengthscales[k], amplitudes[k], jitter=1e-5)
                    for k in range(self.n_factors)
                ])  # (K, T, T)
            else:
                K_all = build_kernel(self.kernel_name, t_obs, lengthscales, amplitudes, jitter=1e-5).unsqueeze(0)

            # GP mean: (n_group, K, T)
            if self.n_covariates > 0 and covs is not None:
                gp_means_scalar = covs[group_pats] @ output_dict["gamma"]  # (n_group, K)
                gp_means = gp_means_scalar.unsqueeze(-1).expand(n_group, self.n_factors, n_t)
            else:
                gp_means = self._zeros((n_group, self.n_factors, n_t))

            # Sample f: batch over (patients, factors) with just 1 pyro.sample call
            # Use Cholesky of each K_k to reparameterize: f = mu + L @ eps
            # where eps ~ Normal(0, I), so we sample eps as (n_group, K, T)
            L_all = torch.linalg.cholesky(K_all)  # (K, T, T)

            group_plate = pyro.plate(f"group_{g_idx}", n_group, dim=-1)

            with pyro.poutine.scale(scale=self.gp_scale):
                with group_plate:
                    f_eps = pyro.sample(
                        f"f_g{g_idx}",
                        dist.Normal(
                            self._zeros((n_group, self.n_factors, n_t)),
                            self._ones((n_group, self.n_factors, n_t)),
                        ).to_event(2),  # event = (K, T), batch = (n_group,)
                    )  # (n_group, K, T)

            # f = mu + L @ eps  (L is (K,T,T), eps is (n_group,K,T))
            f_group = gp_means + torch.einsum("kij,nkj->nki", L_all, f_eps)

            # eta: i.i.d. component (n_group, K, T) — 1 sample call
            with group_plate:
                eta_group = pyro.sample(
                    f"eta_g{g_idx}",
                    dist.Normal(
                        self._zeros((n_group, self.n_factors, n_t)),
                        self._ones((n_group, self.n_factors, n_t)),
                    ).to_event(2),
                )  # (n_group, K, T)

            # Mix: z = sqrt(1-zeta)*f + sqrt(zeta)*amplitude*eta
            # f_group has marginal variance ~ amplitude² (from Cholesky of kernel),
            # eta_group is N(0,1), so scale by amplitude to match.
            zeta_exp = zetas.unsqueeze(0).unsqueeze(-1)  # (1, K, 1)
            amp_exp = amplitudes.unsqueeze(0).unsqueeze(-1)  # (1, K, 1)
            z_group = (torch.sqrt(1 - zeta_exp) * f_group
                       + torch.sqrt(zeta_exp) * amp_exp * eta_group)

            # Scatter into z_all: map group patients to their obs rows
            for i, p in enumerate(group_pats):
                obs_rows = (patient_idx == p).nonzero(as_tuple=True)[0]
                z_all[obs_rows, :] = z_group[i, :, :len(obs_rows)].T

        output_dict["z"] = z_all

        # --- Observation likelihood ---
        obs_plate = pyro.plate("obs", n_obs, dim=-2, device=self.device)
        with obs_plate:
            for m in range(self.n_views):
                with feature_plates[m]:
                    y_loc = torch.matmul(z_all, output_dict[f"w_{m}"])
                    if self.n_covariates > 0 and covs is not None:
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
# TemporalGuide — variational posterior (vectorized, diagonal)
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
        max_obs = self.model.max_obs

        # --- Standard sites ---
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
        site_to_dist["zeta"] = "Beta"

        # Register standard params
        for name, shape in site_to_shape.items():
            if name == "zeta":
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

        # --- GP factor guide: diagonal Normal, stored as (P, K, T_max) ---
        # Much more memory-efficient than per-patient Cholesky
        self.z_mean = PyroParam(
            torch.zeros(n_patients, n_factors, max_obs, device=self.model.device),
            constraints.real,
        )
        self.z_scale = PyroParam(
            0.1 * torch.ones(n_patients, n_factors, max_obs, device=self.model.device),
            constraints.positive,
        )
        # eta guide: also diagonal Normal (P, K, T_max)
        self.eta_mean = PyroParam(
            torch.zeros(n_patients, n_factors, max_obs, device=self.model.device),
            constraints.real,
        )
        self.eta_scale = PyroParam(
            0.1 * torch.ones(n_patients, n_factors, max_obs, device=self.model.device),
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
        # stored as (P, K, T) -> transpose to (P, T, K)
        return self.z_mean.detach().permute(0, 2, 1).cpu().numpy()

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

        # --- Vectorized GP factor guide: 2 sample sites per group ---
        for g_idx, (mask_bool, group_pats) in enumerate(self.model.mask_groups):
            group_pats = group_pats.to(self.model.device)
            n_t = mask_bool.sum().item()
            n_group = group_pats.shape[0]

            group_plate = pyro.plate(f"group_{g_idx}", n_group, dim=-1)

            # f guide: (n_group, K, T) — matches model's reparameterized eps
            mu_f = self.z_mean[group_pats, :, :n_t]   # (n_group, K, T)
            s_f = self.z_scale[group_pats, :, :n_t]    # (n_group, K, T)
            with group_plate:
                pyro.sample(
                    f"f_g{g_idx}",
                    dist.Normal(mu_f, s_f).to_event(2),
                )

            # eta guide: (n_group, K, T)
            mu_e = self.eta_mean[group_pats, :, :n_t]
            s_e = self.eta_scale[group_pats, :, :n_t]
            with group_plate:
                pyro.sample(
                    f"eta_g{g_idx}",
                    dist.Normal(mu_e, s_e).to_event(2),
                )

        return output_dict


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R² = 1 - SS_res / SS_tot (MOFA2/PACMon convention)."""
    ss_res = np.nansum(np.square(y_true - y_pred))
    ss_tot = np.nansum(np.square(y_true))
    if ss_tot == 0:
        return 0.0
    return 1.0 - (ss_res / ss_tot)


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
        covariates: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        prior_confidence: Union[float, str] = "low",
        n_sparse_factors: Optional[int] = None,
        n_dense_factors: Optional[int] = None,
        sample_names: Optional[list] = None,
        feature_names: Optional[Dict[str, list]] = None,
        covariate_names: Optional[list] = None,
        kernel: str = "matern32",
        likelihoods: Optional[Dict[str, str]] = None,
        normalize: bool = True,
        shared_lengthscale: bool = False,
        guide_type: str = "diagonal",
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

        # Sample names
        if sample_names is not None:
            self.sample_names = list(sample_names)
        else:
            # Try to extract from first observation DataFrame
            first_obs = next(iter(observations.values()))
            if isinstance(first_obs, pd.DataFrame):
                self.sample_names = first_obs.index.tolist()
            else:
                self.sample_names = [f"sample_{i}" for i in range(self.n_patients)]

        # Setup observations
        self.view_names = list(observations.keys())
        self.observations, self.feature_names = self._setup_observations(observations)
        if feature_names is not None:
            for vn, names in feature_names.items():
                self.feature_names[vn] = names
        self.n_features = {vn: obs.shape[-1] for vn, obs in self.observations.items()}

        # Setup covariates (accept DataFrame)
        self.covariates = None
        self.n_covariates = 0
        self.covariate_names = None
        if covariates is not None:
            if isinstance(covariates, pd.DataFrame):
                if covariate_names is None:
                    covariate_names = covariates.columns.tolist()
                covariates = covariates.to_numpy(dtype=np.float32)
            self.covariates = np.asarray(covariates, dtype=np.float32)
            self.n_covariates = self.covariates.shape[1]
            if covariate_names is not None:
                self.covariate_names = list(covariate_names)
            else:
                self.covariate_names = [f"cov_{c}" for c in range(self.n_covariates)]

        # Setup priors — save factor names before conversion strips them
        self.prior_confidence = self._setup_prior_confidence(prior_confidence)
        _prior_mask_names = None
        if prior_masks is not None:
            vn0 = self.view_names[0]
            if vn0 in prior_masks and isinstance(prior_masks[vn0], pd.DataFrame):
                _prior_mask_names = prior_masks[vn0].index.tolist()

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
        if _prior_mask_names is not None:
            factor_names.extend(_prior_mask_names)
        elif self.n_sparse_factors > 0:
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
            if arr.ndim == 2:
                arr = arr[:, np.newaxis, :]
            if self.normalize:
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

        processed = {}
        for vn in self.view_names:
            if vn in masks:
                m = masks[vn]
                if isinstance(m, pd.DataFrame):
                    processed[vn] = m.to_numpy(dtype=np.float32)
                else:
                    processed[vn] = np.asarray(m, dtype=np.float32)
            else:
                n_factors = next(iter(masks.values())).shape[0] if isinstance(next(iter(masks.values())), (np.ndarray, pd.DataFrame)) else 0
                processed[vn] = np.zeros((n_factors, self.n_features[vn]), dtype=np.float32)

        prior_scales = {}
        for vn, vm in processed.items():
            prior_scales[vn] = np.clip(
                vm.astype(np.float32) + (1.0 - self.prior_confidence), 1e-8, 1.0
            )

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

            # Gradient clipping (only leaf tensors to avoid PyTorch warning)
            if clip_norm > 0:
                params = [p for p in pyro.get_param_store().values()
                          if p.requires_grad and p.is_leaf and p.grad is not None]
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

    def get_variance_explained(self, per_factor: bool = True) -> dict:
        """Compute R² (variance explained) per view and optionally per factor.

        Follows the MOFA2/PACMon R² formula:
            R² = 1 - SS_res / SS_tot
        where SS_tot = sum(y²) (data assumed centered/normalized).

        Args:
            per_factor: If True, also compute factor-wise R² per view.

        Returns:
            dict with:
                - 'total': {view_name: float} — total R² per view (percentage)
                - 'per_factor': {view_name: (K,) array} — factor-wise R² (percentage)
                    Only present if per_factor=True.
        """
        if not self._trained:
            raise RuntimeError("Model must be trained first.")
        if self.observations is None:
            raise RuntimeError("Variance explained requires observation data.")

        z = self.get_factors()          # (P, T, K)
        ws = self.get_loadings()        # {view: (K, D_m)}
        betas = None
        if self.n_covariates > 0 and self.covariates is not None:
            betas = self.get_covariate_coefficients()  # {view: (C, D_m)}

        result = {"total": {}}
        if per_factor:
            result["per_factor"] = {}

        for vn in self.view_names:
            y_true = self.observations[vn]  # (P, T, D_m)
            w_m = ws[vn]                    # (K, D_m)

            # Reconstruct y_pred per observed (p, t) entry
            # y_pred[p,t] = z[p,t,:] @ w_m + covs[p] @ beta_m
            y_pred = np.zeros_like(y_true)
            for p in range(self.n_patients):
                for t in range(self.n_timepoints):
                    if self.patient_masks_np[p, t]:
                        z_pt = z[p, t, :]  # (K,)
                        if not np.any(np.isnan(z_pt)):
                            y_pred[p, t, :] = z_pt @ w_m
                            if betas is not None:
                                y_pred[p, t, :] += self.covariates[p] @ betas[vn]

            # Mask unobserved entries
            mask_3d = self.patient_masks_np[:, :, np.newaxis]
            y_true_m = np.where(mask_3d, y_true, np.nan)
            y_pred_m = np.where(mask_3d, y_pred, np.nan)

            result["total"][vn] = max(0.0, _r2(y_true_m, y_pred_m)) * 100.0

            if per_factor:
                r2_k = np.zeros(self.n_factors)
                for k in range(self.n_factors):
                    y_pred_k = np.zeros_like(y_true)
                    for p in range(self.n_patients):
                        for t in range(self.n_timepoints):
                            if self.patient_masks_np[p, t]:
                                z_ptk = z[p, t, k]
                                if not np.isnan(z_ptk):
                                    y_pred_k[p, t, :] = z_ptk * w_m[k, :]
                    y_pred_k_m = np.where(mask_3d, y_pred_k, np.nan)
                    r2_k[k] = max(0.0, _r2(y_true_m, y_pred_k_m)) * 100.0
                result["per_factor"][vn] = r2_k

        return result

    def predict(self, new_time_points: np.ndarray):
        """GP posterior prediction at unobserved times."""
        if not self._trained:
            raise RuntimeError("Model must be trained before prediction.")

        new_t = torch.tensor(new_time_points, dtype=torch.float32, device=self.device)
        lengthscales = torch.tensor(self._guide.get_lengthscales(), dtype=torch.float32, device=self.device)
        amplitudes = torch.tensor(self._guide.get_amplitudes(), dtype=torch.float32, device=self.device)
        z_means = self._guide.z_mean.detach()  # (P, K, T)

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
                    z_obs = z_means[p, k, :n_obs]  # (P, K, T) indexing

                    mu_pred = K_new_obs @ K_inv @ z_obs
                    var_pred = torch.diag(K_new - K_new_obs @ K_inv @ K_new_obs.T).clamp(min=0)

                    means[p, :, k] = mu_pred.cpu().numpy()
                    variances[p, :, k] = var_pred.cpu().numpy()

        return {"mean": means, "variance": variances}

    def save(self, directory: str, include_data: bool = True) -> str:
        """Save model state to a directory.

        Writes three files following the PACMon/MuVI pattern:
            metadata.json  — config, names, training state
            params.npz     — guide parameters (loc::*, scale::*, GP params)
            structure.npz  — observations, prior masks, covariates, time grid

        Args:
            directory: Path to the output directory (created if needed).
            include_data: If True (default), include observations and covariates
                in structure.npz. Set False for lightweight saves (get_factors,
                get_loadings, predict still work; get_variance_explained needs
                observations re-provided on load).

        Returns:
            Path to the saved directory.
        """
        if not self._trained:
            raise RuntimeError("Model must be trained before saving.")

        import tpacmon

        directory = Path(directory)
        directory.mkdir(exist_ok=True, parents=True)

        # --- A) metadata.json ---
        metadata = {
            "tpacmon_version": tpacmon.__version__,
            "state_version": "1.0.0",
            "view_names": self.view_names,
            "factor_names": self.factor_names,
            "feature_names": self.feature_names,
            "sample_names": self.sample_names,
            "covariate_names": self.covariate_names,
            "n_sparse_factors": self.n_sparse_factors,
            "n_dense_factors": self.n_dense_factors,
            "n_factors": self.n_factors,
            "n_patients": self.n_patients,
            "n_timepoints": self.n_timepoints,
            "n_covariates": self.n_covariates,
            "n_features": {vn: int(n) for vn, n in self.n_features.items()},
            "kernel": self.kernel,
            "likelihoods": self.likelihoods,
            "prior_confidence": self.prior_confidence,
            "gp_scale": self.gp_scale,
            "normalize": self.normalize,
            "_trained": self._trained,
            "training_history": [float(x) for x in self._training_history],
        }
        with open(directory / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # --- B) params.npz ---
        params = {}
        for name, loc_param in self._guide.locs.named_parameters():
            params[f"loc::{name}"] = loc_param.detach().cpu().numpy()
        for name, scale_param in self._guide.scales.named_parameters():
            params[f"scale::{name}"] = scale_param.detach().cpu().numpy()
        params["z_mean"] = self._guide.z_mean.detach().cpu().numpy()
        params["z_scale"] = self._guide.z_scale.detach().cpu().numpy()
        params["eta_mean"] = self._guide.eta_mean.detach().cpu().numpy()
        params["eta_scale"] = self._guide.eta_scale.detach().cpu().numpy()
        np.savez_compressed(directory / "params.npz", **params)

        # --- C) structure.npz ---
        structure = {
            "time_points": self.time_points_np,
            "patient_masks": self.patient_masks_np,
            "prior_masks": self.prior_masks,
            "prior_scales": (
                self.prior_scales.cpu().numpy()
                if self.prior_scales is not None
                else None
            ),
        }
        if include_data:
            structure["observations"] = self.observations
            structure["covariates"] = self.covariates
        np.savez_compressed(directory / "structure.npz", structure=structure)

        logger.info("Model saved to %s (include_data=%s)", directory, include_data)
        return str(directory)

    @classmethod
    def load(cls, directory: str, map_location: Optional[str] = None) -> "TemporalPACMON":
        """Load a saved model from a directory.

        The loaded model supports all read-only operations (get_factors,
        get_loadings, predict, get_variance_explained, etc.).

        Args:
            directory: Path to saved model directory.
            map_location: Device for tensors. Default: auto-detect.

        Returns:
            TemporalPACMON instance with _trained=True.
        """
        if map_location is None:
            map_location = "cuda" if torch.cuda.is_available() else "cpu"

        directory = Path(directory)

        # --- A) Load metadata ---
        with open(directory / "metadata.json", "r") as f:
            metadata = json.load(f)

        # --- B) Load params ---
        params_raw = np.load(directory / "params.npz")
        params = {k: params_raw[k] for k in params_raw.files}

        # --- C) Load structure ---
        raw_struct = np.load(directory / "structure.npz", allow_pickle=True)
        structure = raw_struct["structure"].item()

        # --- D) Build stub model with placeholder observations ---
        placeholder_obs = {
            vn: np.zeros(
                (metadata["n_patients"], metadata["n_timepoints"],
                 metadata["n_features"][vn]),
                dtype=np.float32,
            )
            for vn in metadata["view_names"]
        }
        placeholder_covs = (
            np.zeros(
                (metadata["n_patients"], metadata["n_covariates"]),
                dtype=np.float32,
            )
            if metadata["n_covariates"] > 0
            else None
        )

        model = cls(
            observations=placeholder_obs,
            time_points=structure["time_points"],
            patient_masks=structure["patient_masks"],
            prior_masks=None,
            covariates=placeholder_covs,
            n_sparse_factors=metadata["n_sparse_factors"],
            n_dense_factors=metadata["n_dense_factors"],
            sample_names=metadata["sample_names"],
            covariate_names=metadata.get("covariate_names"),
            kernel=metadata["kernel"],
            likelihoods=metadata["likelihoods"],
            prior_confidence=metadata["prior_confidence"],
            gp_scale=metadata["gp_scale"],
            normalize=False,  # data already normalized at save time
            device=map_location,
        )

        # --- E) Inject full state ---
        model.view_names = metadata["view_names"]
        model.factor_names = metadata["factor_names"]
        model.feature_names = metadata["feature_names"]
        model.sample_names = metadata["sample_names"]
        model.covariate_names = metadata.get("covariate_names")
        model.n_features = {vn: int(n) for vn, n in metadata["n_features"].items()}

        # Restore observations/covariates from structure (if saved)
        model.observations = structure.get("observations")
        model.covariates = structure.get("covariates")

        # Restore priors
        model.prior_masks = structure["prior_masks"]
        prior_scales_np = structure["prior_scales"]
        if prior_scales_np is not None:
            model.prior_scales = torch.tensor(
                prior_scales_np, dtype=torch.float32, device=map_location
            )
        else:
            model.prior_scales = None

        # Restore factor counts (constructor set these from prior_masks=None)
        model.n_sparse_factors = metadata["n_sparse_factors"]
        model.n_dense_factors = metadata["n_dense_factors"]
        model.n_factors = metadata["n_factors"]

        model._trained = metadata["_trained"]
        model._training_history = metadata["training_history"]

        # Rebuild model & guide, then load params
        model._setup_model_guide()

        # Load guide params (loc::* / scale::* into locs/scales, GP params directly)
        for key, arr in params.items():
            tensor = torch.from_numpy(arr).to(map_location)
            if key.startswith("loc::"):
                name = key[5:]
                getattr(model._guide.locs, name).data.copy_(tensor)
            elif key.startswith("scale::"):
                name = key[7:]
                getattr(model._guide.scales, name).data.copy_(tensor)
            elif key == "z_mean":
                model._guide.z_mean.data.copy_(tensor)
            elif key == "z_scale":
                model._guide.z_scale.data.copy_(tensor)
            elif key == "eta_mean":
                model._guide.eta_mean.data.copy_(tensor)
            elif key == "eta_scale":
                model._guide.eta_scale.data.copy_(tensor)

        logger.info(
            "Model loaded from %s (version %s)",
            directory, metadata.get("tpacmon_version", "unknown"),
        )
        return model

    @property
    def training_history(self):
        return self._training_history
