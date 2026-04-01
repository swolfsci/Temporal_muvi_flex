#!/usr/bin/env python
"""Baseline evaluation: run MuVI and PACMon on the same synthetic data as tpacmon.

MuVI and PACMon are non-temporal models. To compare fairly:
- Observations are flattened from (P, T, D) → (P*T_obs, D), treating each
  patient-visit as an independent sample.
- Prior masks are shared (same gene-set structure).
- Covariates are repeated per observed timepoint.
- Factor scores are reshaped back to (P, T, K) for metric computation.

Usage:
    python evaluations/run_baseline.py --config evaluations/configs/level3_priors.yaml --model muvi
    python evaluations/run_baseline.py --config evaluations/configs/level3_priors.yaml --model pacmon
    python evaluations/run_baseline.py --config evaluations/configs/level4_covariates.yaml --model both
"""

import argparse
import csv
import json
import logging
import time
from pathlib import Path

import numpy as np
import yaml

from evaluations.metrics import (
    align_factors,
    collect_result,
    factor_score_correlation,
    loading_aupr,
    loading_correlation,
    results_to_csv_row,
)
from evaluations.synthetic import generate_evaluation_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def flatten_temporal_data(data):
    """Flatten (P, T, D) observations to (N_obs, D) for non-temporal models.

    Returns:
        flat_obs: {view: (N_obs, D)}
        flat_covs: (N_obs, C) or None
        row_map: list of (patient_idx, time_idx) for each row
    """
    P = data["patient_masks"].shape[0]
    T = data["patient_masks"].shape[1]

    row_map = []
    for p in range(P):
        for t in range(T):
            if data["patient_masks"][p, t]:
                row_map.append((p, t))

    flat_obs = {}
    for vn in data["view_names"]:
        rows = []
        for p, t in row_map:
            row = data["observations"][vn][p, t, :]
            if np.any(np.isnan(row)):
                row = np.nan_to_num(row, nan=0.0)
            rows.append(row)
        flat_obs[vn] = np.array(rows, dtype=np.float32)

    flat_covs = None
    if data["covariates"] is not None:
        flat_covs = np.array(
            [data["covariates"][p] for p, t in row_map], dtype=np.float32
        )

    return flat_obs, flat_covs, row_map


def unflatten_factors(z_flat, row_map, n_patients, n_timepoints):
    """Reshape (N_obs, K) factor scores back to (P, T, K)."""
    K = z_flat.shape[1]
    z = np.full((n_patients, n_timepoints, K), np.nan, dtype=np.float32)
    for i, (p, t) in enumerate(row_map):
        z[p, t, :] = z_flat[i]
    return z


def run_muvi(data, flat_obs, flat_covs, row_map, model_config, training_config, seed):
    """Fit MuVI on flattened data and return metrics."""
    from muvi import MuVI

    n_informed = 0
    prior_masks = None
    if data["prior_masks"] is not None and model_config.get("use_prior_masks", True):
        prior_masks = data["prior_masks"]
        n_informed = prior_masks[data["view_names"][0]].shape[0]

    n_dense = model_config.get("n_dense_factors", 2)

    model_kwargs = {
        "observations": flat_obs,
        "prior_masks": prior_masks,
        "n_factors": n_dense,
        "normalize": True,
        "device": "cpu",
    }
    if flat_covs is not None and model_config.get("use_covariates", True):
        model_kwargs["covariates"] = flat_covs

    model = MuVI(**model_kwargs)

    t0 = time.time()
    model.fit(
        n_epochs=training_config.get("n_epochs", 5000),
        learning_rate=training_config.get("learning_rate", 0.005),
        early_stopping=training_config.get("early_stopping", True),
        seed=seed,
    )
    runtime = time.time() - t0

    # Extract
    z_flat = model.get_factor_scores()  # (N_obs, K)
    learned_w = model.get_factor_loadings()  # {view: (K, D)}

    P = data["patient_masks"].shape[0]
    T = data["patient_masks"].shape[1]
    learned_z = unflatten_factors(z_flat, row_map, P, T)

    return learned_z, learned_w, runtime, model


def run_pacmon(data, flat_obs, flat_covs, row_map, model_config, training_config, seed):
    """Fit PACMon on flattened data and return metrics."""
    from pacmon import PACMON

    prior_masks = None
    if data["prior_masks"] is not None and model_config.get("use_prior_masks", True):
        prior_masks = data["prior_masks"]

    n_dense = model_config.get("n_dense_factors", 2)

    model_kwargs = {
        "observations": flat_obs,
        "prior_masks": prior_masks,
        "n_dense_factors": n_dense,
        "normalize": True,
        "device": "cpu",
    }
    if flat_covs is not None and model_config.get("use_covariates", True):
        model_kwargs["covariates"] = flat_covs

    model = PACMON(**model_kwargs)

    t0 = time.time()
    model.fit(
        n_epochs=training_config.get("n_epochs", 5000),
        learning_rate=training_config.get("learning_rate", 0.005),
        early_stopping=training_config.get("early_stopping", True),
        seed=seed,
    )
    runtime = time.time() - t0

    z_flat = model.get_factor_scores()
    learned_w = model.get_factor_loadings()

    P = data["patient_masks"].shape[0]
    T = data["patient_masks"].shape[1]
    learned_z = unflatten_factors(z_flat, row_map, P, T)

    return learned_z, learned_w, runtime, model


def compute_metrics(data, learned_z, learned_w, runtime, model_name, level, seed, config):
    """Compute standard metrics for a baseline model."""
    T = min(data["true_z"].shape[1], learned_z.shape[1])
    true_z = data["true_z"][:, :T, :]
    lz = learned_z[:, :T, :]

    alignment = align_factors(true_z, lz)

    metrics = {
        "model": model_name,
        "alignment": {
            "permutation": alignment["permutation"].tolist(),
            "sign_flips": alignment["sign_flips"].tolist(),
            "correlations": alignment["correlations"].tolist(),
        },
    }

    metrics["factor_score_correlation"] = factor_score_correlation(true_z, lz, alignment)

    aupr_all, corr_all = [], []
    for vn in data["view_names"]:
        aupr_v = loading_aupr(data["true_w_mask"][vn], learned_w[vn], alignment)
        corr_v = loading_correlation(data["true_w"][vn], learned_w[vn], alignment)
        metrics[f"loading_aupr_{vn}"] = aupr_v
        metrics[f"loading_correlation_{vn}"] = corr_v
        aupr_all.append(aupr_v["overall"])
        corr_all.append(corr_v["mean"])
    metrics["loading_aupr"] = {"overall": float(np.mean(aupr_all)), "mean": float(np.mean(aupr_all))}
    metrics["loading_correlation"] = {"mean": float(np.mean(corr_all))}

    return collect_result(level, seed, {**config, "model": model_name}, metrics, runtime)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def expand_sweep(config):
    sweep = config.get("sweep", None)
    if sweep is None:
        return [config["data"]]
    base = dict(config["data"])
    configs = []
    sweep_param = list(sweep.keys())[0]
    for val in sweep[sweep_param]:
        cfg = dict(base)
        cfg[sweep_param] = val
        configs.append(cfg)
    return configs


def main():
    parser = argparse.ArgumentParser(description="Baseline evaluation: MuVI / PACMon")
    parser.add_argument("--config", required=True, help="tpacmon YAML config file")
    parser.add_argument("--model", required=True, choices=["muvi", "pacmon", "both"])
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    seeds = list(range(args.seeds or config.get("n_seeds", 5)))
    models_to_run = ["muvi", "pacmon"] if args.model == "both" else [args.model]

    level_name = config.get("name", "unnamed")
    output_dir = Path(args.output) if args.output else Path(f"evaluations/results/baseline_{level_name}")
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    model_config = config.get("model", {})
    training_config = config.get("training", {})
    data_configs = expand_sweep(config)
    level = config.get("level", 0)

    if args.dry_run:
        print(f"Level: {level} ({level_name})")
        print(f"Models: {models_to_run}")
        print(f"Seeds: {seeds}")
        print(f"Configs: {len(data_configs)}")
        print(f"Total runs: {len(data_configs) * len(seeds) * len(models_to_run)}")
        return

    log.info(f"Baseline evaluation: {level_name}, models={models_to_run}")

    all_results = []

    for data_config in data_configs:
        for seed in seeds:
            # Generate data once per (config, seed)
            gen_params = {k: v for k, v in data_config.items() if not k.startswith("_")}
            gen_params["seed"] = seed
            data = generate_evaluation_data(**gen_params)
            flat_obs, flat_covs, row_map = flatten_temporal_data(data)

            sweep = config.get("sweep", {})
            sweep_parts = [f"{sp}_{data_config.get(sp)}" for sp in sweep]
            sweep_str = "_".join(sweep_parts) if sweep_parts else "default"

            for model_name in models_to_run:
                run_name = f"{model_name}_seed_{seed}_{sweep_str}"
                run_file = runs_dir / f"{run_name}.json"

                if run_file.exists():
                    log.info(f"  Skipping {run_name}")
                    with open(run_file) as f:
                        all_results.append(json.load(f))
                    continue

                log.info(f"  Running {run_name} ...")
                try:
                    if model_name == "muvi":
                        lz, lw, rt, _ = run_muvi(
                            data, flat_obs, flat_covs, row_map,
                            model_config, training_config, seed
                        )
                    else:
                        lz, lw, rt, _ = run_pacmon(
                            data, flat_obs, flat_covs, row_map,
                            model_config, training_config, seed
                        )

                    result = compute_metrics(
                        data, lz, lw, rt, model_name, level, seed,
                        {k: v for k, v in data_config.items() if not isinstance(v, np.ndarray)},
                    )

                    with open(run_file, "w") as f:
                        json.dump(result, f, indent=2)
                    all_results.append(result)

                    z_corr = result["metrics"].get("factor_score_correlation", {}).get("mean", "N/A")
                    w_aupr = result["metrics"].get("loading_aupr", {}).get("overall", "N/A")
                    log.info(f"    z_corr={z_corr:.3f}  w_aupr={w_aupr:.3f}  time={rt:.1f}s")

                except Exception as e:
                    log.error(f"    FAILED: {e}")
                    with open(run_file.with_suffix(".error.json"), "w") as f:
                        json.dump({"model": model_name, "seed": seed, "error": str(e)}, f, indent=2)

    # Summary CSV
    if all_results:
        csv_rows = [results_to_csv_row(r) for r in all_results]
        # Add model column
        for i, r in enumerate(all_results):
            csv_rows[i]["model"] = r["config"].get("model", "unknown")

        all_keys = []
        seen = set()
        for row in csv_rows:
            for k in row:
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)

        csv_path = output_dir / "summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(csv_rows)
        log.info(f"Summary CSV: {csv_path}")

    log.info("Done.")


if __name__ == "__main__":
    main()
