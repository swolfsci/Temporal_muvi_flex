#!/usr/bin/env python
"""CLI entry point for tpacmon synthetic evaluations.

Usage:
    python evaluations/run_eval.py --config evaluations/configs/level1_dense.yaml
    python evaluations/run_eval.py --config evaluations/configs/level1_dense.yaml --seeds 10
    python evaluations/run_eval.py --config evaluations/configs/level1_dense.yaml --seed 42 --output results/custom/

Each run produces:
    1. One JSON file per (seed, param_combo) with full metrics + config
    2. A summary CSV aggregating all runs
    3. An interpretation-ready JSON with summary statistics across seeds

Output structure:
    results/<level_name>/
        runs/                         # individual run JSONs
            seed_42_noise_0.5.json
            seed_42_noise_1.0.json
            ...
        summary.csv                   # flat CSV, one row per run
        summary.json                  # aggregated stats (mean, std per metric per param combo)
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
    apply_alignment,
    collect_result,
    factor_score_correlation,
    gamma_recovery,
    beta_recovery,
    lengthscale_recovery,
    loading_aupr,
    loading_correlation,
    reconstruction_rmse,
    results_to_csv_row,
    variance_explained,
    zeta_discrimination,
)
from evaluations.synthetic import generate_evaluation_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_single(data_config: dict, model_config: dict, training_config: dict, seed: int) -> dict:
    """Run one evaluation: generate data, fit model, compute all metrics.

    Returns a structured result dict.
    """
    from tpacmon.core.models import TemporalPACMON

    # -------------------------------------------------------------------
    # Generate data
    # -------------------------------------------------------------------
    gen_params = {k: v for k, v in data_config.items() if not k.startswith("_")}
    gen_params["seed"] = seed
    data = generate_evaluation_data(**gen_params)

    K_true = data["true_z"].shape[2]

    # -------------------------------------------------------------------
    # Build model
    # -------------------------------------------------------------------
    model_kwargs = {
        "observations": data["observations"],
        "time_points": data["time_points"],
        "patient_masks": data["patient_masks"],
    }

    if data["covariates"] is not None and model_config.get("use_covariates", True):
        model_kwargs["covariates"] = data["covariates"]

    if data["prior_masks"] is not None and model_config.get("use_prior_masks", True):
        model_kwargs["prior_masks"] = data["prior_masks"]
        n_sparse = data["prior_masks"][data["view_names"][0]].shape[0]
        model_kwargs["n_sparse_factors"] = n_sparse
        n_dense = model_config.get("n_dense_factors", max(0, K_true - n_sparse))
        model_kwargs["n_dense_factors"] = n_dense
    else:
        model_kwargs["n_dense_factors"] = model_config.get("n_dense_factors", K_true)

    for key in ["kernel", "gp_scale", "shared_lengthscale", "guide_type"]:
        if key in model_config:
            model_kwargs[key] = model_config[key]

    model = TemporalPACMON(**model_kwargs)

    # -------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------
    train_params = {
        "n_epochs": training_config.get("n_epochs", 2000),
        "learning_rate": training_config.get("learning_rate", 0.01),
        "early_stopping": training_config.get("early_stopping", True),
        "patience": training_config.get("patience", 20),
        "min_epochs": training_config.get("min_epochs", 100),
        "seed": seed,
    }

    t0 = time.time()
    model.fit(**train_params)
    runtime = time.time() - t0

    # -------------------------------------------------------------------
    # Extract learned parameters
    # -------------------------------------------------------------------
    learned_z = model.get_factors()          # (P, T_max, K_learned)
    learned_w = model.get_loadings()         # {view: (K_learned, D)}
    learned_ls = model.get_lengthscales()    # (K_learned,)
    learned_zeta = model.get_smoothness()    # (K_learned,)

    K_learned = learned_z.shape[2]

    # Trim true_z timepoints to match learned (in case of padding)
    T_true = data["true_z"].shape[1]
    T_learned = learned_z.shape[1]
    T = min(T_true, T_learned)
    true_z_trim = data["true_z"][:, :T, :]
    learned_z_trim = learned_z[:, :T, :]

    # -------------------------------------------------------------------
    # Align factors
    # -------------------------------------------------------------------
    alignment = align_factors(true_z_trim, learned_z_trim)

    # -------------------------------------------------------------------
    # Compute metrics
    # -------------------------------------------------------------------
    metrics = {}
    metrics["alignment"] = {
        "permutation": alignment["permutation"].tolist(),
        "sign_flips": alignment["sign_flips"].tolist(),
        "correlations": alignment["correlations"].tolist(),
    }

    # Factor score correlation
    metrics["factor_score_correlation"] = factor_score_correlation(
        true_z_trim, learned_z_trim, alignment
    )

    # Loading metrics (per view, then average)
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

    # Reconstruction & R²
    for vn in data["view_names"]:
        w_aligned = apply_alignment(learned_w[vn], alignment, axis=0)
        y_pred = learned_z_trim @ w_aligned[:, :data["observations"][vn].shape[2]]
        y_true = data["observations"][vn][:, :T, :]
        metrics[f"rmse_{vn}"] = reconstruction_rmse(y_true, y_pred, data["patient_masks"][:, :T])
        metrics[f"r2_{vn}"] = variance_explained(y_true, y_pred, data["patient_masks"][:, :T])

    r2_vals = [metrics[f"r2_{vn}"] for vn in data["view_names"]]
    metrics["r2"] = float(np.mean(r2_vals))
    rmse_vals = [metrics[f"rmse_{vn}"] for vn in data["view_names"]]
    metrics["rmse"] = float(np.mean(rmse_vals))

    # GP metrics (if temporal factors exist)
    n_temporal = int(data["true_is_temporal"].sum())
    if n_temporal > 0 and len(learned_ls) > 0:
        metrics["lengthscale_recovery"] = lengthscale_recovery(
            data["true_lengthscales"], learned_ls, alignment
        )
        metrics["zeta_discrimination"] = zeta_discrimination(
            data["true_is_temporal"], learned_zeta, alignment
        )

    # Zeta summary (always useful)
    metrics["zeta_mean"] = float(np.mean(learned_zeta))
    metrics["zeta_per_factor"] = learned_zeta.tolist()

    # Covariate metrics
    if data["true_gamma"] is not None:
        try:
            learned_gamma = model.get_covariate_coefficients()
            if learned_gamma is not None:
                # gamma is factor-level: (C, K)
                metrics["gamma_recovery"] = gamma_recovery(
                    data["true_gamma"], learned_gamma, alignment
                )
        except Exception as e:
            log.warning(f"Could not extract gamma: {e}")

    if data["true_beta"] is not None:
        try:
            learned_beta = model.get_covariate_coefficients()
            # beta recovery would need feature-level extraction
            # placeholder for when feature-level beta extraction is available
        except Exception:
            pass

    # Training info
    metrics["epochs_run"] = len(model.loss_history) if hasattr(model, "loss_history") else None
    metrics["final_loss"] = float(model.loss_history[-1]) if hasattr(model, "loss_history") and model.loss_history else None

    # -------------------------------------------------------------------
    # Build config record (for reproducibility)
    # -------------------------------------------------------------------
    config_record = {
        **{k: v for k, v in data_config.items() if not isinstance(v, np.ndarray)},
        **{f"model_{k}": v for k, v in model_config.items()},
        **{f"train_{k}": v for k, v in training_config.items()},
    }

    level = data_config.get("_level", 0)
    return collect_result(level, seed, config_record, metrics, runtime)


def aggregate_results(results: list) -> dict:
    """Compute summary statistics across seeds for each parameter combo.

    Groups by all cfg_* columns (excluding seed), then computes mean/std
    for each metric.
    """
    if not results:
        return {}

    rows = [results_to_csv_row(r) for r in results]
    # Identify grouping keys (cfg_* but not seed)
    cfg_keys = sorted({k for row in rows for k in row if k.startswith("cfg_")})
    metric_keys = sorted({
        k for row in rows for k in row
        if k not in cfg_keys and k not in ("level", "seed", "runtime_sec")
    })

    # Group by config
    groups = {}
    for row in rows:
        group_key = tuple(row.get(k) for k in cfg_keys)
        groups.setdefault(group_key, []).append(row)

    summary = []
    for group_key, group_rows in groups.items():
        entry = {k: v for k, v in zip(cfg_keys, group_key)}
        entry["n_seeds"] = len(group_rows)

        for mk in metric_keys:
            vals = [r[mk] for r in group_rows if r.get(mk) is not None]
            if vals:
                entry[f"{mk}_mean"] = float(np.mean(vals))
                entry[f"{mk}_std"] = float(np.std(vals))
                entry[f"{mk}_min"] = float(np.min(vals))
                entry[f"{mk}_max"] = float(np.max(vals))
            else:
                entry[f"{mk}_mean"] = None

        entry["runtime_sec_mean"] = float(np.mean([r["runtime_sec"] for r in group_rows]))
        summary.append(entry)

    return {"groups": summary, "cfg_keys": cfg_keys, "metric_keys": metric_keys}


def expand_sweep(config: dict) -> list:
    """Expand a config with sweep parameters into individual configs.

    If config has a 'sweep' key, it's a dict of {param_name: [values]}.
    Each value combination produces a separate data_config.
    """
    sweep = config.get("sweep", None)
    if sweep is None:
        return [config["data"]]

    base = dict(config["data"])
    configs = []

    # Single parameter sweep (most common)
    sweep_param = list(sweep.keys())[0]
    sweep_values = sweep[sweep_param]

    for val in sweep_values:
        cfg = dict(base)
        cfg[sweep_param] = val
        configs.append(cfg)

    return configs


def main():
    parser = argparse.ArgumentParser(description="tpacmon synthetic evaluation runner")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--seeds", type=int, default=None, help="Number of seeds (overrides config)")
    parser.add_argument("--seed", type=int, default=None, help="Single seed (overrides config)")
    parser.add_argument("--output", type=str, default=None, help="Output directory (overrides config)")
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit")
    args = parser.parse_args()

    config = load_config(args.config)

    # Determine seeds
    if args.seed is not None:
        seeds = [args.seed]
    elif args.seeds is not None:
        seeds = list(range(args.seeds))
    else:
        seeds = list(range(config.get("n_seeds", 5)))

    # Output directory
    level_name = config.get("name", "unnamed")
    output_dir = Path(args.output) if args.output else Path(f"evaluations/results/{level_name}")
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    model_config = config.get("model", {})
    training_config = config.get("training", {})

    # Expand sweep configs
    data_configs = expand_sweep(config)
    level = config.get("level", 0)

    if args.dry_run:
        print(f"Level: {level} ({level_name})")
        print(f"Seeds: {seeds}")
        print(f"Parameter combos: {len(data_configs)}")
        print(f"Total runs: {len(data_configs) * len(seeds)}")
        print(f"Output: {output_dir}")
        return

    log.info(f"Starting evaluation: {level_name} (level {level})")
    log.info(f"  {len(data_configs)} configs x {len(seeds)} seeds = {len(data_configs) * len(seeds)} runs")
    log.info(f"  Output: {output_dir}")

    all_results = []

    for di, data_config in enumerate(data_configs):
        data_config["_level"] = level

        for seed in seeds:
            # Build a descriptive run name from sweep params
            sweep = config.get("sweep", {})
            sweep_parts = []
            for sp in sweep:
                sweep_parts.append(f"{sp}_{data_config.get(sp)}")
            sweep_str = "_".join(sweep_parts) if sweep_parts else "default"
            run_name = f"seed_{seed}_{sweep_str}"

            run_file = runs_dir / f"{run_name}.json"
            if run_file.exists():
                log.info(f"  Skipping {run_name} (already exists)")
                with open(run_file) as f:
                    all_results.append(json.load(f))
                continue

            log.info(f"  Running {run_name} ...")
            try:
                result = run_single(data_config, model_config, training_config, seed)

                with open(run_file, "w") as f:
                    json.dump(result, f, indent=2)
                all_results.append(result)

                z_corr = result["metrics"].get("factor_score_correlation", {}).get("mean", "N/A")
                r2 = result["metrics"].get("r2", "N/A")
                rt = result["runtime_sec"]
                log.info(f"    z_corr={z_corr:.3f}  R²={r2:.3f}  time={rt:.1f}s" if isinstance(z_corr, float) else f"    time={rt:.1f}s")

            except Exception as e:
                log.error(f"    FAILED: {e}")
                # Save error record
                error_result = {
                    "level": level, "seed": seed,
                    "config": {k: v for k, v in data_config.items() if not isinstance(v, np.ndarray)},
                    "error": str(e),
                }
                with open(run_file.with_suffix(".error.json"), "w") as f:
                    json.dump(error_result, f, indent=2)

    # -------------------------------------------------------------------
    # Write summary CSV
    # -------------------------------------------------------------------
    if all_results:
        csv_rows = [results_to_csv_row(r) for r in all_results]
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

        # -------------------------------------------------------------------
        # Write aggregated summary JSON
        # -------------------------------------------------------------------
        agg = aggregate_results(all_results)
        agg_path = output_dir / "summary.json"
        with open(agg_path, "w") as f:
            json.dump(agg, f, indent=2)
        log.info(f"Summary JSON: {agg_path}")

    log.info("Done.")


if __name__ == "__main__":
    main()
