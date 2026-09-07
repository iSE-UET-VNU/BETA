"""Run RQ3 Hyperparameter Ablations for Table 12 & Table 13.

Ensures that the default config (K=1, H=3, update='exponential', M=20) is shared
identically between Table 12 (Exponential) and Table 13 (M=20).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

import functools
print = functools.partial(print, flush=True)

# Support running directly or from repository root
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from beta import BETA, BETAConfig
from beta.config import OPERATORS
from beta.data import load_openml, split_train_val_test
from beta.knowledge import ReferenceLibrary, normalize_id
from beta.similarity import SimilarityEncoder
from beta import evaluate, metafeatures, transfer, aco


def df_to_markdown(df: pd.DataFrame) -> str:
    headers = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in df.columns:
            v = row[col]
            if isinstance(v, float):
                vals.append(f"{v:.4f}" if not np.isnan(v) else "-")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


# The 10 datasets evaluated in RQ3 (Table 10, 11, 12, 13)
RQ3_DATASETS = [
    {"name": "kc1-binary", "id": 1066},
    {"name": "usp05", "id": 1047},
    {"name": "sleuth-ex2016", "id": 862},
    {"name": "calendarDOW", "id": 40663},
    {"name": "mc2", "id": 1054},
    {"name": "fri-c1", "id": 876},
    {"name": "ipums-la-99", "id": 378},
    {"name": "madelon", "id": 1485},
    {"name": "mfeat-fourier", "id": 14},
    {"name": "robot-failures-lp5", "id": 1520},
]

# The 7 unique configurations to evaluate
# Default config (K=1, H=3, update='exponential', M=20) is shared between Table 12 & 13
CONFIGS_TO_RUN = {
    "default": {"n_ants": 20, "update_strategy": "exponential"},   # Table 12: Exponential | Table 13: M=20
    "rank": {"n_ants": 20, "update_strategy": "rank"},             # Table 12: Rank
    "uniform": {"n_ants": 20, "update_strategy": "uniform"},       # Table 12: Uniform
    "m_5": {"n_ants": 5, "update_strategy": "exponential"},         # Table 13: M=5
    "m_10": {"n_ants": 10, "update_strategy": "exponential"},       # Table 13: M=10
    "m_15": {"n_ants": 15, "update_strategy": "exponential"},       # Table 13: M=15
    "m_25": {"n_ants": 25, "update_strategy": "exponential"},       # Table 13: M=25
}


def load_dataset_with_retry(openml_id: int, max_retries: int = 3, retry_delay: float = 3.0):
    for attempt in range(1, max_retries + 1):
        try:
            return load_openml(openml_id)
        except Exception as exc:
            if attempt == max_retries:
                raise
            print(f"[Warning] Failed to fetch OpenML id {openml_id} (attempt {attempt}/{max_retries}): {exc}. Retrying in {retry_delay}s...")
            time.sleep(retry_delay)


def evaluate_dataset_configs(
    beta_model: BETA,
    dataset_info: Dict[str, Any],
    downstream_evaluator: str = "autogluon",
    autogluon_time_limit: int = 60,
    seed: int = 42,
    existing_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    dataset_name = dataset_info["name"]
    openml_id = dataset_info["id"]
    print(f"\n{'='*60}\nEvaluating Dataset: {dataset_name} (OpenML ID: {openml_id})\n{'='*60}")

    results = existing_results or {}

    # Load dataset once
    t0 = time.time()
    dataset = load_dataset_with_retry(openml_id)
    print(f"Loaded dataset {dataset_name}: shape={dataset.X.shape}, num_classes={dataset.y.nunique()} ({time.time()-t0:.2f}s)")

    # 1. Similarity Learning & Heuristic Transfer (Constant across all RQ3 configs: K=1, H=3)
    t1 = time.time()
    query_meta = metafeatures.lookup_or_compute(
        normalize_id(dataset.id),
        beta_model.library.metafeatures_full,
        dataset.X,
        dataset.y,
    )
    query_scaled = beta_model._scale_query(query_meta)
    neighbors, eta = beta_model._transfer(dataset.id, query_scaled)
    transfer_config = beta_model._best_transferred_pipeline(neighbors)
    print(f"Phase 1 & 2 complete: Top neighbor = {neighbors[0][0]} (sim={neighbors[0][1]:.4f}) ({time.time()-t1:.2f}s)")

    # Train / Val / Test split (60 / 20 / 20) with fixed seed
    X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(
        dataset.X, dataset.y, val_ratio=0.2, test_ratio=0.2, seed=seed
    )

    # Search evaluator is proxy (fast LR)
    search_evaluator = evaluate.make_evaluator(
        "proxy", X_train, y_train, X_val, y_val,
        eval_metric="f1_macro",
    )

    # Caches for downstream evaluation to avoid refitting AutoGluon on identical pipelines
    val_cache: Dict[Tuple[Tuple[str, str], ...], Optional[float]] = {}
    test_cache: Dict[Tuple[Tuple[str, str], ...], float] = {}

    def score_downstream_val(cfg_dict: Dict[str, str]) -> Optional[float]:
        key = tuple(sorted(cfg_dict.items()))
        if key not in val_cache:
            score_fn = evaluate.proxy_score if downstream_evaluator == "proxy" else evaluate.autogluon_score
            extra = {} if downstream_evaluator == "proxy" else {"time_limit": autogluon_time_limit}
            val_cache[key] = score_fn(cfg_dict, X_train, y_train, X_val, y_val, eval_metric="f1_macro", **extra)
        return val_cache[key]

    def score_downstream_test(cfg_dict: Dict[str, str]) -> float:
        key = tuple(sorted(cfg_dict.items()))
        if key not in test_cache:
            test_cache[key] = evaluate.test_score(
                cfg_dict,
                downstream_evaluator,
                X_train, y_train, X_test, y_test,
                autogluon_time_limit=autogluon_time_limit,
                eval_metric="f1_macro",
            )
        return test_cache[key]

    # Pre-evaluate transferred pipeline on validation once
    print("Evaluating transferred pipeline P_trans with downstream model...")
    transfer_val_score = score_downstream_val(transfer_config)
    print(f"P_trans val score: {transfer_val_score}")

    # 2. Iterate through configs
    for cfg_key, cfg_params in CONFIGS_TO_RUN.items():
        if cfg_key in results and results[cfg_key].get("score") is not None:
            print(f"Skipping already completed config {cfg_key}: score = {results[cfg_key]['score']:.4f}")
            continue

        print(f"\n--- Running config: {cfg_key} (ants={cfg_params['n_ants']}, update={cfg_params['update_strategy']}) ---")
        t_start = time.time()

        # Run ACO search
        search_results = aco.search(
            OPERATORS,
            eta,
            search_evaluator,
            n_ants=cfg_params["n_ants"],
            n_iterations=10,
            alpha=1.0,
            beta=2.0,
            evaporation=0.2,
            elite_size=3,
            update_strategy=cfg_params["update_strategy"],
            seed=seed,
        )
        if not search_results:
            raise RuntimeError(f"ACO search returned no candidate for config {cfg_key}")
        best_search_config, search_proxy_score = search_results[0]
        t_search = time.time() - t_start

        # Select final pipeline using downstream model (P_search vs P_trans)
        search_val_score = score_downstream_val(best_search_config)
        s_val = search_val_score if search_val_score is not None else -np.inf
        t_val = transfer_val_score if transfer_val_score is not None else -np.inf

        if s_val >= t_val:
            chosen, source = best_search_config, "search"
        else:
            chosen, source = transfer_config, "transfer"

        # Evaluate final pipeline on held-out test split with downstream model
        score = score_downstream_test(chosen)
        t_total = time.time() - t_start

        print(f"Result for {cfg_key}: F1-Macro = {score:.4f} (chosen via: {source}, search_time={t_search:.1f}s, total_time={t_total:.1f}s)")

        results[cfg_key] = {
            "score": round(float(score), 4),
            "source": source,
            "search_time": round(t_search, 2),
            "total_time": round(t_total, 2),
            "pipeline": {k: v for k, v in chosen.items() if k != "name"},
        }

    return results


def format_tables(all_results: Dict[str, Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Table 12: Exponential, Rank, Uniform
    t12_rows = []
    for d in RQ3_DATASETS:
        name = d["name"]
        d_res = all_results.get(name, {})
        exp_score = d_res.get("default", {}).get("score")
        rank_score = d_res.get("rank", {}).get("score")
        uni_score = d_res.get("uniform", {}).get("score")
        t12_rows.append({
            "Dataset": name,
            "Exponential": exp_score,
            "Rank": rank_score,
            "Uniform": uni_score,
        })
    df_12 = pd.DataFrame(t12_rows)
    avg_12 = {
        "Dataset": "Average",
        "Exponential": df_12["Exponential"].mean(skipna=True),
        "Rank": df_12["Rank"].mean(skipna=True),
        "Uniform": df_12["Uniform"].mean(skipna=True),
    }
    df_12_with_avg = pd.concat([df_12, pd.DataFrame([avg_12])], ignore_index=True)

    # Table 13: M=5, 10, 15, 20, 25
    t13_rows = []
    for d in RQ3_DATASETS:
        name = d["name"]
        d_res = all_results.get(name, {})
        m5 = d_res.get("m_5", {}).get("score")
        m10 = d_res.get("m_10", {}).get("score")
        m15 = d_res.get("m_15", {}).get("score")
        m20 = d_res.get("default", {}).get("score")  # IDENTICAL TO Table 12 Exponential!
        m25 = d_res.get("m_25", {}).get("score")
        t13_rows.append({
            "Dataset": name,
            "M = 5": m5,
            "M = 10": m10,
            "M = 15": m15,
            "M = 20": m20,
            "M = 25": m25,
        })
    df_13 = pd.DataFrame(t13_rows)
    avg_13 = {
        "Dataset": "Average",
        "M = 5": df_13["M = 5"].mean(skipna=True),
        "M = 10": df_13["M = 10"].mean(skipna=True),
        "M = 15": df_13["M = 15"].mean(skipna=True),
        "M = 20": df_13["M = 20"].mean(skipna=True),
        "M = 25": df_13["M = 25"].mean(skipna=True),
    }
    df_13_with_avg = pd.concat([df_13, pd.DataFrame([avg_13])], ignore_index=True)

    return df_12_with_avg, df_13_with_avg


def to_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{l" + "c" * (len(df.columns) - 1) + "}",
        "\\toprule",
        " & ".join(df.columns) + " \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        is_avg = row["Dataset"] == "Average"
        if is_avg:
            lines.append("\\midrule")
        vals = []
        for col in df.columns:
            v = row[col]
            if isinstance(v, float):
                vals.append(f"{v:.3f}" if not np.isnan(v) else "-")
            else:
                vals.append(str(v))
        lines.append(" & ".join(vals) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="RQ3 Hyperparameter Ablations for Table 12 & Table 13")
    parser.add_argument("--dataset-idx", type=int, default=None, help="Index of single dataset to run (0-9)")
    parser.add_argument("--dataset", type=str, default=None, help="Name of specific dataset to run")
    parser.add_argument("--downstream", choices=["proxy", "autogluon"], default="autogluon")
    parser.add_argument("--time-limit", type=int, default=300, help="AutoGluon time limit in seconds")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filter target datasets
    if args.dataset_idx is not None:
        if not (0 <= args.dataset_idx < len(RQ3_DATASETS)):
            raise ValueError(f"--dataset-idx must be between 0 and {len(RQ3_DATASETS)-1}")
        selected_datasets = [RQ3_DATASETS[args.dataset_idx]]
        checkpoint_path = out_dir / f"rq3_results_{selected_datasets[0]['name']}.json"
    elif args.dataset is not None:
        matches = [d for d in RQ3_DATASETS if d["name"].lower() == args.dataset.lower()]
        if not matches:
            raise ValueError(f"Unknown dataset {args.dataset}. Options: {[d['name'] for d in RQ3_DATASETS]}")
        selected_datasets = matches
        checkpoint_path = out_dir / f"rq3_results_{selected_datasets[0]['name']}.json"
    else:
        selected_datasets = RQ3_DATASETS
        checkpoint_path = out_dir / "rq3_results_all.json"

    # Load existing checkpoint if present
    all_results: Dict[str, Dict[str, Any]] = {}
    if checkpoint_path.exists() and not args.force_rerun:
        try:
            all_results = json.loads(checkpoint_path.read_text())
            print(f"Loaded existing checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"Could not load existing checkpoint: {e}")

    # Build BETA model
    print(f"Initializing BETA model from pretrained assets...")
    cfg = BETAConfig()
    cfg.eval_metric = "f1_macro"
    cfg.search_evaluator = "proxy"
    cfg.downstream_evaluator = args.downstream
    cfg.autogluon_time_limit = args.time_limit
    cfg.seed = args.seed
    beta_model = BETA.from_pretrained(cfg)

    # Run selected datasets
    for d in selected_datasets:
        dataset_name = d["name"]
        d_existing = {} if args.force_rerun else all_results.get(dataset_name, {})
        d_res = evaluate_dataset_configs(
            beta_model,
            d,
            downstream_evaluator=args.downstream,
            autogluon_time_limit=args.time_limit,
            seed=args.seed,
            existing_results=d_existing,
        )
        all_results[dataset_name] = d_res

        # Save checkpoint after each dataset
        checkpoint_path.write_text(json.dumps(all_results, indent=2))
        print(f"Saved checkpoint to {checkpoint_path}")

    # Format and save tables
    df_12, df_13 = format_tables(all_results)

    print("\n" + "=" * 60)
    print("Table 12: BETA's performance under different ACO pheromone update rules (F1-Macro)")
    print("=" * 60)
    print(df_to_markdown(df_12))

    print("\n" + "=" * 60)
    print("Table 13: Sensitivity to the number of ants M in ACO (F1-Macro)")
    print("=" * 60)
    print(df_to_markdown(df_13))

    # Save CSVs
    prefix = f"_{selected_datasets[0]['name']}" if len(selected_datasets) == 1 else ""
    t12_csv = out_dir / f"table12{prefix}.csv"
    t13_csv = out_dir / f"table13{prefix}.csv"
    df_12.to_csv(t12_csv, index=False)
    df_13.to_csv(t13_csv, index=False)

    # Save LaTeX tables
    latex_t12 = to_latex_table(
        df_12,
        caption="BETA's performance under different ACO pheromone update rules.",
        label="tab:pheromone_update",
    )
    latex_t13 = to_latex_table(
        df_13,
        caption="Sensitivity to the number of ants $M$ in ACO.",
        label="tab:ants_sensitivity",
    )
    (out_dir / f"table12{prefix}.tex").write_text(latex_t12)
    (out_dir / f"table13{prefix}.tex").write_text(latex_t13)

    print(f"\nArtifacts written to:")
    print(f"  - {checkpoint_path}")
    print(f"  - {t12_csv}")
    print(f"  - {t13_csv}")


if __name__ == "__main__":
    main()
