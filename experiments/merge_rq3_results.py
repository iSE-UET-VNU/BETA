"""Merge results from multiple individual Kaggle/local runs into complete Tables 12 and 13."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import functools
print = functools.partial(print, flush=True)

import numpy as np
import pandas as pd

from run_rq3_ablations import RQ3_DATASETS, format_tables, to_latex_table, df_to_markdown


def merge_result_files(input_paths: List[Path]) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for p in input_paths:
        if not p.is_file() or not p.suffix == ".json":
            continue
        try:
            data = json.loads(p.read_text())
            for dataset_name, configs in data.items():
                if dataset_name not in merged:
                    merged[dataset_name] = {}
                for cfg_key, res in configs.items():
                    merged[dataset_name][cfg_key] = res
            print(f"Merged results from {p.name}")
        except Exception as e:
            print(f"Error reading {p}: {e}")
    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge RQ3 ablation results from parallel Kaggle runs")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=".",
        help="Directory containing rq3_results_*.json files or paths",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=[],
        help="Explicit list of json result files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save merged tables",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_files = []
    if args.files:
        input_files = [Path(f) for f in args.files]
    else:
        in_dir = Path(args.input_dir)
        input_files = sorted(in_dir.glob("rq3_results_*.json"))
        # Exclude rq3_results_all.json if looking for per-dataset files
        input_files = [f for f in input_files if f.name != "rq3_results_all.json"]

    if not input_files:
        print(f"No result JSON files found in {args.input_dir}!")
        return

    print(f"Found {len(input_files)} result file(s) to merge: {[f.name for f in input_files]}")
    merged_results = merge_result_files(input_files)

    # Save merged JSON
    merged_json_path = out_dir / "rq3_results_all.json"
    merged_json_path.write_text(json.dumps(merged_results, indent=2))
    print(f"\nSaved combined results to {merged_json_path}")

    # Generate tables
    df_12, df_13 = format_tables(merged_results)

    print("\n" + "=" * 60)
    print("Table 12: BETA's performance under different ACO pheromone update rules (F1-Macro)")
    print("=" * 60)
    print(df_to_markdown(df_12))

    print("\n" + "=" * 60)
    print("Table 13: Sensitivity to the number of ants M in ACO (F1-Macro)")
    print("=" * 60)
    print(df_to_markdown(df_13))

    # Save CSVs
    t12_csv = out_dir / "table12.csv"
    t13_csv = out_dir / "table13.csv"
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
    (out_dir / "table12.tex").write_text(latex_t12)
    (out_dir / "table13.tex").write_text(latex_t13)

    print(f"\nCompleted! Tables written to:")
    print(f"  - {t12_csv}")
    print(f"  - {t13_csv}")
    print(f"  - {out_dir / 'table12.tex'}")
    print(f"  - {out_dir / 'table13.tex'}")


if __name__ == "__main__":
    main()
