"""`beta recommend | build-matrix | train-similarity`"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from . import BETA, BETAConfig
from .config import DEFAULT_METAFEATURES, DEFAULT_PERFORMANCE_MATRIX, DEFAULT_REFERENCE_PIPELINES


def _load_config(args: argparse.Namespace) -> BETAConfig:
    cfg = BETAConfig.from_yaml(args.config) if getattr(args, "config", None) else BETAConfig()
    for item in getattr(args, "set", None) or []:
        key, _, value = item.partition("=")
        cfg.override(key, value)
    return cfg


def cmd_recommend(args: argparse.Namespace) -> None:
    cfg = _load_config(args)
    if args.downstream:
        cfg.downstream_evaluator = args.downstream
    model = BETA.from_pretrained(cfg)
    if args.openml is not None:
        result = model.recommend(openml_id=args.openml)
    else:
        df = pd.read_csv(args.csv)
        X, y = df.drop(columns=[args.target]), df[args.target]
        result = model.recommend(X=X, y=y, dataset_id=args.dataset_id)

    print(json.dumps({"pipeline": result.pipeline, "source": result.source, "score": result.score, "eta": result.eta}, indent=2))


def cmd_build_matrix(args: argparse.Namespace) -> None:
    from . import evaluate
    from .data import load_openml, split_train_val_test

    pipeline_configs = json.loads(DEFAULT_REFERENCE_PIPELINES.read_text())
    dataset_ids = [line.strip() for line in open(args.datasets) if line.strip()]
    rows: dict = {}
    for dataset_id in dataset_ids:
        dataset = load_openml(int(dataset_id))
        X_train, y_train, X_val, y_val, _, _ = split_train_val_test(dataset.X, dataset.y)
        rows[f"D_{dataset_id}"] = {
            cfg["name"]: evaluate.autogluon_score(cfg, X_train, y_train, X_val, y_val) for cfg in pipeline_configs
        }
        print(f"done: {dataset_id}", file=sys.stderr)
    pd.DataFrame(rows).to_csv(args.out)


def cmd_train_similarity(args: argparse.Namespace) -> None:
    from .knowledge import ReferenceLibrary
    from .similarity import train_similarity_encoder

    perf = pd.read_csv(args.matrix, index_col=0)
    meta = pd.read_csv(args.metafeatures or DEFAULT_METAFEATURES, index_col=0)
    library = ReferenceLibrary.build(perf, meta)
    encoder = train_similarity_encoder(library.metafeatures, library.performance_matrix_imputed, seed=args.seed)
    encoder.save(args.out)
    print(f"saved similarity encoder to {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="beta")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("recommend")
    p.add_argument("--openml", type=int)
    p.add_argument("--csv")
    p.add_argument("--target")
    p.add_argument("--dataset-id")
    p.add_argument("--config")
    p.add_argument("--set", action="append")
    p.add_argument("--downstream", choices=["proxy", "autogluon"])
    p.set_defaults(func=cmd_recommend)

    p = sub.add_parser("build-matrix")
    p.add_argument("--datasets", required=True, help="text file, one OpenML id per line")
    p.add_argument("--out", default=str(DEFAULT_PERFORMANCE_MATRIX))
    p.set_defaults(func=cmd_build_matrix)

    p = sub.add_parser("train-similarity")
    p.add_argument("--matrix", default=str(DEFAULT_PERFORMANCE_MATRIX))
    p.add_argument("--metafeatures")
    p.add_argument("--out", default="assets/similarity_encoder.pt")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_train_similarity)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
