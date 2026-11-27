"""Main experiment: BETA vs AutoGluon's default preprocessing on repeated stratified splits.

Writes one JSON line per (dataset, seed, arm). Rerunning the same command skips rows already
recorded as ok, so an interrupted run resumes where it stopped. Failures are recorded, not retried.

    python experiments/run_rq1.py --datasets 1066 876 --seeds 0 1 2 --out results/main.jsonl
    python experiments/run_rq1.py --summarize results/main.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from beta import BETA, BETAConfig
from beta.data import Dataset, load_csv, load_openml, split_indices
from beta.evaluate import (
    InfrastructureError, _sanitize_columns, check_fit_duration, classification_problem_type, compute_metric,
)

ROOT = Path(__file__).resolve().parent.parent
ARMS = ("beta", "default")


def code_fingerprint() -> str:
    """HEAD plus a hash of any uncommitted change under beta/ or experiments/, so results name the exact code."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    head = git("rev-parse", "--short", "HEAD") if (ROOT / ".git").exists() else ""
    if not head:  # no git (e.g. the Kaggle bundle): hash the source files themselves
        files = sorted(list((ROOT / "beta").glob("*.py")) + list((ROOT / "experiments").glob("*.py")))
        digest = hashlib.sha1(b"".join(f.name.encode() + f.read_bytes() for f in files)).hexdigest()[:12]
        return f"files:{digest}"
    dirty = git("diff", "HEAD", "--", "beta", "experiments")
    return f"{head}+{hashlib.sha1(dirty.encode()).hexdigest()[:8]}" if dirty else head


def load_dataset(dataset_id: int, data_dir: str | None, target: str = "target") -> tuple[Dataset, str]:
    """`<data_dir>/<id>.csv` when given (pins the exact table), else OpenML. Returns (dataset, source)."""
    if not data_dir:
        return load_openml(dataset_id), f"openml:{dataset_id}"
    path = Path(data_dir) / f"{dataset_id}.csv"
    sha = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    dataset = load_csv(str(path), target, dataset_id=f"D_{dataset_id}", only_empty_is_missing=True,
                       nominal_columns=nominal_columns(Path(data_dir).parent / "manifest.json", dataset_id))
    return dataset, f"{path.name}:sha256={sha}"


def nominal_columns(manifest: Path, dataset_id: int) -> List[str]:
    """Columns OpenML declares nominal, recorded by prepare_data.py (a CSV loses the types)."""
    if not manifest.exists():
        return []
    entries = {e["id"]: e for e in json.loads(manifest.read_text())}
    return entries.get(dataset_id, {}).get("nominal_columns", [])


class SplitMismatch(RuntimeError):
    """The split computed here differs from the saved one: the environment or the data changed."""


def split_record(out: Path, dataset_id: int, seed: int, split, splits_dir: Optional[Path] = None) -> str:
    """Save the split indices once (for methods run later or elsewhere) and return a short hash.
    If a saved split exists, the recomputed one must be identical, else SplitMismatch."""
    train_idx, val_idx, test_idx = (sorted(int(i) for i in part) for part in split)
    payload = {"train": train_idx, "val": val_idx, "test": test_idx}
    digest = hashlib.sha1(json.dumps(payload).encode()).hexdigest()[:12]
    path = Path(splits_dir or out.parent / "splits") / str(dataset_id) / f"seed{seed}.json"
    if path.exists():
        saved = hashlib.sha1(json.dumps(json.loads(path.read_text())).encode()).hexdigest()[:12]
        if saved != digest:
            raise SplitMismatch(f"dataset {dataset_id} seed {seed}: split {digest} differs from saved {saved} ({path})")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
    return digest


def run_default(X_train, y_train, X_test, y_test, time_limit: int, eval_metric: str, keep_predictions: bool = False) -> Dict:
    """AutoGluon with its own feature generator on the raw split: the paper's 'Default' column."""
    from autogluon.tabular import TabularPredictor

    train_df = X_train.copy()
    train_df["target"] = y_train.values
    test_df, train_df = _sanitize_columns(X_test, train_df)
    workdir = Path(tempfile.gettempdir()) / f"beta_default_{uuid.uuid4().hex}"
    try:
        predictor = TabularPredictor(
            label="target", path=str(workdir), problem_type=classification_problem_type(y_train),
            eval_metric=eval_metric, verbosity=0, learner_kwargs={"random_state": 42},
        )
        started = time.time()
        predictor.fit(train_data=train_df, time_limit=time_limit, presets="best_quality", dynamic_stacking=False)
        check_fit_duration(time.time() - started, time_limit)
        preds = predictor.predict(test_df)
        out = {"score": compute_metric(y_test.reset_index(drop=True), preds.reset_index(drop=True), eval_metric),
               "models": predictor.model_names()}
        if keep_predictions:
            out["predictions"] = {"y_true": [v.item() if hasattr(v, "item") else v for v in y_test.reset_index(drop=True)],
                                  "y_pred": [v.item() if hasattr(v, "item") else v for v in preds.reset_index(drop=True)]}
        return out
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def run_beta(model: BETA, dataset: Dataset, cfg: BETAConfig) -> Dict:
    beta = BETA(model.library, model.encoder, model.pipeline_configs, cfg)
    rec = beta.recommend(X=dataset.X, y=dataset.y, dataset_id=dataset.id)
    return {
        "score": rec.score, "pipeline": rec.pipeline, "source": rec.source,
        "neighbors": [[str(d), float(s)] for d, s in rec.neighbors],
        "search_evaluations": rec.search_evaluations, "search_seconds": rec.search_seconds,
        "search_hit_time_budget": rec.search_hit_time_budget,
        **({"predictions": rec.predictions} if rec.predictions else {}),
    }


def make_config(seed: int, args: argparse.Namespace) -> BETAConfig:
    cfg = BETAConfig()
    cfg.seed = seed
    cfg.downstream_evaluator = "autogluon"
    cfg.autogluon_time_limit = args.time_limit
    cfg.eval_metric = args.metric
    cfg.store_predictions = bool(getattr(args, "predictions", False))
    for item in args.set or []:
        key, _, value = item.partition("=")
        cfg.override(key, value)
    return cfg


INFRA_ATTEMPTS = 3  # attempts of one run before an infrastructure failure is accepted as final


def run_states(out: Path) -> Dict[tuple, tuple]:
    """(dataset, seed, arm) -> (latest status, number of attempts started)."""
    states: Dict[tuple, tuple] = {}
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                key = (r["dataset"], r["seed"], r["arm"])
                attempts = states.get(key, (None, 0))[1] + (r["status"] == "started")
                states[key] = (r["status"], attempts)
    return states


def done_keys(out: Path, retry_errors: bool = False) -> set:
    """Runs that need no further attempt.

    ok: done. error (the method failed): done, unless retry_errors is set for a runner bug.
    started (the process died mid-run) or infra_error (suspended, out of memory): retried with
    identical settings, up to INFRA_ATTEMPTS attempts. Every retry stays in the file for disclosure.
    """
    done = set()
    for key, (status, attempts) in run_states(out).items():
        if status == "ok" or (status == "error" and not retry_errors) or attempts >= INFRA_ATTEMPTS:
            done.add(key)
    return done


def parse_window(text: str) -> tuple:
    """'08:15-11:30' -> (495, 690) minutes after midnight."""
    start, end = (int(h) * 60 + int(m) for h, m in (part.split(":") for part in text.split("-")))
    return start, end


def blocked_reason(now: datetime, windows: Sequence[tuple], margin_minutes: int, pause_file: Path) -> Optional[str]:
    """Why a new run must not start now: a pause file, or a run started now could reach a quiet window."""
    if pause_file.exists():
        return f"pause file {pause_file} exists"
    minute = now.hour * 60 + now.minute
    for start, end in windows:
        opens = start - margin_minutes
        if (minute - opens) % 1440 < (end - opens) % 1440:
            return f"quiet window {start // 60:02d}:{start % 60:02d}-{end // 60:02d}:{end % 60:02d}"
    return None


def wait_until_allowed(args: argparse.Namespace, pause_file: Path) -> None:
    windows = [parse_window(w) for w in args.quiet or []]
    announced = None
    while (reason := blocked_reason(datetime.now(), windows, args.margin_minutes, pause_file)) is not None:
        if reason != announced:
            print(f"{datetime.now():%Y-%m-%d %H:%M} waiting: {reason}", flush=True)
            announced = reason
        time.sleep(60)


def run(args: argparse.Namespace) -> None:
    import autogluon.tabular

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pause_file = out.parent / "PAUSE"
    finished = done_keys(out, args.retry_errors)
    model = BETA.from_pretrained()
    meta = {
        "code": code_fingerprint(), "autogluon": autogluon.tabular.__version__,
        "python": platform.python_version(), "machine": f"{platform.machine()} x{os.cpu_count()}",
        "time_limit": args.time_limit, "metric": args.metric, "overrides": args.set or [],
    }
    for dataset_id in args.datasets:
        dataset = source = None
        for seed in args.seeds:
            for arm in args.arms:
                if (dataset_id, seed, arm) in finished:
                    continue
                if dataset is None:
                    dataset, source = load_dataset(dataset_id, args.data_dir)
                cfg = make_config(seed, args)
                split = split_indices(dataset.y, cfg.split.val_ratio, cfg.split.test_ratio, seed, cfg.split.stratify)
                X_tr, X_va, X_te = (dataset.X.iloc[idx].reset_index(drop=True) for idx in split)
                y_tr, _, y_te = (dataset.y.iloc[idx].reset_index(drop=True) for idx in split)
                base = {"dataset": dataset_id, "seed": seed, "arm": arm, "data": source, **meta,
                        "split": split_record(out, dataset_id, seed, split, args.splits_dir),
                        "problem_type": classification_problem_type(y_tr),
                        "n_train": len(X_tr), "n_val": len(X_va), "n_test": len(X_te)}
                attempts = run_states(out).get((dataset_id, seed, arm), (None, 0))[1]
                while attempts < INFRA_ATTEMPTS:
                    wait_until_allowed(args, pause_file)
                    attempts += 1
                    append(out, {**base, "status": "started", "attempt": attempts, "at": f"{datetime.now():%Y-%m-%d %H:%M:%S}"})
                    row = {**base, "attempt": attempts}
                    start = time.time()
                    try:
                        detail = run_beta(model, dataset, cfg) if arm == "beta" else run_default(
                            X_tr, y_tr, X_te, y_te, args.time_limit, args.metric, args.predictions)
                        preds = detail.pop("predictions", None)
                        if preds:
                            row.update(save_predictions(out, dataset_id, seed, arm, preds))
                        row.update(status="ok", **detail)
                    except (InfrastructureError, MemoryError) as err:  # the machine failed: retried as is
                        row.update(status="infra_error", error=f"{type(err).__name__}: {err}")
                    except Exception as err:  # the method failed: recorded, never rerun with other settings
                        row.update(status="error", error=f"{type(err).__name__}: {err}", trace=traceback.format_exc()[-2000:])
                    row["seconds"] = round(time.time() - start, 1)
                    append(out, row)
                    print(f"{datetime.now():%H:%M} {dataset_id} seed={seed} {arm:7} {row['status']:11} "
                          f"score={row.get('score', float('nan')):.4f} {row['seconds']:.0f}s", flush=True)
                    if row["status"] != "infra_error":
                        break
    if all((d, s, a) in done_keys(out) for d in args.datasets for s in args.seeds for a in args.arms):
        (out.parent / f"{out.stem}.done").write_text(f"{datetime.now():%Y-%m-%d %H:%M:%S}\n")
        print("all requested runs are recorded", flush=True)


def save_predictions(out: Path, dataset_id: int, seed: int, arm: str, preds: Dict[str, list]) -> Dict:
    """Write y_true / y_pred to predictions/<out stem>/<id>_seed<k>_<arm>.json.gz; return both metrics."""
    import gzip
    from sklearn.metrics import accuracy_score, f1_score
    path = out.parent / "predictions" / out.stem / f"{dataset_id}_seed{seed}_{arm}.json.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as fh:
        json.dump(preds, fh)
    y_true, y_pred = [str(v) for v in preds["y_true"]], [str(v) for v in preds["y_pred"]]
    return {"predictions_file": str(path.relative_to(out.parent)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro"))}


def append(out: Path, row: Dict) -> None:
    with out.open("a") as fh:
        fh.write(json.dumps(row) + "\n")


def summarize(path: str) -> pd.DataFrame:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    retried = df[df.status.isin(["started", "infra_error"])]
    df = df.drop_duplicates(["dataset", "seed", "arm"], keep="last")  # latest attempt per run
    if "error" not in df:
        df["error"] = None
    df.loc[df.status == "started", "error"] = "process died during the run"
    failed = df[df.status != "ok"]
    ok = df[df.status == "ok"]
    wide = ok.pivot_table(index=["dataset", "seed"], columns="arm", values="score")
    lines = []
    for dataset_id, block in wide.groupby(level="dataset"):
        rec = {"dataset": dataset_id}
        for arm in ARMS:
            if arm in block:
                vals = block[arm].dropna()
                rec.update({f"{arm}_mean": vals.mean(), f"{arm}_sd": vals.std(ddof=1) if len(vals) > 1 else np.nan,
                            f"{arm}_min": vals.min(), f"{arm}_max": vals.max(), f"{arm}_n": len(vals)})
        if set(ARMS) <= set(block.columns):
            diff = (block["beta"] - block["default"]).dropna()
            rec.update(diff_mean=diff.mean(), wins=int((diff > 1e-9).sum()), ties=int((diff.abs() <= 1e-9).sum()),
                       losses=int((diff < -1e-9).sum()))
        lines.append(rec)
    table = pd.DataFrame(lines)
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        print(table.to_string(index=False))
    if len(failed):
        print(f"\n{len(failed)} failed run(s):")
        print(failed[["dataset", "seed", "arm", "status", "error"]].to_string(index=False))
    extra = retried.groupby(["dataset", "seed", "arm"]).size() - 1  # one "started" row per attempt is normal
    if (extra > 0).any():
        print(f"\nruns redone after an infrastructure failure (disclose): {int((extra > 0).sum())}")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", type=int, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--time-limit", type=int, default=300, help="seconds per AutoGluon fit")
    parser.add_argument("--metric", choices=["accuracy", "f1_macro"], default="accuracy")
    parser.add_argument("--set", action="append", help="config override for the BETA arm, e.g. aco.n_ants=25")
    parser.add_argument("--data-dir", help="folder of <id>.csv files with a 'target' column; default: fetch from OpenML")
    parser.add_argument("--retry-errors", action="store_true",
                        help="also rerun runs recorded as method failures (only after fixing a runner bug; disclose)")
    parser.add_argument("--quiet", action="append", metavar="HH:MM-HH:MM",
                        help="local time window in which no run may be in progress (repeatable)")
    parser.add_argument("--margin-minutes", type=int, default=40,
                        help="do not start a run this many minutes before a quiet window")
    parser.add_argument("--out", default="results/rq1.jsonl")
    parser.add_argument("--splits-dir", type=Path, default=None,
                        help="saved splits to reuse and check (default: <out folder>/splits)")
    parser.add_argument("--predictions", action="store_true",
                        help="store test predictions and record accuracy and f1_macro for every run")
    parser.add_argument("--summarize", metavar="JSONL")
    args = parser.parse_args()
    if args.summarize:
        summarize(args.summarize)
    elif args.datasets:
        run(args)
    else:
        parser.error("give --datasets to run, or --summarize JSONL")


if __name__ == "__main__":
    main()
