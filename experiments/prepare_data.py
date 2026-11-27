"""Build the 30 evaluation datasets of Table 2 from their original sources.

Writes data/eval/<id>.csv (original columns plus a `target` column) and data/manifest.json
(source, version, shape, class count and SHA-256 of every file). No rows are capped and no
classes are filtered.

    python experiments/prepare_data.py
    python experiments/prepare_data.py --only 378 1047
    python experiments/prepare_data.py --compare-diffprep     # also diff against DiffPrep's CSVs
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
from sklearn.datasets import fetch_openml

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "eval"
MANIFEST = ROOT / "data" / "manifest.json"

DIFFPREP_COMMIT = "6499edfbe2358288fbd0bde75f366f76da24ac27"
DIFFPREP_URL = "https://raw.githubusercontent.com/chu-data-lab/DiffPrep/" + DIFFPREP_COMMIT + "/data/{folder}/{file}"

# name, id, Table 2 (#S, #F, #C), how to load.
# Table 2 counts the target among the features for rows 14-30 but not for rows 1-13.
DATASETS = [
    ("kc1-binary", 1066, (145, 94, 2), {}),
    ("usp05", 1047, (203, 16, 11), {}),
    ("sleuth-ex2016", 862, (87, 10, 2), {}),
    ("calendarDOW", 40663, (399, 32, 5), {}),
    ("mc2", 1054, (161, 39, 2), {}),
    ("fri-c1", 876, (100, 50, 2), {}),
    ("mfeat-morphological", 18, (2000, 6, 10), {}),
    ("robot-failures-lp5", 1520, (164, 90, 5), {}),
    ("autoUniv-au4", 1548, (2500, 100, 3), {}),
    ("ipums-la-99", 378, (8844, 60, 7), {}),
    ("madelon", 1485, (2600, 500, 2), {}),
    ("mfeat-fourier", 14, (2000, 76, 10), {}),
    ("colic", 27, (368, 22, 2), {}),
    ("abalone", 44956, (4177, 9, 28), {"diffprep": "abalone"}),  # rings used as class labels
    ("ada_prior", 1037, (4562, 15, 2), {"diffprep": "ada_prior"}),
    ("avila", 42932, (20867, 11, 12), {"target": "10", "drop": ["train", "test"], "diffprep": "avila"}),
    ("connect-4", 40668, (67557, 43, 3), {"diffprep": "connect-4"}),
    ("eeg", 1471, (14980, 15, 2), {"diffprep": "eeg"}),
    # not on OpenML; Table 2's row repeats connect-4's numbers (the file has 9,367 rows, 8 features, 2 classes)
    ("google", 100000, (67557, 43, 3), {"source": "diffprep", "diffprep": "google"}),
    ("house", 42165, (1460, 81, 2), {"binarize": ("SalePrice", 150000), "diffprep": "house_prices"}),
    ("jungle_chess", 41027, (44819, 7, 3), {"diffprep": "jungle_chess_2pcs_raw_endgame_complete"}),
    ("micro", 41671, (20000, 21, 5), {"diffprep": "microaggregation2"}),
    ("mozilla4", 1046, (15545, 6, 2), {"diffprep": "mozilla4"}),
    ("obesity", 46597, (2111, 17, 7), {"diffprep": "obesity"}),
    ("page-blocks", 30, (5473, 11, 5), {"diffprep": "page-blocks"}),
    ("pbcseq", 802, (1945, 19, 2), {"diffprep": "pbcseq"}),
    ("pol", 722, (15000, 49, 2), {"diffprep": "pol"}),
    ("run_or_walk", 40922, (88588, 7, 2), {"diffprep": "Run_or_walk_information"}),
    ("uscensus", 1119, (32561, 15, 2), {"diffprep": "USCensus"}),
    ("wall-robot-nav", 1497, (5456, 25, 4), {"diffprep": "wall-robot-navigation"}),
]


def load_diffprep(folder: str) -> tuple[pd.DataFrame, pd.Series]:
    with urlopen(DIFFPREP_URL.format(folder=folder, file="info.json")) as response:
        label = json.loads(response.read())["label"]
    frame = pd.read_csv(DIFFPREP_URL.format(folder=folder, file="data.csv"), keep_default_na=False, na_values=[""])
    return frame.drop(columns=[label]), frame[label]


def load_openml_source(dataset_id: int, how: dict) -> tuple[pd.DataFrame, pd.Series, str, list]:
    bunch = fetch_openml(data_id=dataset_id, as_frame=True, parser="auto", target_column=None, n_retries=6, delay=10.0)
    frame = bunch.frame.drop(columns=how.get("drop", []))
    target = how.get("target") or bunch.details["default_target_attribute"]
    if "binarize" in how:
        column, threshold = how["binarize"]
        y = (frame[column] > threshold).astype(int)
        target = column
    else:
        y = frame[target]
    X = frame.drop(columns=[target])
    nominal = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
    return X, y, f"openml:{dataset_id} v{bunch.details['version']}", nominal


def compare(X: pd.DataFrame, y: pd.Series, X_ref: pd.DataFrame, y_ref: pd.Series) -> dict:
    same_classes = sorted(y.value_counts().tolist()) == sorted(y_ref.value_counts().tolist())
    return {"rows": [len(X), len(X_ref)], "features": [X.shape[1], X_ref.shape[1]],
            "class_counts_equal": same_classes,
            "identical": bool(len(X) == len(X_ref) and X.shape[1] == X_ref.shape[1] and same_classes)}


def build(name: str, dataset_id: int, table2: tuple, how: dict, compare_diffprep: bool) -> dict:
    if how.get("source") == "diffprep":
        X, y = load_diffprep(how["diffprep"])
        source, nominal = f"diffprep@{DIFFPREP_COMMIT[:7]}:{how['diffprep']}", []  # a CSV has no declared types
    else:
        X, y, source, nominal = load_openml_source(dataset_id, how)
    X = X.reset_index(drop=True)
    frame = X.assign(target=y.reset_index(drop=True).values)
    path = OUT_DIR / f"{dataset_id}.csv"
    frame.to_csv(path, index=False)

    rows, features, classes = len(X), X.shape[1], int(y.nunique())
    t_rows, t_features, t_classes = table2
    entry = {
        "name": name, "id": dataset_id, "source": source, "file": f"data/eval/{dataset_id}.csv",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": rows, "features": features, "classes": classes, "nominal_columns": nominal,
        "table2": {"rows": t_rows, "features": t_features, "classes": t_classes},
        "matches_table2": rows == t_rows and classes == t_classes and features in (t_features, t_features - 1),
    }
    if compare_diffprep and "diffprep" in how and how.get("source") != "diffprep":
        entry["diffprep_check"] = compare(X, y, *load_diffprep(how["diffprep"]))
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", type=int, nargs="+", help="dataset ids to rebuild")
    parser.add_argument("--compare-diffprep", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {e["id"]: e for e in json.loads(MANIFEST.read_text())} if MANIFEST.exists() else {}
    for name, dataset_id, table2, how in DATASETS:
        if args.only and dataset_id not in args.only:
            continue
        entry = build(name, dataset_id, table2, how, args.compare_diffprep)
        manifest[dataset_id] = entry
        check = entry.get("diffprep_check", {})
        print(f"{dataset_id:>6} {name:20} {entry['rows']:>6} x {entry['features']:<4} {entry['classes']:>2} classes  "
              f"table2={'ok' if entry['matches_table2'] else 'DIFFERS'}"
              + (f"  diffprep={'same' if check.get('identical') else 'DIFFERS ' + json.dumps(check)}" if check else ""),
              flush=True)
    order = [d for _, d, _, _ in DATASETS]
    MANIFEST.write_text(json.dumps([manifest[d] for d in order if d in manifest], indent=2) + "\n")


if __name__ == "__main__":
    main()
