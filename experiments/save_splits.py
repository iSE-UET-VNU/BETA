"""Compute the splits of the given datasets and seeds exactly as run_rq1.py does, save them (or check them against
the saved files: a difference raises SplitMismatch), and print one hash per split.
Run: python experiments/save_splits.py --datasets 1066 ... --seeds 0 1 2 3 4 --data-dir data/eval --splits-dir results/splits"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_rq1  # noqa: E402
from beta.data import split_indices  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--datasets", type=int, nargs="+", required=True)
p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
p.add_argument("--data-dir", required=True)
p.add_argument("--splits-dir", type=Path, required=True)
a = p.parse_args()
for d in a.datasets:
    ds, _ = run_rq1.load_dataset(d, a.data_dir)
    for s in a.seeds:
        cfg = run_rq1.BETAConfig(); cfg.seed = s
        split = split_indices(ds.y, cfg.split.val_ratio, cfg.split.test_ratio, s, cfg.split.stratify)
        print(d, s, run_rq1.split_record(a.splits_dir / "x.jsonl", d, s, split, a.splits_dir), flush=True)
