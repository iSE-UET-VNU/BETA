import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
run_rq1 = pytest.importorskip("run_rq1")


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _row(dataset, seed, arm, score=None, status="ok"):
    row = {"dataset": dataset, "seed": seed, "arm": arm, "status": status}
    if status == "ok":
        row["score"] = score
    else:
        row["error"] = "RuntimeError: boom"
    return row


def test_summary_pairs_arms_by_seed(tmp_path, capsys):
    out = tmp_path / "r.jsonl"
    _write(out, [
        _row(1, 0, "beta", 0.80), _row(1, 0, "default", 0.70),
        _row(1, 1, "beta", 0.60), _row(1, 1, "default", 0.70),
        _row(1, 2, "beta", 0.70), _row(1, 2, "default", 0.70),
    ])
    table = run_rq1.summarize(str(out)).set_index("dataset")
    assert table.loc[1, "beta_mean"] == pytest.approx(0.70)
    assert table.loc[1, "beta_sd"] == pytest.approx(0.10)
    assert (table.loc[1, "wins"], table.loc[1, "ties"], table.loc[1, "losses"]) == (1, 1, 1)


def test_failed_runs_are_reported_and_left_out_of_the_means(tmp_path, capsys):
    out = tmp_path / "r.jsonl"
    _write(out, [_row(1, 0, "beta", 0.9), _row(1, 1, "beta", status="error")])
    table = run_rq1.summarize(str(out)).set_index("dataset")
    assert table.loc[1, "beta_n"] == 1
    assert "1 failed run(s)" in capsys.readouterr().out


def test_csv_dataset_is_pinned_by_hash_and_keeps_the_id(tmp_path):
    (tmp_path / "876.csv").write_text("a,b,target\n1,x,yes\n2,y,no\n3,x,yes\n")
    dataset, source = run_rq1.load_dataset(876, str(tmp_path))
    assert dataset.id == "D_876" and list(dataset.X.columns) == ["a", "b"] and len(dataset.y) == 3
    assert source.startswith("876.csv:sha256=")


def test_labels_named_null_or_none_are_kept_as_classes(tmp_path):
    # usp05 has a class called NULL; house_prices uses None as a category
    (tmp_path / "1047.csv").write_text("a,b,target\n1,None,NULL\n2,x,BC/S\n3,,NULL\n4,x,\n")
    dataset, _ = run_rq1.load_dataset(1047, str(tmp_path))
    assert len(dataset.y) == 3                     # only the row with an empty label is dropped
    assert dataset.y.nunique() == 2                # NULL is a class
    assert dataset.X["b"].tolist()[:2] == ["None", "x"]  # "None" is a category value, not missing


def test_resume_does_not_silently_rerun_recorded_failures(tmp_path):
    out = tmp_path / "r.jsonl"
    _write(out, [_row(1, 0, "beta", 0.9), _row(1, 1, "beta", status="error")])
    assert run_rq1.done_keys(out) == {(1, 0, "beta"), (1, 1, "beta")}
    assert run_rq1.done_keys(out, retry_errors=True) == {(1, 0, "beta")}


def test_summary_uses_the_latest_attempt_of_a_run(tmp_path, capsys):
    out = tmp_path / "r.jsonl"
    _write(out, [_row(1, 0, "beta", status="error"), _row(1, 0, "beta", 0.8)])
    table = run_rq1.summarize(str(out)).set_index("dataset")
    assert table.loc[1, "beta_n"] == 1 and "failed run" not in capsys.readouterr().out


def test_split_is_saved_once_and_hashed(tmp_path):
    out = tmp_path / "r.jsonl"
    split = ([3, 1], [2], [0])
    digest = run_rq1.split_record(out, 876, 0, split)
    saved = json.loads((tmp_path / "splits" / "876" / "seed0.json").read_text())
    assert saved == {"train": [1, 3], "val": [2], "test": [0]}
    assert run_rq1.split_record(out, 876, 0, split) == digest and len(digest) == 12


def test_nominal_columns_from_the_manifest_are_read_as_categories(tmp_path):
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "862.csv").write_text("code,x,target\n0,1.5,a\n2,2.5,b\n")
    (tmp_path / "manifest.json").write_text(json.dumps([{"id": 862, "nominal_columns": ["code"]}]))
    dataset, _ = run_rq1.load_dataset(862, str(tmp_path / "eval"))
    assert dataset.X["code"].tolist() == ["0", "2"]


def test_a_run_that_died_or_was_suspended_is_retried_up_to_three_times(tmp_path):
    out = tmp_path / "r.jsonl"
    _write(out, [{"dataset": 1, "seed": 0, "arm": "beta", "status": "started"}])
    assert (1, 0, "beta") not in run_rq1.done_keys(out)  # process died: retry
    _write(out, [{"dataset": 1, "seed": 0, "arm": "beta", "status": s} for s in ("started", "infra_error")] * 2)
    assert (1, 0, "beta") not in run_rq1.done_keys(out)  # two attempts so far
    _write(out, [{"dataset": 1, "seed": 0, "arm": "beta", "status": "started"}] * 3)
    assert (1, 0, "beta") in run_rq1.done_keys(out)  # three attempts: accepted as a failure


def test_quiet_window_blocks_runs_that_could_overlap_it(tmp_path):
    from datetime import datetime
    window = [run_rq1.parse_window("09:00-11:30")]
    pause = tmp_path / "PAUSE"
    at = lambda h, m: datetime(2026, 11, 22, h, m)
    assert run_rq1.blocked_reason(at(8, 0), window, 40, pause) is None
    assert run_rq1.blocked_reason(at(8, 30), window, 40, pause)  # would still be running at 09:00
    assert run_rq1.blocked_reason(at(10, 0), window, 40, pause)
    assert run_rq1.blocked_reason(at(11, 30), window, 40, pause) is None
    pause.touch()
    assert "pause file" in run_rq1.blocked_reason(at(14, 0), window, 40, pause)


def test_quiet_window_across_midnight():
    from datetime import datetime
    window = [run_rq1.parse_window("23:30-01:00")]
    assert run_rq1.blocked_reason(datetime(2026, 11, 22, 0, 15), window, 0, Path("/nonexistent/PAUSE"))
    assert run_rq1.blocked_reason(datetime(2026, 11, 22, 12, 0), window, 0, Path("/nonexistent/PAUSE")) is None
