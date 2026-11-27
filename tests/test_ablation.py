"""RQ2 variants (Tables 5-9), the alternative optimizers, prediction storage and the saved-split check."""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

import beta
from beta import BETA, BETAConfig, optimizers, search_space, restrict
from beta.config import OPERATORS, CROSS_FEATURE_STEPS

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
import run_rq1  # noqa: E402


def score_by_index(cfg):
    """Deterministic toy objective: higher operator index = better."""
    return float(sum(OPERATORS[s].index(cfg[s]) for s in OPERATORS))


def toy_evaluator(calls=None):
    def evaluate(configs):
        if calls is not None:
            calls.extend(dict(c) for c in configs)
        return [(c, score_by_index(c)) for c in configs]
    return evaluate


@pytest.mark.parametrize("name", optimizers.OPTIMIZERS)
def test_optimizer_respects_budget_space_and_order(name):
    calls = []
    out = optimizers.search(name, dict(OPERATORS), toy_evaluator(calls), budget=40, seed=3)
    assert 0 < len(out) <= 40
    scores = [s for _, s in out]
    assert scores == sorted(scores, reverse=True)
    keys = {tuple(sorted(c.items())) for c, _ in out}
    assert len(keys) == len(out)                                  # distinct pipelines
    assert all(c[s] in OPERATORS[s] for c, _ in out for s in OPERATORS)
    assert len(calls) == len(out)                                 # each pipeline evaluated once
    again = optimizers.search(name, dict(OPERATORS), toy_evaluator(), budget=40, seed=3)
    assert [c for c, _ in again] == [c for c, _ in out]           # seeded


@pytest.mark.parametrize("name", optimizers.OPTIMIZERS)
def test_optimizer_stops_when_every_evaluation_fails(name):
    stats = {}
    assert optimizers.search(name, dict(OPERATORS), lambda cfgs: [], budget=30, seed=0, stats=stats) == []
    assert stats["evaluations"] == 0


def test_unknown_optimizer_and_ablation_value_are_rejected():
    with pytest.raises(ValueError):
        optimizers.search("hill", dict(OPERATORS), toy_evaluator())
    cfg = BETAConfig()
    cfg.override("ablation.mode", "half")
    with pytest.raises(ValueError):
        cfg.ablation.validate()


def test_no_cross_feature_space_fixes_those_steps_to_none():
    space = search_space("no_cross_feature")
    assert all(space[s] == ["none"] for s in CROSS_FEATURE_STEPS)
    assert {s: v for s, v in space.items() if s not in CROSS_FEATURE_STEPS} == {
        s: v for s, v in OPERATORS.items() if s not in CROSS_FEATURE_STEPS}
    full = {s: v[-1] for s, v in OPERATORS.items()}
    assert all(restrict(full, space)[s] == "none" for s in CROSS_FEATURE_STEPS)


@pytest.fixture(scope="module")
def model():
    return BETA.from_pretrained()


@pytest.fixture(scope="module")
def data():
    X, y = make_classification(n_samples=160, n_features=6, n_informative=4, random_state=0)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(6)]), pd.Series(y)


def recommend(model, data, monkeypatch=None, calls=None, **ablation):
    cfg = BETAConfig()
    cfg.aco.n_ants, cfg.aco.n_iterations = 5, 3      # small search: these are logic tests
    for k, v in ablation.items():
        cfg.override(f"ablation.{k}", v)
    if calls is not None:
        real = beta.evaluate.make_evaluator
        def spy(*a, **kw):
            inner = real(*a, **kw)
            return lambda cfgs: (calls.extend(dict(c) for c in cfgs), inner(cfgs))[1]
        monkeypatch.setattr(beta.evaluate, "make_evaluator", spy)
    X, y = data
    return BETA(model.library, model.encoder, model.pipeline_configs, cfg).recommend(X=X, y=y, dataset_id="query")


def test_transfer_only_returns_the_transferred_pipeline_without_search(model, data, monkeypatch):
    monkeypatch.setattr(beta.aco, "search", lambda *a, **k: pytest.fail("ACO must not run"))
    rec = recommend(model, data, mode="transfer_only")
    assert rec.source == "transfer" and rec.search_evaluations == 0


def test_search_only_uses_uniform_eta_and_skips_eq14(model, data, monkeypatch):
    monkeypatch.setattr(beta.evaluate, "select_final", lambda *a, **k: pytest.fail("Eq. 14 must not run"))
    rec = recommend(model, data, mode="search_only")
    assert rec.source == "search" and rec.search_evaluations > 0
    assert all(set(v) == {1.0} for v in rec.eta.values())


def test_uniform_heuristic_keeps_eq14(model, data, monkeypatch):
    called = []
    real = beta.evaluate.select_final
    monkeypatch.setattr(beta.evaluate, "select_final", lambda *a, **k: (called.append(1), real(*a, **k))[1])
    rec = recommend(model, data, heuristic="uniform")
    assert called and all(set(v) == {1.0} for v in rec.eta.values())


def test_other_optimizer_replaces_aco(model, data, monkeypatch):
    monkeypatch.setattr(beta.aco, "search", lambda *a, **k: pytest.fail("ACO must not run"))
    rec = recommend(model, data, optimizer="ga")
    assert 0 < rec.search_evaluations <= 15


def test_no_cross_feature_never_evaluates_those_operators(model, data, monkeypatch):
    calls = []
    rec = recommend(model, data, monkeypatch, calls, operator_space="no_cross_feature")
    assert calls and all(c[s] == "none" for c in calls for s in CROSS_FEATURE_STEPS)
    assert all(rec.pipeline[s] == "none" for s in CROSS_FEATURE_STEPS)


def test_metafeature_similarity_does_not_use_the_encoder(model, data, monkeypatch):
    monkeypatch.setattr(model.encoder, "embed", lambda *a, **k: pytest.fail("encoder must not be used"))
    rec = recommend(model, data, similarity="metafeature_cosine")
    assert rec.neighbors and -1.0 <= rec.neighbors[0][1] <= 1.0


def test_default_configuration_is_unchanged(model, data):
    """Same pipeline, neighbours and score as the locked code (tests/data/default_regression.json)."""
    expected = json.loads((ROOT / "tests/data/default_regression.json").read_text())
    rec = recommend(model, data)
    assert rec.pipeline == expected["pipeline"] and rec.source == expected["source"]
    assert [d for d, _ in rec.neighbors] == expected["neighbors"]
    assert rec.score == pytest.approx(expected["score"], abs=1e-12)
    assert rec.search_evaluations == expected["search_evaluations"]


def test_split_record_refuses_a_different_saved_split(tmp_path):
    out = tmp_path / "r.jsonl"
    split = (np.arange(0, 6), np.arange(6, 8), np.arange(8, 10))
    h = run_rq1.split_record(out, 7, 0, split)
    assert run_rq1.split_record(out, 7, 0, split) == h                       # identical: accepted
    other = (np.arange(1, 7), np.array([0, 7]), np.arange(8, 10))
    with pytest.raises(run_rq1.SplitMismatch):
        run_rq1.split_record(out, 7, 0, other)


def test_save_predictions_writes_file_and_both_metrics(tmp_path):
    out = tmp_path / "rq2" / "v.jsonl"; out.parent.mkdir()
    rec = run_rq1.save_predictions(out, 5, 1, "beta", {"y_true": [0, 0, 1, 1], "y_pred": [0, 1, 1, 1]})
    assert rec["accuracy"] == pytest.approx(0.75)
    assert rec["f1_macro"] == pytest.approx((2 / 3 + 0.8) / 2)
    with gzip.open(out.parent / rec["predictions_file"], "rt") as fh:
        assert json.load(fh)["y_pred"] == [0, 1, 1, 1]


def test_fingerprint_without_git_hashes_the_sources(tmp_path, monkeypatch):
    (tmp_path / "beta").mkdir(); (tmp_path / "experiments").mkdir()
    (tmp_path / "beta" / "a.py").write_text("x = 1\n")
    monkeypatch.setattr(run_rq1, "ROOT", tmp_path)
    first = run_rq1.code_fingerprint()
    (tmp_path / "beta" / "a.py").write_text("x = 2\n")
    assert first.startswith("files:") and run_rq1.code_fingerprint() != first
