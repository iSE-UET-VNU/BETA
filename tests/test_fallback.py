"""Empty search: BETA falls back to the best transferred pipeline that runs on the training rows (it used to stop with
"RuntimeError: aco search produced no valid pipeline")."""
import pandas as pd
import pytest
from sklearn.datasets import make_classification

import beta
from beta import BETA, BETAConfig


@pytest.fixture(scope="module")
def model():
    return BETA.from_pretrained()


@pytest.fixture(scope="module")
def data():
    X, y = make_classification(n_samples=160, n_features=6, n_informative=4, random_state=0)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(6)]), pd.Series(y)


def recommend(model, data):
    cfg = BETAConfig()
    cfg.aco.n_ants, cfg.aco.n_iterations = 5, 3
    X, y = data
    return BETA(model.library, model.encoder, model.pipeline_configs, cfg).recommend(X=X, y=y, dataset_id="query")


def test_empty_search_falls_back_to_the_transferred_pipeline(model, data, monkeypatch):
    monkeypatch.setattr(beta.aco, "search", lambda *a, **k: [])
    expected = recommend(model, data)  # run once to learn the neighbours, then compare with the transferred pipeline
    best = BETA(model.library, model.encoder, model.pipeline_configs, BETAConfig())._best_transferred_pipeline(expected.neighbors)
    assert expected.source == "transfer_fallback"
    assert expected.pipeline == {k: v for k, v in best.items() if k != "name"}


def test_fallback_skips_a_transferred_pipeline_that_fails_on_the_training_rows(model, data, monkeypatch):
    monkeypatch.setattr(beta.aco, "search", lambda *a, **k: [])
    first = recommend(model, data).pipeline
    real = beta.runs_on_training_rows
    monkeypatch.setattr(beta, "runs_on_training_rows",
                        lambda cfg, *a: False if {k: v for k, v in cfg.items() if k != "name"} == first else real(cfg, *a))
    second = recommend(model, data)
    assert second.source == "transfer_fallback"
    assert second.pipeline != first
