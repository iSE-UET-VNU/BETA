import numpy as np
import pandas as pd
import pytest

from beta import metafeatures


def _data(n=60, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({"num": rng.normal(size=n), "two_values": rng.randint(0, 2, size=n), "cat": rng.choice(["a", "b", "c"], n)})
    y = pd.Series(rng.randint(0, 2, size=n))  # label-encoded class, as the loaders produce
    return X, y


def test_label_encoded_class_counts_as_symbolic_like_openml():
    X, y = _data()
    mf = metafeatures.compute_metafeatures(X, y)
    assert (mf["NumberOfNumericFeatures"], mf["NumberOfSymbolicFeatures"]) == (2, 2)  # cat + class
    assert mf["NumberOfBinaryFeatures"] == 1  # only the class; a numeric 0/1 column is not "binary" in OpenML


def test_non_test_mode_ignores_the_full_dataset_table():
    X, y = _data()
    table = pd.DataFrame({"NumberOfInstances": [999]}, index=["1066"])
    mf = metafeatures.query_metafeatures("non_test", "1066", table, X, y, X.iloc[:48], y.iloc[:48])
    assert mf["NumberOfInstances"] == 48


def test_lookup_mode_reads_the_table():
    X, y = _data()
    table = pd.DataFrame({"NumberOfInstances": [999]}, index=["1066"])
    assert metafeatures.query_metafeatures("lookup", "1066", table, X, y, X, y)["NumberOfInstances"] == 999


def test_recommend_never_shows_test_rows_to_retrieval(monkeypatch):
    pytest.importorskip("torch")
    from beta import BETA, BETAConfig
    from beta.data import split_train_val_test

    X, y = _data(100, seed=1)
    X["row"] = np.arange(100)
    seen = {}
    original = metafeatures.compute_metafeatures

    def spy(X_seen, y_seen, seed=42):
        seen["rows"] = set(X_seen["row"])
        return original(X_seen, y_seen, seed=seed)

    monkeypatch.setattr(metafeatures, "compute_metafeatures", spy)
    cfg = BETAConfig()
    cfg.aco.n_ants, cfg.aco.n_iterations = 2, 1
    BETA.from_pretrained(cfg).recommend(X=X, y=y, dataset_id="D_1066")  # 1066 is in the table
    test_rows = set(split_train_val_test(X, y, 0.2, 0.2, cfg.seed)[4]["row"])
    assert seen["rows"] and not seen["rows"] & test_rows
    assert len(seen["rows"]) == 80
