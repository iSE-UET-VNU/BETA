import numpy as np
import pandas as pd

from beta.data import load_csv, split_indices, split_train_val_test


def test_missing_categorical_values_stay_missing(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("colour,size,target\nred,1,a\n,2,b\nblue,,a\n")
    dataset = load_csv(str(path), "target", only_empty_is_missing=True)
    assert dataset.X["colour"].isna().tolist() == [False, True, False]  # not the category "nan"
    assert dataset.X["colour"].iloc[0] == "red"


def test_declared_nominal_columns_are_categories_even_when_they_look_numeric(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("code,x,target\n0,1.5,a\n2,2.5,b\n,3.5,a\n")
    dataset = load_csv(str(path), "target", only_empty_is_missing=True, nominal_columns=["code"])
    assert not pd.api.types.is_numeric_dtype(dataset.X["code"])
    assert dataset.X["code"].tolist()[:2] == ["0", "2"] and pd.isna(dataset.X["code"].iloc[2])
    assert pd.api.types.is_numeric_dtype(dataset.X["x"])


def test_split_indices_match_the_split_frames():
    X = pd.DataFrame({"row": np.arange(100)})
    y = pd.Series([1] * 10 + [0] * 90)
    train_idx, val_idx, test_idx = split_indices(y, 0.2, 0.2, seed=3)
    X_tr, _, X_va, _, X_te, _ = split_train_val_test(X, y, 0.2, 0.2, seed=3)
    assert list(X_tr.row) == list(train_idx) and list(X_va.row) == list(val_idx) and list(X_te.row) == list(test_idx)
