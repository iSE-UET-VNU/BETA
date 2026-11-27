import numpy as np
import pandas as pd
import pytest

from beta.data import split_train_val_test


def _imbalanced(n=100, n_minority=10):
    X = pd.DataFrame({"row": np.arange(n)})
    y = pd.Series([1] * n_minority + [0] * (n - n_minority))
    return X, y


@pytest.mark.parametrize("seed", range(20))
def test_split_keeps_class_proportions_in_every_part(seed):
    X, y = _imbalanced()
    _, y_tr, _, y_va, _, y_te = split_train_val_test(X, y, 0.2, 0.2, seed=seed)
    assert (int(y_tr.sum()), int(y_va.sum()), int(y_te.sum())) == (6, 2, 2)


def test_split_is_a_partition_with_paper_sizes():
    X, y = _imbalanced()
    X_tr, _, X_va, _, X_te, _ = split_train_val_test(X, y, 0.2, 0.2, seed=0)
    rows = np.concatenate([X_tr.row, X_va.row, X_te.row])
    assert (len(X_tr), len(X_va), len(X_te)) == (60, 20, 20)
    assert sorted(rows) == list(range(100))


def test_unstratified_split_matches_earlier_versions():
    X, y = _imbalanced()
    _, _, _, _, X_te, _ = split_train_val_test(X, y, 0.2, 0.2, seed=42, stratify=False)
    assert list(X_te.row) == list(np.random.RandomState(42).permutation(100)[:20])


def test_class_too_small_to_stratify_falls_back_with_a_warning():
    X = pd.DataFrame({"row": np.arange(50)})
    y = pd.Series([2] + [1] * 9 + [0] * 40)  # class 2 has a single row
    with pytest.warns(UserWarning, match="unstratified"):
        X_tr, _, X_va, _, X_te, _ = split_train_val_test(X, y, 0.2, 0.2, seed=0)
    assert (len(X_tr), len(X_va), len(X_te)) == (30, 10, 10)
