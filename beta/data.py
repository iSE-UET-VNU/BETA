"""Dataset loading (OpenML, CSV)"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


@dataclass
class Dataset:
    id: str
    X: pd.DataFrame
    y: pd.Series


def _clean_xy(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    # Categorical values become text; missing values stay missing (astype(str) alone would turn
    # them into the category "nan", depending on the pandas version).
    for col in X.select_dtypes(exclude="number").columns:
        X[col] = X[col].astype(object).where(X[col].isna(), X[col].astype(str))
    X = X.dropna(axis=1, how="all")
    keep = ~pd.isna(y)
    X, y = X[keep].reset_index(drop=True), y[keep].reset_index(drop=True)
    if y.dtype == "object" or str(y.dtype) == "category":
        y = pd.Series(LabelEncoder().fit_transform(y), name=y.name)
    return X, y


def load_openml(dataset_id: int) -> Dataset:
    from sklearn.datasets import fetch_openml

    fetched = fetch_openml(data_id=dataset_id, as_frame=True, parser="auto")
    X, y = _clean_xy(fetched.data.copy(), fetched.target.copy())
    return Dataset(id=f"D_{dataset_id}", X=X, y=y)


def load_csv(
    path: str, target_column: str, dataset_id: Optional[str] = None,
    only_empty_is_missing: bool = False, nominal_columns: Sequence[str] = (),
) -> Dataset:
    """With only_empty_is_missing=True, values such as "NULL", "NA" or "None" stay category values
    (usp05 has a class named NULL); only empty cells are missing. `nominal_columns` are read as
    categories even when their values look like numbers (a CSV does not keep OpenML's types)."""
    kwargs = {"keep_default_na": False, "na_values": [""]} if only_empty_is_missing else {}
    df = pd.read_csv(path, dtype={c: str for c in nominal_columns}, low_memory=False, **kwargs)
    if target_column not in df.columns:
        raise ValueError(f"target column {target_column!r} not in {path}")
    X, y = _clean_xy(df.drop(columns=[target_column]), df[target_column])
    return Dataset(id=dataset_id or path, X=X, y=y)


def _stratified_indices(y: pd.Series, n_val: int, n_test: int, seed: int):
    """Test, then val, carved off by class; raises ValueError when a class is too small to split."""
    from sklearn.model_selection import train_test_split

    labels = np.asarray(y)
    idx = np.arange(len(labels))
    rest_idx, test_idx = train_test_split(idx, test_size=n_test, stratify=labels, random_state=seed)
    train_idx, val_idx = train_test_split(rest_idx, test_size=n_val, stratify=labels[rest_idx], random_state=seed)
    return train_idx, val_idx, test_idx


def split_indices(y: pd.Series, val_ratio: float = 0.2, test_ratio: float = 0.2, seed: int = 42, stratify: bool = True):
    """(train_idx, val_idx, test_idx). Stratified by class unless `stratify=False` or a class is too
    small to stratify, in which case it is the seeded random permutation earlier versions used."""
    n = len(y)
    n_val, n_test = int(n * val_ratio), int(n * test_ratio)
    split = None
    if stratify and n_val > 0 and n_test > 0:
        try:
            split = _stratified_indices(y, n_val, n_test, seed)
        except ValueError as err:
            warnings.warn(f"stratified split not possible ({err}); using an unstratified split")
    if split is None:
        idx = np.random.RandomState(seed).permutation(n)
        split = idx[n_test + n_val :], idx[n_test : n_test + n_val], idx[:n_test]
    return split


def split_train_val_test(
    X: pd.DataFrame, y: pd.Series, val_ratio: float = 0.2, test_ratio: float = 0.2, seed: int = 42,
    stratify: bool = True,
):
    """X_train, y_train, X_val, y_val, X_test, y_test for the split given by `split_indices`."""
    train_idx, val_idx, test_idx = split_indices(y, val_ratio, test_ratio, seed, stratify)
    parts = []
    for split_idx in (train_idx, val_idx, test_idx):
        parts.append(X.iloc[split_idx].reset_index(drop=True))
        parts.append(y.iloc[split_idx].reset_index(drop=True))
    return tuple(parts)  # X_train, y_train, X_val, y_val, X_test, y_test
