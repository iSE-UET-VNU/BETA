"""Dataset loading (OpenML, CSV)"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


@dataclass
class Dataset:
    id: str
    X: pd.DataFrame
    y: pd.Series


def _clean_xy(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = X[col].astype(str)
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


def load_csv(path: str, target_column: str, dataset_id: Optional[str] = None) -> Dataset:
    df = pd.read_csv(path)
    if target_column not in df.columns:
        raise ValueError(f"target column {target_column!r} not in {path}")
    X, y = _clean_xy(df.drop(columns=[target_column]), df[target_column])
    return Dataset(id=dataset_id or path, X=X, y=y)


def split_train_val_test(
    X: pd.DataFrame, y: pd.Series, val_ratio: float = 0.2, test_ratio: float = 0.2, seed: int = 42
):
    n = len(y)
    n_val, n_test = int(n * val_ratio), int(n * test_ratio)
    idx = np.random.RandomState(seed).permutation(n)
    test_idx, val_idx, train_idx = idx[:n_test], idx[n_test : n_test + n_val], idx[n_test + n_val :]
    parts = []
    for split_idx in (train_idx, val_idx, test_idx):
        parts.append(X.iloc[split_idx].reset_index(drop=True))
        parts.append(y.iloc[split_idx].reset_index(drop=True))
    return tuple(parts)  # X_train, y_train, X_val, y_val, X_test, y_test
