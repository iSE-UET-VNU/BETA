from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif, mutual_info_classif
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler


def build_imputer(method: str, is_categorical: bool) -> Optional[SimpleImputer]:
    if method == "none":
        return None
    if is_categorical:
        return SimpleImputer(strategy="most_frequent")
    if method == "knn":
        return KNNImputer(n_neighbors=5)
    return SimpleImputer(strategy=method)


def build_scaler(method: str) -> Optional[Any]:
    return {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
        "maxabs": MaxAbsScaler(),
    }.get(method)


def build_encoder(method: str) -> Optional[Any]:
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False) if method == "onehot" else None


def outlier_keep_mask(X_num: pd.DataFrame, method: str) -> pd.Series:
    if method == "iqr":
        mask = pd.Series(True, index=X_num.index)
        for col in X_num.columns:
            q1, q3 = X_num[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr > 0:
                mask &= (X_num[col] >= q1 - 1.5 * iqr) & (X_num[col] <= q3 + 1.5 * iqr)
        return mask
    if method == "zscore":
        return pd.Series((np.abs(zscore(X_num)) < 3).all(axis=1), index=X_num.index)
    if method == "mad":
        mask = pd.Series(True, index=X_num.index)
        for col in X_num.columns:
            median = X_num[col].median()
            mad = (X_num[col] - median).abs().median()
            if mad > 0:
                mask &= (0.6745 * (X_num[col] - median) / mad).abs() < 3.5
        return mask
    if method == "lof":
        return pd.Series(LocalOutlierFactor(n_neighbors=20).fit_predict(X_num) == 1, index=X_num.index)
    if method == "isolation_forest":
        return pd.Series(IsolationForest(contamination=0.05, random_state=42).fit_predict(X_num) == 1, index=X_num.index)
    return pd.Series(True, index=X_num.index)


def build_selector(method: str, X: pd.DataFrame, y: pd.Series) -> Optional[Any]:
    if method == "none":
        return None
    if method == "variance_threshold":
        selector = VarianceThreshold(threshold=0.01)
    elif method == "k_best":
        selector = SelectKBest(f_classif, k=min(20, X.shape[1]))
    else:
        selector = SelectKBest(lambda Xv, yv: mutual_info_classif(Xv, yv, discrete_features="auto"), k=min(20, X.shape[1]))
    selector.fit(X, y.values.ravel())
    return selector


def build_reducer(method: str, X: pd.DataFrame) -> Optional[Any]:
    if method == "none" or X.shape[1] <= 1 or len(X) < 2:
        return None
    n_components = min(10, X.shape[1], len(X) - 1)
    return PCA(n_components=n_components) if method == "pca" else TruncatedSVD(n_components=n_components)
