"""Executes a six-step pipeline config: fit on train, transform val/test."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .config import STEP_ORDER
from . import operators as ops


class Pipeline:
    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.num_imputer = self.cat_imputer = None
        self.scaler = None
        self.encoder = None
        self.selector = None
        self.reducer = None
        self.num_cols = self.cat_cols = None
        self.selected_cols = None
        self.fitted = False

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        X = X.copy()
        X.columns = X.columns.astype(str)
        self.num_cols = X.select_dtypes(include="number").columns.tolist()
        self.cat_cols = X.select_dtypes(exclude="number").columns.tolist()
        X_num = X[self.num_cols].copy() if self.num_cols else None
        X_cat = X[self.cat_cols].copy() if self.cat_cols else None

        for step in STEP_ORDER:
            if step == "imputation":
                X_num, X_cat = self._fit_imputation(X_num, X_cat)
            elif step == "scaling":
                X_num = self._fit_scaling(X_num)
            elif step == "encoding":
                X_cat = self._fit_encoding(X_cat)
            elif step == "outlier_removal":
                X_num, X_cat, y = self._fit_outlier_removal(X_num, X_cat, y)
            elif step == "feature_selection":
                X_num, X_cat = self._fit_feature_selection(X_num, X_cat, y)
            elif step == "dimensionality_reduction":
                X_num = self._fit_dimred(X_num)

        self.fitted = True
        return self._concat(X_num, X_cat), y

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("call fit_transform before transform")
        X = X.copy()
        X.columns = X.columns.astype(str)
        X_num = X[self.num_cols].copy() if self.num_cols else None
        X_cat = X[self.cat_cols].copy() if self.cat_cols else None

        for step in STEP_ORDER:
            if step == "imputation":
                X_num, X_cat = self._transform_imputation(X_num, X_cat)
            elif step == "scaling":
                X_num = self._transform_scaling(X_num)
            elif step == "encoding":
                X_cat = self._transform_encoding(X_cat)
            elif step == "feature_selection":
                X_num, X_cat = self._transform_feature_selection(X_num, X_cat)
            elif step == "dimensionality_reduction":
                X_num = self._transform_dimred(X_num)
            # outlier_removal only drops rows at fit time (train split)

        return self._concat(X_num, X_cat)

    @staticmethod
    def _concat(X_num: Optional[pd.DataFrame], X_cat: Optional[pd.DataFrame]) -> pd.DataFrame:
        parts = [p for p in (X_num, X_cat) if p is not None]
        return pd.concat(parts, axis=1).reset_index(drop=True) if parts else pd.DataFrame(index=[])

    # --- imputation ---
    def _fit_imputation(self, X_num, X_cat):
        method = self.config["imputation"]
        self.num_imputer = ops.build_imputer(method, is_categorical=False)
        self.cat_imputer = ops.build_imputer(method, is_categorical=True)
        if X_num is not None and self.num_imputer is not None:
            X_num = pd.DataFrame(self.num_imputer.fit_transform(X_num), index=X_num.index, columns=X_num.columns)
        if X_cat is not None and self.cat_imputer is not None:
            X_cat = pd.DataFrame(self.cat_imputer.fit_transform(X_cat), index=X_cat.index, columns=X_cat.columns)
        return X_num, X_cat

    def _transform_imputation(self, X_num, X_cat):
        if X_num is not None and self.num_imputer is not None:
            X_num = pd.DataFrame(self.num_imputer.transform(X_num), index=X_num.index, columns=X_num.columns)
        if X_cat is not None and self.cat_imputer is not None:
            X_cat = pd.DataFrame(self.cat_imputer.transform(X_cat), index=X_cat.index, columns=X_cat.columns)
        return X_num, X_cat

    # --- scaling ---
    def _fit_scaling(self, X):
        if X is None:
            return X
        self.scaler = ops.build_scaler(self.config["scaling"])
        return pd.DataFrame(self.scaler.fit_transform(X), index=X.index, columns=X.columns) if self.scaler else X

    def _transform_scaling(self, X):
        return pd.DataFrame(self.scaler.transform(X), index=X.index, columns=X.columns) if X is not None and self.scaler else X

    # --- encoding ---
    def _fit_encoding(self, X_cat):
        if X_cat is None:
            return X_cat
        self.encoder = ops.build_encoder(self.config["encoding"])
        if self.encoder is None:
            return X_cat
        arr = self.encoder.fit_transform(X_cat)
        return pd.DataFrame(arr, index=X_cat.index, columns=self.encoder.get_feature_names_out(X_cat.columns))

    def _transform_encoding(self, X_cat):
        if X_cat is None or self.encoder is None:
            return X_cat
        arr = self.encoder.transform(X_cat)
        return pd.DataFrame(arr, index=X_cat.index, columns=self.encoder.get_feature_names_out(X_cat.columns))

    # --- outlier removal (drops rows, train split only) ---
    def _fit_outlier_removal(self, X_num, X_cat, y):
        method = self.config["outlier_removal"]
        if X_num is None or method == "none":
            return X_num, X_cat, y
        mask = ops.outlier_keep_mask(X_num, method)
        if not bool(mask.any()) or bool(mask.all()):
            return X_num, X_cat, y
        X_num = X_num.loc[mask].reset_index(drop=True)
        if X_cat is not None:
            X_cat = X_cat.loc[mask].reset_index(drop=True)
        y = y.loc[mask.values].reset_index(drop=True)
        return X_num, X_cat, y

    # --- feature selection ---
    def _fit_feature_selection(self, X_num, X_cat, y):
        method = self.config["feature_selection"]
        is_cat_numeric = X_cat is not None and all(pd.api.types.is_numeric_dtype(X_cat[c]) for c in X_cat.columns)
        if method == "none" or (X_num is None and not is_cat_numeric):
            return X_num, X_cat

        X_all = pd.concat([p for p in (X_num, X_cat if is_cat_numeric else None) if p is not None], axis=1)
        self.selector = ops.build_selector(method, X_all, y)
        self.selected_cols = X_all.columns[self.selector.get_support()]
        X_selected = X_all[self.selected_cols]
        num_set, cat_set = set(X_num.columns) if X_num is not None else set(), set(X_cat.columns) if is_cat_numeric else set()
        X_num_out = X_selected[[c for c in self.selected_cols if c in num_set]] if num_set else None
        X_cat_out = X_selected[[c for c in self.selected_cols if c in cat_set]] if is_cat_numeric else X_cat
        return X_num_out, X_cat_out

    def _transform_feature_selection(self, X_num, X_cat):
        if self.selector is None:
            return X_num, X_cat
        is_cat_numeric = X_cat is not None and all(pd.api.types.is_numeric_dtype(X_cat[c]) for c in X_cat.columns)
        X_all = pd.concat([p for p in (X_num, X_cat if is_cat_numeric else None) if p is not None], axis=1)
        arr = self.selector.transform(X_all)
        X_selected = pd.DataFrame(arr, index=X_all.index, columns=self.selected_cols)
        num_set = set(X_num.columns) if X_num is not None else set()
        cat_set = set(X_cat.columns) if is_cat_numeric else set()
        X_num_out = X_selected[[c for c in self.selected_cols if c in num_set]] if num_set else None
        X_cat_out = X_selected[[c for c in self.selected_cols if c in cat_set]] if is_cat_numeric else X_cat
        return X_num_out, X_cat_out

    # --- dimensionality reduction ---
    def _fit_dimred(self, X):
        method = self.config["dimensionality_reduction"]
        self.reducer = ops.build_reducer(method, X) if X is not None else None
        if self.reducer is None:
            return X
        arr = self.reducer.fit_transform(X)
        return pd.DataFrame(arr, index=X.index, columns=[f"dr_{i}" for i in range(arr.shape[1])])

    def _transform_dimred(self, X):
        if X is None or self.reducer is None:
            return X
        arr = self.reducer.transform(X)
        return pd.DataFrame(arr, index=X.index, columns=[f"dr_{i}" for i in range(arr.shape[1])])
