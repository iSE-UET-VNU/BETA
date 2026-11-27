"""OpenML-style meta-features m(D): table lookup, or computed for unseen datasets."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


_UNCOMPUTED_COLUMNS = (
    "CfsSubsetEval_DecisionStumpAUC", "CfsSubsetEval_DecisionStumpErrRate",
    "CfsSubsetEval_DecisionStumpKappa", "CfsSubsetEval_NaiveBayesAUC",
    "CfsSubsetEval_NaiveBayesErrRate", "CfsSubsetEval_NaiveBayesKappa",
    "CfsSubsetEval_kNN1NAUC", "CfsSubsetEval_kNN1NErrRate", "CfsSubsetEval_kNN1NKappa",
    "J48.00001.AUC", "J48.00001.ErrRate", "J48.00001.Kappa",
    "J48.0001.AUC", "J48.0001.ErrRate", "J48.0001.Kappa",
    "J48.001.AUC", "J48.001.ErrRate", "J48.001.Kappa",
    "REPTreeDepth1AUC", "REPTreeDepth1ErrRate", "REPTreeDepth1Kappa",
    "REPTreeDepth2AUC", "REPTreeDepth2ErrRate", "REPTreeDepth2Kappa",
    "REPTreeDepth3AUC", "REPTreeDepth3ErrRate", "REPTreeDepth3Kappa",
    "RandomTreeDepth1AUC", "RandomTreeDepth1ErrRate", "RandomTreeDepth1Kappa",
    "RandomTreeDepth2AUC", "RandomTreeDepth2ErrRate", "RandomTreeDepth2Kappa",
    "RandomTreeDepth3AUC", "RandomTreeDepth3ErrRate", "RandomTreeDepth3Kappa",
    "AutoCorrelation",
)


def compute_metafeatures(X: pd.DataFrame, y: pd.Series, seed: int = 42) -> Dict[str, Any]:
    from scipy.stats import entropy
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline as SkPipeline
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
    from sklearn.tree import DecisionTreeClassifier

    X, y = pd.DataFrame(X).copy(), pd.Series(y).copy()
    meta: Dict[str, Any] = {}
    n_instances, n_predictors = X.shape
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    symbolic_cols = X.select_dtypes(exclude=[np.number]).columns

    # As in OpenML: the class is a nominal attribute and counts among the features, even when it
    # arrives label-encoded as integers; "binary" means a nominal attribute with two values.
    n_features = n_predictors + 1
    n_numeric = len(numeric_cols)
    n_symbolic = len(symbolic_cols) + 1

    meta["NumberOfInstances"] = n_instances
    meta["NumberOfFeatures"] = n_features
    meta["NumberOfNumericFeatures"] = n_numeric
    meta["NumberOfSymbolicFeatures"] = n_symbolic
    meta["NumberOfBinaryFeatures"] = int(
        sum(X[c].nunique(dropna=True) == 2 for c in symbolic_cols) + (1 if y.nunique(dropna=True) == 2 else 0)
    )
    meta["PercentageOfNumericFeatures"] = 100.0 * n_numeric / max(n_features, 1)
    meta["PercentageOfSymbolicFeatures"] = 100.0 * n_symbolic / max(n_features, 1)
    meta["PercentageOfBinaryFeatures"] = 100.0 * meta["NumberOfBinaryFeatures"] / max(n_features, 1)
    meta["Dimensionality"] = n_features / max(n_instances, 1)

    n_missing = int(X.isna().sum().sum())
    n_inst_missing = int(X.isna().any(axis=1).sum())
    meta["NumberOfMissingValues"] = n_missing
    meta["NumberOfInstancesWithMissingValues"] = n_inst_missing
    meta["PercentageOfMissingValues"] = 100.0 * n_missing / max(n_instances * n_predictors, 1)
    meta["PercentageOfInstancesWithMissingValues"] = 100.0 * n_inst_missing / max(n_instances, 1)

    y_enc = LabelEncoder().fit_transform(y)
    class_counts = np.bincount(y_enc)
    probs = class_counts / class_counts.sum()
    meta["NumberOfClasses"] = len(class_counts)
    meta["MajorityClassSize"] = int(class_counts.max())
    meta["MinorityClassSize"] = int(class_counts.min())
    meta["MajorityClassPercentage"] = 100.0 * float(probs.max())
    meta["MinorityClassPercentage"] = 100.0 * float(probs.min())
    meta["ClassEntropy"] = float(entropy(probs, base=2))  # OpenML reports entropy in bits

    if len(symbolic_cols) > 0:
        distinct = X[symbolic_cols].nunique(dropna=True)
        meta["MaxNominalAttDistinctValues"] = float(distinct.max())
        meta["MinNominalAttDistinctValues"] = float(distinct.min())
        meta["MeanNominalAttDistinctValues"] = float(distinct.mean())
        meta["StdvNominalAttDistinctValues"] = float(distinct.std())
    else:
        for k in ("Max", "Min", "Mean", "Stdv"):
            meta[f"{k}NominalAttDistinctValues"] = np.nan

    # Landmarkers as OpenML computes them with Weka: 10-fold CV, attributes normalised for kNN.
    n_splits = int(min(10, np.min(class_counts))) if len(class_counts) else 0
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        def make_preprocessor(normalise: bool) -> ColumnTransformer:
            numeric = (SkPipeline([("imp", SimpleImputer(strategy="median")), ("scale", MinMaxScaler())])
                       if normalise else SimpleImputer(strategy="median"))
            return ColumnTransformer(
                [
                    ("num", numeric, numeric_cols),
                    (
                        "cat",
                        SkPipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                    ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
                        symbolic_cols,
                    ),
                ],
                remainder="drop",
            )

        def eval_model(model, normalise: bool):
            aucs, accs, kappas = [], [], []
            for tr, te in cv.split(X, y_enc):
                pipe = SkPipeline([("prep", make_preprocessor(normalise)), ("mdl", model)])
                pipe.fit(X.iloc[tr], y_enc[tr])
                preds = pipe.predict(X.iloc[te])
                accs.append(accuracy_score(y_enc[te], preds))
                kappas.append(cohen_kappa_score(y_enc[te], preds))
                aucs.append(
                    roc_auc_score(y_enc[te], pipe.predict_proba(X.iloc[te])[:, 1])
                    if len(np.unique(y_enc)) == 2 else np.nan
                )
            with np.errstate(invalid="ignore"):
                auc = float(np.nanmean(aucs)) if not np.all(np.isnan(aucs)) else np.nan
            return auc, float(1.0 - np.nanmean(accs)), float(np.nanmean(kappas))

        for prefix, model, normalise in (
            ("DecisionStump", DecisionTreeClassifier(max_depth=1, random_state=seed), False),
            ("NaiveBayes", GaussianNB(), False),
            ("kNN1N", KNeighborsClassifier(n_neighbors=1), True),
        ):
            try:
                auc, err, kap = eval_model(model, normalise)
            except Exception:
                auc = err = kap = np.nan
            meta[f"{prefix}AUC"] = auc
            meta[f"{prefix}ErrRate"] = err
            meta[f"{prefix}Kappa"] = kap
    else:
        for prefix in ("DecisionStump", "NaiveBayes", "kNN1N"):
            for suffix in ("AUC", "ErrRate", "Kappa"):
                meta[f"{prefix}{suffix}"] = np.nan

    for col in _UNCOMPUTED_COLUMNS:
        meta.setdefault(col, np.nan)
    return meta


def lookup_or_compute(dataset_id: str, table: pd.DataFrame, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None) -> Dict[str, Any]:
    try:
        return table.loc[[dataset_id]].iloc[0].to_dict()
    except KeyError:
        if X is None or y is None:
            raise KeyError(f"{dataset_id!r} is not in the metafeature table and no X/y was given to compute it")
        return compute_metafeatures(X, y)


QUERY_METAFEATURE_MODES = ("non_test", "lookup")


def query_metafeatures(
    mode: str, dataset_id: str, table: pd.DataFrame,
    X_full: pd.DataFrame, y_full: pd.Series, X_non_test: pd.DataFrame, y_non_test: pd.Series, seed: int = 42,
) -> Dict[str, Any]:
    """m(D_new) for retrieval.

    non_test: computed from the training and validation rows only, so no test row or label reaches
              retrieval. Columns this module cannot compute are left missing and imputed downstream.
    lookup:   the precomputed OpenML row for the dataset (computed on the full dataset), else
              computed on the full dataset; the behaviour of earlier versions.
    """
    if mode == "non_test":
        return compute_metafeatures(X_non_test, y_non_test, seed=seed)
    if mode == "lookup":
        return lookup_or_compute(dataset_id, table, X_full, y_full)
    raise ValueError(f"unknown query_metafeatures mode {mode!r}; expected one of {QUERY_METAFEATURE_MODES}")


def load_table(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)
