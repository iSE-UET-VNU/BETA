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
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    from sklearn.tree import DecisionTreeClassifier

    X, y = pd.DataFrame(X).copy(), pd.Series(y).copy()
    meta: Dict[str, Any] = {}
    n_instances, n_predictors = X.shape
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    symbolic_cols = X.select_dtypes(exclude=[np.number]).columns

    target_is_numeric = pd.api.types.is_numeric_dtype(y)  
    n_features = n_predictors + 1
    n_numeric = len(numeric_cols) + (1 if target_is_numeric else 0)
    n_symbolic = len(symbolic_cols) + (0 if target_is_numeric else 1)

    meta["NumberOfInstances"] = n_instances
    meta["NumberOfFeatures"] = n_features
    meta["NumberOfNumericFeatures"] = n_numeric
    meta["NumberOfSymbolicFeatures"] = n_symbolic
    meta["NumberOfBinaryFeatures"] = int(
        sum(X[c].nunique(dropna=True) == 2 for c in X.columns) + (1 if y.nunique(dropna=True) == 2 else 0)
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

    n_splits = int(min(3, np.min(class_counts))) if len(class_counts) else 0
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        preprocessor = ColumnTransformer(
            [
                ("num", SimpleImputer(strategy="median"), numeric_cols),
                (
                    "cat",
                    SkPipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
                    symbolic_cols,
                ),
            ],
            remainder="drop",
        )

        def eval_model(model):
            aucs, accs, kappas = [], [], []
            for tr, te in cv.split(X, y_enc):
                pipe = SkPipeline([("prep", preprocessor), ("mdl", model)])
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

        for prefix, model in (
            ("DecisionStump", DecisionTreeClassifier(max_depth=1, random_state=seed)),
            ("NaiveBayes", GaussianNB()),
            ("kNN1N", KNeighborsClassifier(n_neighbors=1)),
        ):
            try:
                auc, err, kap = eval_model(model)
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


def load_table(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)
