"""Poxy and AutoGluon downstream evaluators"""
from __future__ import annotations

import re
import tempfile
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from .pipeline import Pipeline


def detect_problem_type(y: pd.Series) -> str:
    return "regression" if np.issubdtype(y.dtype, np.number) and y.nunique() > 50 else "classification"


def proxy_score(config: Dict[str, str], X_train, y_train, X_val, y_val) -> Optional[float]:
    """Logistic-regression proxy: max accuracy over a small C x class_weight sweep."""
    try:
        pipe = Pipeline(config)
        X_train_p, y_train_p = pipe.fit_transform(X_train, y_train)
        X_val_p = pipe.transform(X_val)
    except Exception:
        return None  
    if X_train_p.empty or X_val_p.empty or len(X_train_p) != len(y_train_p):
        return None
    best = -np.inf
    for C in (0.01, 0.1, 1.0):
        for class_weight in (None, "balanced"):
            try:
                clf = LogisticRegression(C=C, class_weight=class_weight, max_iter=3000, random_state=42)
                clf.fit(X_train_p, y_train_p)
                best = max(best, accuracy_score(y_val, clf.predict(X_val_p)))
            except Exception:
                continue
    return float(best) if np.isfinite(best) else None


def _sanitize_columns(*frames: pd.DataFrame) -> List[pd.DataFrame]:
    mapping, seen = {}, set()
    for col in frames[0].columns:
        clean = re.sub(r"[^a-zA-Z0-9_]", "_", str(col)) or "feat"
        candidate, i = clean, 1
        while candidate in seen:
            candidate, i = f"{clean}_{i}", i + 1
        seen.add(candidate)
        mapping[col] = candidate
    return [f.rename(columns=mapping) for f in frames]


def autogluon_score(config: Dict[str, str], X_train, y_train, X_eval, y_eval, target_column: str = "target", time_limit: int = 60) -> Optional[float]:
    from autogluon.tabular import TabularPredictor
    from autogluon.features.generators import IdentityFeatureGenerator

    try:
        pipe = Pipeline(config)
        X_train_p, y_train_p = pipe.fit_transform(X_train, y_train)
        X_eval_p = pipe.transform(X_eval)
    except Exception:
        return None
    if X_train_p.empty or X_eval_p.empty:
        return None

    train_df = X_train_p.copy()
    train_df[target_column] = y_train_p.values
    eval_df, train_df = _sanitize_columns(X_eval_p, train_df)

    workdir = Path(tempfile.gettempdir()) / f"beta_ag_{uuid.uuid4().hex}"
    try:
        predictor = TabularPredictor(label=target_column, path=str(workdir), eval_metric="accuracy", verbosity=0)
        predictor.fit(
            train_data=train_df, time_limit=time_limit, presets="best_quality", dynamic_stacking=False,
            feature_generator=IdentityFeatureGenerator(), raise_on_no_models_fitted=False,
        )
        if not predictor.model_names():
            return None
        preds = predictor.predict(eval_df)
        return float(accuracy_score(y_eval.reset_index(drop=True), preds.reset_index(drop=True)))
    except Exception:
        return None
    finally:
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)


def make_evaluator(evaluator: str, X_train, y_train, X_val, y_val, target_column: str = "target", autogluon_time_limit: int = 60) -> Callable:
    score_fn = proxy_score if evaluator == "proxy" else autogluon_score
    extra = {} if evaluator == "proxy" else {"time_limit": autogluon_time_limit}

    def evaluate_batch(configs: List[Dict[str, str]]) -> List[Tuple[Dict[str, str], float]]:
        results = []
        for cfg in configs:
            score = score_fn(cfg, X_train, y_train, X_val, y_val, **extra)
            if score is not None:
                results.append((cfg, score))
        return results

    return evaluate_batch


def select_final(
    search_config: Dict[str, str],
    transfer_config: Dict[str, str],
    evaluator: str,
    X_train, y_train, X_val, y_val,
    autogluon_time_limit: int = 60,
) -> Tuple[Dict[str, str], str]:
    """P* = argmax over {P_search, P_trans}, evaluated on the training portion."""
    score_fn = proxy_score if evaluator == "proxy" else autogluon_score
    extra = {} if evaluator == "proxy" else {"time_limit": autogluon_time_limit}
    search_score = score_fn(search_config, X_train, y_train, X_val, y_val, **extra)
    transfer_score = score_fn(transfer_config, X_train, y_train, X_val, y_val, **extra)
    search_score = search_score if search_score is not None else -np.inf
    transfer_score = transfer_score if transfer_score is not None else -np.inf
    return (search_config, "search") if search_score >= transfer_score else (transfer_config, "transfer")


def test_score(config: Dict[str, str], evaluator: str, X_train, y_train, X_test, y_test, autogluon_time_limit: int = 60) -> float:
    """Held-out test performance of the final chosen pipeline."""
    score_fn = proxy_score if evaluator == "proxy" else autogluon_score
    extra = {} if evaluator == "proxy" else {"time_limit": autogluon_time_limit}
    score = score_fn(config, X_train, y_train, X_test, y_test, **extra)
    if score is None:
        raise RuntimeError("final pipeline failed to evaluate on the test split")
    return score
