"""Poxy and AutoGluon downstream evaluators"""
from __future__ import annotations

import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from .pipeline import Pipeline


class InfrastructureError(RuntimeError):
    """The machine failed, not the method (process suspended mid-fit, out of memory). Such a run is
    redone with identical settings instead of being recorded as the method's result."""


# AutoGluon stops at a wall-clock time limit. A fit that took far longer than its limit was suspended
# (laptop asleep, container paused) and stopped early on waking, so its score is not a real result.
FIT_OVERRUN_FACTOR, FIT_OVERRUN_SLACK = 2.0, 120.0


def check_fit_duration(seconds: float, time_limit: float) -> None:
    if seconds > FIT_OVERRUN_FACTOR * time_limit + FIT_OVERRUN_SLACK:
        raise InfrastructureError(
            f"AutoGluon fit took {seconds:.0f} s for a {time_limit:.0f} s limit: the process was suspended"
        )


def detect_problem_type(y: pd.Series) -> str:
    return "regression" if np.issubdtype(y.dtype, np.number) and y.nunique() > 50 else "classification"


def classification_problem_type(y) -> str:
    """AutoGluon problem type for a class label. Passed explicitly so AutoGluon never guesses
    regression from integer labels with many classes (e.g. abalone's 28 ring counts)."""
    return "binary" if pd.Series(y).nunique() == 2 else "multiclass"


def compute_metric(y_true, y_pred, metric: str = "accuracy") -> float:
    if metric == "f1_macro":
        return float(f1_score(y_true, y_pred, average="macro"))
    return float(accuracy_score(y_true, y_pred))


def proxy_score(config: Dict[str, str], X_train, y_train, X_val, y_val, eval_metric: str = "accuracy") -> Optional[float]:
    """Logistic-regression proxy: max accuracy/f1_macro over a small C x class_weight sweep."""
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
                score = compute_metric(y_val, clf.predict(X_val_p), eval_metric)
                best = max(best, score)
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


def autogluon_score(
    config: Dict[str, str],
    X_train, y_train, X_eval, y_eval,
    target_column: str = "target",
    time_limit: int = 60,
    eval_metric: str = "accuracy",
    predictions_out: Optional[dict] = None,
) -> Optional[float]:
    """Fit AutoGluon on the preprocessed training data and score X_eval. If `predictions_out` is a dict,
    it receives y_true and y_pred of the evaluated rows, so any metric can be computed later."""
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
    ag_metric = "f1_macro" if eval_metric == "f1_macro" else "accuracy"
    try:
        predictor = TabularPredictor(
            label=target_column,
            path=str(workdir),
            problem_type=classification_problem_type(y_train_p),
            eval_metric=ag_metric,
            verbosity=0,
            learner_kwargs={"random_state": 42},
        )
        started = time.time()
        predictor.fit(
            train_data=train_df, time_limit=time_limit, presets="best_quality", dynamic_stacking=False,
            feature_generator=IdentityFeatureGenerator(), raise_on_no_models_fitted=False,
        )
        check_fit_duration(time.time() - started, time_limit)
        if not predictor.model_names():
            return None
        preds = predictor.predict(eval_df)
        if predictions_out is not None:
            predictions_out["y_true"] = [v.item() if hasattr(v, "item") else v for v in y_eval.reset_index(drop=True)]
            predictions_out["y_pred"] = [v.item() if hasattr(v, "item") else v for v in preds.reset_index(drop=True)]
        return compute_metric(y_eval.reset_index(drop=True), preds.reset_index(drop=True), eval_metric)
    except InfrastructureError:
        raise
    except MemoryError as err:
        raise InfrastructureError("out of memory during an AutoGluon fit") from err
    except Exception:
        return None
    finally:
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)


def make_evaluator(
    evaluator: str,
    X_train, y_train, X_val, y_val,
    target_column: str = "target",
    autogluon_time_limit: int = 60,
    eval_metric: str = "accuracy",
) -> Callable:
    score_fn = proxy_score if evaluator == "proxy" else autogluon_score
    extra: Dict[str, Any] = {"eval_metric": eval_metric}
    if evaluator != "proxy":
        extra["time_limit"] = autogluon_time_limit

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
    eval_metric: str = "accuracy",
) -> Tuple[Dict[str, str], str]:
    """P* = argmax over {P_search, P_trans}, evaluated on the training portion."""
    score_fn = proxy_score if evaluator == "proxy" else autogluon_score
    extra: Dict[str, Any] = {"eval_metric": eval_metric}
    if evaluator != "proxy":
        extra["time_limit"] = autogluon_time_limit
    search_score = score_fn(search_config, X_train, y_train, X_val, y_val, **extra)
    transfer_score = score_fn(transfer_config, X_train, y_train, X_val, y_val, **extra)
    search_score = search_score if search_score is not None else -np.inf
    transfer_score = transfer_score if transfer_score is not None else -np.inf
    return (search_config, "search") if search_score >= transfer_score else (transfer_config, "transfer")


def test_score(
    config: Dict[str, str],
    evaluator: str,
    X_train, y_train, X_test, y_test,
    autogluon_time_limit: int = 60,
    eval_metric: str = "accuracy",
    predictions_out: Optional[dict] = None,
) -> float:
    """Held-out test performance of the final chosen pipeline (predictions kept if `predictions_out` is given;
    only the AutoGluon evaluator produces them)."""
    score_fn = proxy_score if evaluator == "proxy" else autogluon_score
    extra: Dict[str, Any] = {"eval_metric": eval_metric}
    if evaluator != "proxy":
        extra["time_limit"] = autogluon_time_limit
        if predictions_out is not None:
            extra["predictions_out"] = predictions_out
    score = score_fn(config, X_train, y_train, X_test, y_test, **extra)
    if score is None:
        raise RuntimeError("final pipeline failed to evaluate on the test split")
    return score
