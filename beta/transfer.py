"""Phase 2 - Heuristic Transferring: dataset retrieval to per-step priors."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

EPS = 1e-8


def dataset_similarities(reference_embeddings: np.ndarray, dataset_ids: Sequence[str], query_embedding: np.ndarray) -> List[Tuple[str, float]]:
    """u_i = sim(z_new, z_i) for every reference dataset."""
    sims = cosine_similarity(reference_embeddings, query_embedding.reshape(1, -1)).ravel()
    ranked = sorted(zip(dataset_ids, sims.tolist()), key=lambda item: (-item[1], str(item[0])))
    return [(d, s) for d, s in ranked if np.isfinite(s)]


def top_k_neighbors(similarities: Sequence[Tuple[str, float]], top_k: int) -> List[Tuple[str, float]]:
    """N_K(D_new): the K most behaviorally similar reference datasets"""
    return list(similarities[: max(1, top_k)])


def top_h_pipelines(performance_matrix: pd.DataFrame, neighbors: Sequence[Tuple[str, float]], top_h: int) -> Dict[str, List[Tuple[str, float]]]:
    """P^H_i: the H highest-scoring reference pipelines for each retrieved neighbor"""
    result: Dict[str, List[Tuple[str, float]]] = {}
    for dataset_id, _sim in neighbors:
        if dataset_id not in performance_matrix.columns:
            continue
        scores = pd.to_numeric(performance_matrix[dataset_id], errors="coerce").dropna().sort_values(ascending=False)
        result[dataset_id] = list(scores.head(max(1, top_h)).items())
    return result


def aggregate_heuristics(
    top_h: Mapping[str, Sequence[Tuple[str, float]]],
    pipeline_configs: Sequence[Mapping[str, Any]],
    operators: Mapping[str, Sequence[str]],
) -> Dict[str, np.ndarray]:
    """eta_{s,o} = Avg(S(p) : p_s = o) over the transferred set P_trans"""
    cfg_by_name = {str(cfg["name"]): cfg for cfg in pipeline_configs}
    transferred = [(name, score) for records in top_h.values() for name, score in records]

    eta: Dict[str, np.ndarray] = {}
    for step, step_operators in operators.items():
        values = np.full(len(step_operators), np.nan)
        for idx, operator in enumerate(step_operators):
            scores = [
                score for name, score in transferred
                if cfg_by_name.get(name, {}).get(step) == operator and np.isfinite(score)
            ]
            if scores:
                values[idx] = float(np.mean(scores))
        observed = values[np.isfinite(values)]
        values[~np.isfinite(values)] = float(np.min(observed)) if observed.size else 1.0
        eta[step] = values
    return eta


def normalize_with_floor(raw_eta: Mapping[str, np.ndarray], floor: float = 0.05) -> Dict[str, np.ndarray]:
    floor = min(max(float(floor), 0.0), 0.95)
    normalized: Dict[str, np.ndarray] = {}
    for step, raw in raw_eta.items():
        arr = np.asarray(raw, dtype=float)
        lo, hi = float(arr.min()), float(arr.max())
        normalized[step] = np.ones_like(arr) if hi - lo <= EPS else np.clip(
            floor + (1.0 - floor) * (arr - lo) / (hi - lo + EPS), floor, 1.0
        )
    return normalized
