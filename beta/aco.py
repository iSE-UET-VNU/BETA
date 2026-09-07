"""Phase 3 - Pipeline Refining: ACO over the transferred heuristic."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Tuple

import numpy as np

EPS = 1e-8


def sampling_probabilities(pheromone: np.ndarray, eta: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Pr(o|s) proportional to tau^alpha * eta^beta ."""
    probs = (np.asarray(pheromone) ** alpha) * (np.asarray(eta) ** beta)
    total = probs.sum()
    return probs / total if total > 0 and np.isfinite(total) else np.ones_like(probs) / len(probs)


def _cfg_key(cfg: Dict[str, str]) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted(cfg.items()))


def search(
    options: Mapping[str, List[str]],
    eta: Dict[str, np.ndarray],
    evaluate_fn: Callable[[List[Dict[str, str]]], List[Tuple[Dict[str, str], float]]],
    n_ants: int = 20,
    n_iterations: int = 10,
    alpha: float = 1.0,
    beta: float = 2.0,
    evaporation: float = 0.2,
    elite_size: int = 3,
    update_strategy: str = "exponential",
    seed: int = 42,
) -> List[Tuple[Dict[str, str], float]]:
    """Returns all evaluated (config, score) pairs, best first. evaluate_fn scores S(D,P)."""
    rng = np.random.RandomState(seed)
    step_order = list(options.keys())
    eta_safe = {step: np.clip(np.nan_to_num(np.asarray(eta[step], dtype=float), nan=EPS), EPS, None) for step in step_order}
    pheromone = {step: np.ones(len(options[step])) for step in step_order}
    cache: Dict[Tuple[Tuple[str, str], ...], Tuple[Dict[str, str], float]] = {}

    for _ in range(n_iterations):
        batch = []
        for _ant in range(n_ants):
            cfg = {}
            for step in step_order:
                probs = sampling_probabilities(pheromone[step], eta_safe[step], alpha, beta)
                idx = rng.choice(len(options[step]), p=probs)
                cfg[step] = options[step][idx]
            if _cfg_key(cfg) not in cache:
                batch.append(cfg)
        if not batch:
            continue

        for cfg, score in evaluate_fn(batch):
            cache[_cfg_key(cfg)] = (dict(cfg), float(score))

        for step in pheromone:
            pheromone[step] *= (1 - evaporation)

        ranked = sorted(cache.values(), key=lambda x: x[1], reverse=True)
        elite = ranked[: max(1, elite_size)]
        scores = np.array([s for _, s in elite])

        # Pheromone update strategy: exponential (default), rank, uniform
        strategy = (update_strategy or "exponential").lower()
        if strategy == "uniform":
            reward = np.ones_like(scores)
        elif strategy == "rank":
            k = len(scores)
            reward = np.array([(k - r) / k for r in range(k)])
        elif strategy == "exponential":
            if len(scores) > 1 and (scores.max() - scores.min()) > EPS:
                norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + EPS)
                reward = np.exp(norm_scores) / np.e
            else:
                reward = np.ones_like(scores)
        else:  # fallback linear Eq. 13
            reward = (scores - scores.min()) / (scores.max() - scores.min() + EPS) if len(scores) > 1 else np.ones_like(scores)

        for (cfg, _score), delta in zip(elite, reward):
            for step in step_order:
                pheromone[step][options[step].index(cfg[step])] += delta

    return sorted(cache.values(), key=lambda x: x[1], reverse=True)
