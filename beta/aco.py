"""Phase 3 - Pipeline Refining: ACO over the transferred heuristic (Sec. 3.3, Eq. 12-13)."""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-8

# "linear" is Eq. 13 of the paper; the others are the RQ3 variants (Table 12).
UPDATE_STRATEGIES = ("linear", "exponential", "rank", "uniform")

Scored = Tuple[Dict[str, str], float]


def pheromone_rewards(elite_scores: Sequence[float], strategy: str, score_min: float, score_max: float) -> np.ndarray:
    """Deposit for each elite pipeline (best first).

    score_min / score_max range over all M candidates of the iteration, as in Eq. 13:
    Delta(P) = (Score(P) - Score_min) / (Score_max - Score_min + eps).
    """
    scores = np.asarray(elite_scores, dtype=float)
    if strategy == "uniform":
        return np.ones_like(scores)
    if strategy == "rank":
        k = len(scores)
        return np.array([(k - r) / k for r in range(k)])
    delta = (scores - score_min) / (score_max - score_min + EPS)
    if strategy == "linear":
        return delta
    if strategy == "exponential":
        return np.exp(delta) / np.e
    raise ValueError(f"unknown pheromone update strategy {strategy!r}; expected one of {UPDATE_STRATEGIES}")


def update_pheromone(
    pheromone: Dict[str, np.ndarray],
    candidates: Sequence[Scored],
    options: Mapping[str, List[str]],
    elite_size: int,
    evaporation: float,
    strategy: str,
) -> Dict[str, np.ndarray]:
    """Eq. 13: tau <- (1 - rho) tau + sum over E of 1[P_s = o] Delta(P).

    `candidates` are this iteration's scored pipelines; E is their top `elite_size`.
    """
    for step in pheromone:
        pheromone[step] *= (1 - evaporation)
    if not candidates:
        return pheromone
    ranked = sorted(candidates, key=lambda c: c[1], reverse=True)
    all_scores = [s for _, s in ranked]
    elite = ranked[: max(1, elite_size)]
    rewards = pheromone_rewards([s for _, s in elite], strategy, min(all_scores), max(all_scores))
    for (cfg, _), delta in zip(elite, rewards):
        for step in pheromone:
            pheromone[step][options[step].index(cfg[step])] += delta
    return pheromone


def sampling_probabilities(pheromone: np.ndarray, eta: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Pr(o|s) proportional to tau^alpha * eta^beta (Eq. 12)."""
    probs = (np.asarray(pheromone) ** alpha) * (np.asarray(eta) ** beta)
    total = probs.sum()
    return probs / total if total > 0 and np.isfinite(total) else np.ones_like(probs) / len(probs)


def _cfg_key(cfg: Dict[str, str]) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted(cfg.items()))


def search(
    options: Mapping[str, List[str]],
    eta: Dict[str, np.ndarray],
    evaluate_fn: Callable[[List[Dict[str, str]]], List[Scored]],
    n_ants: int = 20,
    n_iterations: int = 10,
    alpha: float = 1.0,
    beta: float = 2.0,
    evaporation: float = 0.2,
    elite_size: int = 3,
    update_strategy: str = "linear",
    seed: int = 42,
    history: Optional[list] = None,
    time_budget: Optional[float] = None,
    stats: Optional[dict] = None,
) -> List[Scored]:
    """Returns all evaluated (config, score) pairs, best first. evaluate_fn scores S(D,P).

    Pipelines are scored once; an ant that builds an already-scored pipeline reuses its score.
    If `history` is a list, one record per iteration (best score, pheromone) is appended to it.
    `time_budget` (seconds) stops the search once exceeded; `stats`, if a dict, receives the number
    of evaluations, the seconds spent, and whether the budget stopped the search.
    """
    started = time.monotonic()
    out_of_time = (lambda: time.monotonic() - started >= time_budget) if time_budget is not None else (lambda: False)
    hit_budget = False
    strategy = (update_strategy or "linear").lower()
    if strategy not in UPDATE_STRATEGIES:
        raise ValueError(f"unknown pheromone update strategy {update_strategy!r}; expected one of {UPDATE_STRATEGIES}")
    rng = np.random.RandomState(seed)
    step_order = list(options.keys())
    eta_safe = {step: np.clip(np.nan_to_num(np.asarray(eta[step], dtype=float), nan=EPS), EPS, None) for step in step_order}
    pheromone = {step: np.ones(len(options[step])) for step in step_order}
    cache: Dict[Tuple[Tuple[str, str], ...], Scored] = {}

    for iteration in range(n_iterations):
        if out_of_time():
            hit_budget = True
            break
        built: Dict[Tuple[Tuple[str, str], ...], Dict[str, str]] = {}
        for _ in range(n_ants):
            cfg = {}
            for step in step_order:
                probs = sampling_probabilities(pheromone[step], eta_safe[step], alpha, beta)
                cfg[step] = options[step][rng.choice(len(options[step]), p=probs)]
            built.setdefault(_cfg_key(cfg), cfg)

        for cfg in (cfg for key, cfg in built.items() if key not in cache):
            if out_of_time():
                hit_budget = True
                break
            for scored_cfg, score in evaluate_fn([cfg]):
                cache[_cfg_key(scored_cfg)] = (dict(scored_cfg), float(score))
        candidates = [cache[key] for key in built if key in cache]  # pipelines that failed to evaluate drop out
        update_pheromone(pheromone, candidates, options, elite_size, evaporation, strategy)

        if history is not None:
            history.append({
                "iteration": iteration,
                "best_score": max((s for _, s in cache.values()), default=None),
                "pheromone": {step: values.tolist() for step, values in pheromone.items()},
            })
        if hit_budget:
            break

    if stats is not None:
        stats.update(evaluations=len(cache), seconds=round(time.monotonic() - started, 1), hit_time_budget=hit_budget)
    return sorted(cache.values(), key=lambda x: x[1], reverse=True)
