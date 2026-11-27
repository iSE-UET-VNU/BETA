"""RQ2 (Table 7): alternative optimizers for the Pipeline Refining phase.

Ported from the research code (SolutionRecommendation, src/automl_aco/search/optimizers.py) with the same
search rules. Each optimizer searches the same operator space with the same proxy evaluator as ACO and does
not use the transferred heuristic. The budget counts distinct pipelines evaluated (as in the research code);
ACO's budget is M x iterations ant proposals, of which repeats reuse a cached score, so these optimizers get
at least as many distinct evaluations as ACO: the comparison can only understate ACO.

`search` returns all evaluated (config, score) pairs, best first, like `aco.search`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from math import exp, log, sqrt
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

OPTIMIZERS = ("random", "greedy", "sa", "ga", "mcts", "tpe")

Scored = Tuple[Dict[str, str], float]
Key = Tuple[Tuple[str, str], ...]


def _key(cfg: Mapping[str, str], steps: Sequence[str]) -> Key:
    return tuple((s, cfg[s]) for s in steps)


def _random_cfg(options: Mapping[str, List[str]], rng: np.random.RandomState) -> Dict[str, str]:
    return {s: vals[int(rng.randint(0, len(vals)))] for s, vals in options.items()}


def _random_unseen(options, steps, rng, cache, max_tries: int = 500) -> Optional[Dict[str, str]]:
    for _ in range(max_tries):
        cfg = _random_cfg(options, rng)
        if _key(cfg, steps) not in cache:
            return cfg
    return None


def _neighbor(cfg: Mapping[str, str], options: Mapping[str, List[str]], rng: np.random.RandomState) -> Dict[str, str]:
    """Change the operator of one randomly chosen step."""
    steps = list(options.keys())
    step = steps[int(rng.randint(0, len(steps)))]
    others = [v for v in options[step] if v != cfg[step]]
    out = dict(cfg)
    if others:
        out[step] = others[int(rng.randint(0, len(others)))]
    return out


@dataclass
class _Node:
    depth: int
    choices: List[str]
    parent: Optional["_Node"] = None
    action: Optional[str] = None
    visits: int = 0
    value_sum: float = 0.0
    children: Dict[str, "_Node"] = field(default_factory=dict)
    untried: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.untried = list(self.choices)


def _ucb(node: _Node, parent_visits: int, c: float = 1.4) -> float:
    if node.visits == 0:
        return float("inf")
    return node.value_sum / node.visits + c * sqrt(max(log(max(parent_visits, 1)), 0.0) / node.visits)


def search(
    optimizer: str,
    options: Mapping[str, List[str]],
    evaluate_fn: Callable[[List[Dict[str, str]]], List[Scored]],
    budget: int = 200,
    seed: int = 42,
    time_budget: Optional[float] = None,
    stats: Optional[dict] = None,
) -> List[Scored]:
    optimizer = optimizer.lower().strip()
    if optimizer not in OPTIMIZERS:
        raise ValueError(f"unknown optimizer {optimizer!r}; expected one of {OPTIMIZERS}")
    started = time.monotonic()
    rng = np.random.RandomState(seed)
    steps = list(options.keys())
    space = int(np.prod([len(options[s]) for s in steps], dtype=np.int64))
    budget = max(0, min(int(budget), space))
    cache: Dict[Key, Scored] = {}
    attempts = {"n": 0, "calls": 0, "hit": False}
    max_attempts = max(50, 25 * budget)       # new pipelines tried (as in the research code)
    max_calls = max(1000, 100 * budget)       # safety cap: every proposal, cached or not, so no loop can spin forever

    def out_of_time() -> bool:
        if time_budget is not None and time.monotonic() - started >= time_budget:
            attempts["hit"] = True
            return True
        return False

    def budget_left() -> bool:
        return (len(cache) < budget and attempts["n"] < max_attempts and attempts["calls"] < max_calls
                and not out_of_time())

    def score(cfg: Dict[str, str]) -> Optional[float]:
        attempts["calls"] += 1
        k = _key(cfg, steps)
        if k in cache:
            return cache[k][1]
        attempts["n"] += 1
        for scored_cfg, s in evaluate_fn([dict(cfg)]):
            cache[k] = (dict(scored_cfg), float(s))
            return float(s)
        return None  # failed to evaluate: not cached, counts as an attempt

    def fill_random() -> None:
        while budget_left():
            score(_random_cfg(options, rng))

    if optimizer == "random":
        fill_random()

    elif optimizer == "sa":
        cur = _random_unseen(options, steps, rng, cache) or _random_cfg(options, rng)
        cur_score = score(cur)
        cur_score = -np.inf if cur_score is None else cur_score
        temp = 1.0
        while budget_left():
            nxt = _neighbor(cur, options, rng)
            s = score(nxt)
            if s is not None:
                delta = s - cur_score
                if delta >= 0 or rng.rand() < exp(delta / max(temp, 1e-6)):
                    cur, cur_score = nxt, s
            temp *= 0.97

    elif optimizer == "greedy":
        cur = _random_unseen(options, steps, rng, cache) or _random_cfg(options, rng)
        cur_score = score(cur)
        cur_score = -np.inf if cur_score is None else cur_score
        while budget_left():
            best, best_score, improved = cur, cur_score, False
            for step in steps:
                for val in options[step]:
                    if not budget_left():
                        break
                    if val == cur[step]:
                        continue
                    cand = dict(cur); cand[step] = val
                    s = score(cand)
                    if s is not None and s > best_score:
                        best, best_score, improved = cand, s, True
            if improved:
                cur, cur_score = best, best_score
            else:  # local optimum: restart from an unseen random pipeline (within the budget;
                # the research code evaluated this restart even when the budget was used up)
                if not budget_left():
                    break
                cur = _random_unseen(options, steps, rng, cache) or _random_cfg(options, rng)
                s = score(cur)
                cur_score = s if s is not None else cur_score

    elif optimizer == "ga":
        pop_size = min(20, max(4, budget // 5))
        population: List[Scored] = []
        while len(population) < pop_size and budget_left():
            cfg = _random_unseen(options, steps, rng, cache) or _random_cfg(options, rng)
            s = score(cfg)
            if s is not None:
                population.append((cfg, s))
        while budget_left() and population:
            population.sort(key=lambda x: x[1], reverse=True)
            survivors = population[: max(2, pop_size // 2)]
            children: List[Scored] = []
            while len(children) + len(survivors) < pop_size and budget_left():
                p1 = survivors[int(rng.randint(0, len(survivors)))][0]
                p2 = survivors[int(rng.randint(0, len(survivors)))][0]
                child = {}
                for step in steps:
                    child[step] = p1[step] if rng.rand() < 0.5 else p2[step]
                    if rng.rand() < 0.15:
                        child[step] = options[step][int(rng.randint(0, len(options[step])))]
                s = score(child)
                if s is not None:
                    children.append((child, s))
            population = survivors + children
        fill_random()

    elif optimizer == "tpe":
        n_init = min(max(10, budget // 10), budget)
        while len(cache) < n_init and budget_left():
            cfg = _random_unseen(options, steps, rng, cache)
            if cfg is None:
                break
            score(cfg)
        gamma, n_candidates, prior = 0.25, 48, 1.0
        while budget_left():
            ranked = sorted(cache.items(), key=lambda kv: kv[1][1], reverse=True)
            if len(ranked) < 2:
                cfg = _random_unseen(options, steps, rng, cache)
                if cfg is None:
                    break
                score(cfg)
                continue
            n_good = max(1, int(np.ceil(gamma * len(ranked))))
            good = [k for k, _ in ranked[:n_good]]
            bad = [k for k, _ in ranked[n_good:]] or good
            good_c = {s: {v: prior for v in options[s]} for s in steps}
            bad_c = {s: {v: prior for v in options[s]} for s in steps}
            for keys, counts in ((good, good_c), (bad, bad_c)):
                for k in keys:
                    for s, v in k:
                        counts[s][v] += 1.0

            def model_score(cfg):
                total = 0.0
                for s in steps:
                    pg = good_c[s][cfg[s]] / sum(good_c[s].values())
                    pb = bad_c[s][cfg[s]] / sum(bad_c[s].values())
                    total += log(max(pg, 1e-12)) - log(max(pb, 1e-12))
                return total

            best_cfg, best_ms = None, -np.inf
            for _ in range(n_candidates):
                cand = {}
                for s in steps:
                    vals = options[s]
                    probs = np.array([good_c[s][v] for v in vals], dtype=float)
                    cand[s] = vals[int(rng.choice(len(vals), p=probs / probs.sum()))]
                if _key(cand, steps) in cache:
                    continue
                ms = model_score(cand)
                if ms > best_ms:
                    best_cfg, best_ms = cand, ms
            if best_cfg is None:
                best_cfg = _random_unseen(options, steps, rng, cache)
                if best_cfg is None:
                    break
            score(best_cfg)

    elif optimizer == "mcts":
        root = _Node(depth=0, choices=options[steps[0]])

        def node_cfg(node: _Node) -> Dict[str, str]:
            actions: List[Optional[str]] = [None] * len(steps)
            cur = node
            while cur is not None and cur.action is not None and cur.depth > 0:
                actions[cur.depth - 1] = cur.action
                cur = cur.parent
            return {s: (actions[i] if actions[i] is not None else options[s][int(rng.randint(0, len(options[s])))])
                    for i, s in enumerate(steps)}

        while budget_left():
            node = root
            while not node.untried and node.children:
                parent = node
                node = max(parent.children.values(), key=lambda ch: _ucb(ch, parent.visits))
            if node.untried and node.depth < len(steps):
                action = node.untried.pop(int(rng.randint(0, len(node.untried))))
                depth = node.depth + 1
                child = _Node(depth=depth, choices=options[steps[depth]] if depth < len(steps) else [],
                              parent=node, action=action)
                node.children[action] = child
                node = child
            s = score(node_cfg(node))
            reward = s if s is not None else 0.0
            cur = node
            while cur is not None:
                cur.visits += 1
                cur.value_sum += reward
                cur = cur.parent

    if stats is not None:
        stats.update(evaluations=len(cache), seconds=round(time.monotonic() - started, 1), hit_time_budget=attempts["hit"])
    return sorted(cache.values(), key=lambda x: x[1], reverse=True)
