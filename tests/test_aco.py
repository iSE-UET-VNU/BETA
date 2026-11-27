import numpy as np
import pytest

from beta import aco
from beta.config import OPERATORS

ONE_STEP = {"s": ["a", "b", "c"]}


def _constant_eval(batch):
    return [(cfg, 0.5) for cfg in batch]


def _flat_eta(options=OPERATORS):
    return {step: np.ones(len(ops)) for step, ops in options.items()}


def test_eq13_normalises_over_all_candidates_of_the_iteration():
    # candidates span 0.5..0.9; the elite is the top three
    np.testing.assert_allclose(aco.pheromone_rewards([0.9, 0.8, 0.7], "linear", 0.5, 0.9), [1.0, 0.75, 0.5], atol=1e-6)


def test_update_deposits_on_the_iterations_elite_only():
    pheromone = {"s": np.ones(3)}
    candidates = [({"s": "a"}, 0.9), ({"s": "b"}, 0.7), ({"s": "c"}, 0.5)]
    aco.update_pheromone(pheromone, candidates, ONE_STEP, elite_size=2, evaporation=0.2, strategy="linear")
    # a: 0.8 + 1.0, b: 0.8 + (0.7-0.5)/(0.9-0.5), c: evaporated only (not in the elite)
    np.testing.assert_allclose(pheromone["s"], [1.8, 1.3, 0.8], atol=1e-6)


def test_iteration_of_already_scored_pipelines_still_evaporates():
    history = []
    single = {"s": ["only"]}
    aco.search(single, _flat_eta(single), _constant_eval, n_ants=2, n_iterations=3,
               evaporation=0.2, update_strategy="linear", history=history)
    # one candidate per iteration: Delta = 0, so tau decays 1 -> 0.8 -> 0.64 -> 0.512
    np.testing.assert_allclose([h["pheromone"]["s"][0] for h in history], [0.8, 0.64, 0.512], atol=1e-6)


def test_linear_is_the_default_rule():
    from beta.config import ACOConfig
    assert ACOConfig().update_strategy == "linear"


@pytest.mark.parametrize("strategy", ["linear", "exponential", "rank", "uniform"])
def test_every_named_strategy_runs(strategy):
    results = aco.search(OPERATORS, _flat_eta(), _constant_eval, n_ants=3, n_iterations=2, update_strategy=strategy)
    assert results


def test_misspelled_strategy_is_rejected():
    with pytest.raises(ValueError, match="update strategy"):
        aco.search(OPERATORS, _flat_eta(), _constant_eval, n_ants=3, n_iterations=2, update_strategy="exponentail")


def test_time_budget_stops_the_search_and_says_so():
    import time

    def slow_eval(batch):
        time.sleep(0.05)
        return [(cfg, 0.5) for cfg in batch]

    stats = {}
    aco.search(OPERATORS, _flat_eta(), slow_eval, n_ants=20, n_iterations=10, time_budget=0.2, stats=stats)
    assert stats["hit_time_budget"] and 1 <= stats["evaluations"] < 20


def test_no_budget_runs_every_iteration():
    stats = {}
    aco.search(OPERATORS, _flat_eta(), _constant_eval, n_ants=3, n_iterations=2, stats=stats)
    assert not stats["hit_time_budget"] and stats["evaluations"] >= 1
