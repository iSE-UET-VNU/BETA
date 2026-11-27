"""BETA: Learning Dataset Similarity from Preprocessing Behavior for Preprocessing Pipeline Optimization."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from . import aco, evaluate, metafeatures, optimizers, transfer
from .config import (
    BETAConfig, CROSS_FEATURE_STEPS, OPERATORS, DEFAULT_METAFEATURES, DEFAULT_PERFORMANCE_MATRIX,
    DEFAULT_REFERENCE_PIPELINES, DEFAULT_SIMILARITY_ENCODER,
)
from .data import Dataset, load_openml, split_train_val_test
from .knowledge import ReferenceLibrary, normalize_id
from .similarity import SimilarityEncoder, behavioral_similarity

__all__ = ["BETA", "BETAConfig", "Recommendation"]


@dataclass
class Recommendation:
    pipeline: Dict[str, str]
    source: str  # "search" | "transfer"
    score: float
    eta: Dict[str, List[float]]
    neighbors: List[tuple]
    search_evaluations: int = 0
    search_seconds: float = 0.0
    search_hit_time_budget: bool = False
    predictions: Optional[Dict[str, list]] = None  # y_true / y_pred of the test split, if requested


def search_space(operator_space: str) -> "Dict[str, List[str]]":
    """Operator options per step; Table 8's variant fixes the cross-feature steps to 'none'."""
    if operator_space == "no_cross_feature":
        return {s: (["none"] if s in CROSS_FEATURE_STEPS else list(v)) for s, v in OPERATORS.items()}
    return {s: list(v) for s, v in OPERATORS.items()}


def restrict(config: Dict[str, Any], options: Dict[str, List[str]]) -> Dict[str, Any]:
    """Map a pipeline into the search space (operators outside it become 'none')."""
    return {k: (v if k not in options or v in options[k] else "none") for k, v in config.items()}


class BETA:
    def __init__(self, library: ReferenceLibrary, encoder: SimilarityEncoder, pipeline_configs: List[Dict[str, Any]], config: Optional[BETAConfig] = None):
        self.library = library
        self.encoder = encoder
        self.pipeline_configs = pipeline_configs
        self.config = config or BETAConfig()
        if getattr(self.config, "similarity", None) and getattr(self.config.similarity, "metric_objective", None):
            self.encoder.metric_objective = self.config.similarity.metric_objective

    @classmethod
    def from_pretrained(cls, config: Optional[BETAConfig] = None) -> "BETA":
        perf = pd.read_csv(DEFAULT_PERFORMANCE_MATRIX, index_col=0)
        meta = pd.read_csv(DEFAULT_METAFEATURES, index_col=0)
        pipeline_configs = json.loads(DEFAULT_REFERENCE_PIPELINES.read_text())
        assert sorted(perf.index) == sorted(c["name"] for c in pipeline_configs), "performance matrix / reference pipeline mismatch"
        library = ReferenceLibrary.build(perf, meta)
        encoder = SimilarityEncoder.load(str(DEFAULT_SIMILARITY_ENCODER))
        return cls(library, encoder, pipeline_configs, config)

    def recommend(self, openml_id: Optional[int] = None, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None, dataset_id: Optional[str] = None) -> Recommendation:
        cfg = self.config
        if openml_id is not None:
            dataset = load_openml(openml_id)
        elif X is not None and y is not None:
            dataset = Dataset(id=dataset_id or "query", X=X, y=y)
        else:
            raise ValueError("pass either openml_id or (X, y)")

        X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(
            dataset.X, dataset.y, cfg.split.val_ratio, cfg.split.test_ratio, cfg.seed, stratify=cfg.split.stratify,
        )
        query_meta = metafeatures.query_metafeatures(
            cfg.query_metafeatures, normalize_id(dataset.id), self.library.metafeatures_full,
            dataset.X, dataset.y, pd.concat([X_train, X_val]), pd.concat([y_train, y_val]), seed=cfg.seed,
        )
        query_scaled = self._scale_query(query_meta)

        abl = cfg.ablation
        abl.validate()
        options = search_space(abl.operator_space)
        neighbors, eta = self._transfer(dataset.id, query_scaled)
        eta = {s: np.asarray(eta[s])[[OPERATORS[s].index(o) for o in options[s]]] for s in options}
        if abl.heuristic == "uniform" or abl.mode == "search_only":
            eta = {s: np.ones(len(v)) for s, v in options.items()}
        transfer_config = restrict(self._best_transferred_pipeline(neighbors), options)

        search_stats: Dict[str, Any] = {"evaluations": 0, "seconds": 0.0, "hit_time_budget": False}
        if abl.mode == "transfer_only":
            chosen, source = transfer_config, "transfer"
        else:
            evaluator = evaluate.make_evaluator(
                cfg.search_evaluator, X_train, y_train, X_val, y_val,
                autogluon_time_limit=cfg.autogluon_time_limit,
                eval_metric=cfg.eval_metric,
            )
            if abl.optimizer == "aco":
                search_results = aco.search(
                    options, eta, evaluator,
                    n_ants=cfg.aco.n_ants, n_iterations=cfg.aco.n_iterations, alpha=cfg.aco.alpha,
                    beta=cfg.aco.beta, evaporation=cfg.aco.evaporation, elite_size=cfg.aco.elite_size,
                    update_strategy=cfg.aco.update_strategy, seed=cfg.seed,
                    time_budget=cfg.aco.time_budget, stats=search_stats,
                )
            else:
                search_results = optimizers.search(
                    abl.optimizer, options, evaluator, budget=cfg.aco.n_ants * cfg.aco.n_iterations,
                    seed=cfg.seed, time_budget=cfg.aco.time_budget, stats=search_stats,
                )
            if not search_results:
                raise RuntimeError(f"{abl.optimizer} search produced no valid pipeline")
            search_config, _ = search_results[0]
            if abl.mode == "search_only":
                chosen, source = search_config, "search"
            else:
                chosen, source = evaluate.select_final(
                    search_config, transfer_config, cfg.downstream_evaluator, X_train, y_train, X_val, y_val,
                    cfg.autogluon_time_limit, eval_metric=cfg.eval_metric,
                )
        predictions: Optional[Dict[str, list]] = {} if cfg.store_predictions else None
        score = evaluate.test_score(
            chosen, cfg.downstream_evaluator, X_train, y_train, X_test, y_test,
            cfg.autogluon_time_limit, eval_metric=cfg.eval_metric, predictions_out=predictions,
        )
        return Recommendation(
            pipeline={k: v for k, v in chosen.items() if k != "name"}, source=source, score=score,
            eta={s: np.asarray(v).tolist() for s, v in eta.items()}, neighbors=neighbors,
            search_evaluations=search_stats["evaluations"], search_seconds=search_stats["seconds"],
            search_hit_time_budget=search_stats["hit_time_budget"], predictions=predictions or None,
        )

    def _scale_query(self, query_meta: Dict[str, Any]) -> np.ndarray:
        row = pd.DataFrame([query_meta]).reindex(columns=self.library.metafeatures.columns)
        imputed = self.library.imputer.transform(row)
        return self.library.scaler.transform(imputed)[0]

    def _transfer(self, dataset_id: str, query_scaled: np.ndarray):
        cfg = self.config.transfer
        if self.config.ablation.similarity == "metafeature_cosine":
            # Table 5: cosine on the standardised meta-features, no learned representation
            sims = transfer.dataset_similarities(
                np.asarray(self.library.metafeatures_scaled), list(self.library.metafeatures.index), query_scaled,
                encoder=None,
            )
        else:
            reference_embeddings = self.encoder.embed(self.library.metafeatures_scaled)
            query_embedding = self.encoder.embed(query_scaled.reshape(1, -1))[0]
            sims = transfer.dataset_similarities(
                reference_embeddings,
                list(self.library.metafeatures.index),
                query_embedding,
                encoder=self.encoder,
            )
        sims = [(d, s) for d, s in sims if normalize_id(d) != normalize_id(dataset_id)]
        neighbors = transfer.top_k_neighbors(sims, cfg.top_k)
        # Eq. 11 averages over the neighbor's known S(D,P); the imputed matrix fills the (real)
        # gaps in our performance matrix so a sparsely-measured K=1 neighbor doesn't zero out eta.
        top_h = transfer.top_h_pipelines(self.library.performance_matrix_imputed, neighbors, cfg.top_h)
        raw_eta = transfer.aggregate_heuristics(top_h, self.pipeline_configs, OPERATORS)
        return neighbors, transfer.normalize_with_floor(raw_eta)

    def _best_transferred_pipeline(self, neighbors) -> Dict[str, Any]:
        # Eq. 14's P_trans is one pipeline taken wholesale: prefer an actually-measured score over
        # an imputed one, since an imputed value here would fabricate the winning pipeline itself.
        best_name, best_score = None, -np.inf
        cfg_by_name = {c["name"]: c for c in self.pipeline_configs}
        for dataset_id, _sim in neighbors:
            scores = pd.to_numeric(self.library.performance_matrix[dataset_id], errors="coerce").dropna()
            if scores.empty:
                scores = self.library.performance_matrix_imputed[dataset_id]
            name, score = scores.idxmax(), scores.max()
            if score > best_score:
                best_name, best_score = name, score
        return dict(cfg_by_name[best_name])
