"""BETA: Learning Dataset Similarity from Preprocessing Behavior for Preprocessing Pipeline Optimization."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from . import aco, evaluate, metafeatures, transfer
from .config import (
    BETAConfig, OPERATORS, DEFAULT_METAFEATURES, DEFAULT_PERFORMANCE_MATRIX,
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


class BETA:
    def __init__(self, library: ReferenceLibrary, encoder: SimilarityEncoder, pipeline_configs: List[Dict[str, Any]], config: Optional[BETAConfig] = None):
        self.library = library
        self.encoder = encoder
        self.pipeline_configs = pipeline_configs
        self.config = config or BETAConfig()

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

        query_meta = metafeatures.lookup_or_compute(normalize_id(dataset.id), self.library.metafeatures_full, dataset.X, dataset.y)
        query_scaled = self._scale_query(query_meta)

        neighbors, eta = self._transfer(dataset.id, query_scaled)

        X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(
            dataset.X, dataset.y, cfg.split.val_ratio, cfg.split.test_ratio, cfg.seed
        )
        evaluator = evaluate.make_evaluator(cfg.downstream_evaluator, X_train, y_train, X_val, y_val, autogluon_time_limit=cfg.autogluon_time_limit)
        search_results = aco.search(
            OPERATORS, eta, evaluator,
            n_ants=cfg.aco.n_ants, n_iterations=cfg.aco.n_iterations, alpha=cfg.aco.alpha,
            beta=cfg.aco.beta, evaporation=cfg.aco.evaporation, elite_size=cfg.aco.elite_size, seed=cfg.seed,
        )
        if not search_results:
            raise RuntimeError("ACO search produced no valid pipeline")
        search_config, _ = search_results[0]

        transfer_config = self._best_transferred_pipeline(neighbors)
        chosen, source = evaluate.select_final(
            search_config, transfer_config, cfg.downstream_evaluator, X_train, y_train, X_val, y_val, cfg.autogluon_time_limit
        )
        score = evaluate.test_score(chosen, cfg.downstream_evaluator, X_train, y_train, X_test, y_test, cfg.autogluon_time_limit)
        return Recommendation(pipeline={k: v for k, v in chosen.items() if k != "name"}, source=source, score=score, eta={s: v.tolist() for s, v in eta.items()}, neighbors=neighbors)

    def _scale_query(self, query_meta: Dict[str, Any]) -> np.ndarray:
        row = pd.DataFrame([query_meta]).reindex(columns=self.library.metafeatures.columns)
        imputed = self.library.imputer.transform(row)
        return self.library.scaler.transform(imputed)[0]

    def _transfer(self, dataset_id: str, query_scaled: np.ndarray):
        cfg = self.config.transfer
        reference_embeddings = self.encoder.embed(self.library.metafeatures_scaled)
        query_embedding = self.encoder.embed(query_scaled.reshape(1, -1))[0]
        sims = transfer.dataset_similarities(reference_embeddings, list(self.library.metafeatures.index), query_embedding)
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
