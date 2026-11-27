"""Operator space, pipeline prototype, and config dataclasses."""
from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Transformation types and operators, in the order the pipeline executes them.
OPERATORS: "OrderedDict[str, List[str]]" = OrderedDict(
    [
        ("imputation", ["none", "mean", "median", "most_frequent", "constant", "knn"]),
        ("scaling", ["none", "standard", "minmax", "robust", "maxabs"]),
        ("encoding", ["none", "onehot"]),
        ("outlier_removal", ["none", "iqr", "zscore", "mad", "lof", "isolation_forest"]),
        ("feature_selection", ["none", "variance_threshold", "k_best", "mutual_info"]),
        ("dimensionality_reduction", ["none", "pca", "svd"]),
    ]
)

STEP_ORDER: List[str] = list(OPERATORS.keys())

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
DEFAULT_PERFORMANCE_MATRIX = ASSETS_DIR / "performance_matrix.csv"
DEFAULT_METAFEATURES = ASSETS_DIR / "dataset_metafeatures.csv"
DEFAULT_REFERENCE_PIPELINES = ASSETS_DIR / "reference_pipelines.json"
DEFAULT_SIMILARITY_ENCODER = ASSETS_DIR / "similarity_encoder.pt"


def pipeline_space_size() -> int:
    return math.prod(len(v) for v in OPERATORS.values())


@dataclass
class SimilarityConfig:
    hidden_dim: int = 64
    embed_dim: int = 64
    epochs: int = 100
    lr: float = 1e-3
    similarity_target: str = "row_zscore_cosine"
    metric_objective: str = "embedding_cosine"  # "embedding_cosine" | "projector_product"  


@dataclass
class TransferConfig:
    top_k: int = 1 
    top_h: int = 3 


@dataclass
class ACOConfig:
    n_ants: int = 20  
    n_iterations: int = 10
    alpha: float = 1.0
    beta: float = 2.0
    evaporation: float = 0.2  
    elite_size: int = 3  
    update_strategy: str = "linear"  # "linear" (paper Eq. 13) | "exponential" | "rank" | "uniform"
    time_budget: Optional[float] = None  # seconds for the whole search; None = no limit


@dataclass
class SplitConfig:
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    stratify: bool = True  # False reproduces the unstratified splits of earlier versions


ABLATION_CHOICES = {
    "similarity": ("learned", "metafeature_cosine"),       # Table 5
    "heuristic": ("transferred", "uniform"),               # Table 6
    "optimizer": ("aco", "random", "greedy", "sa", "ga", "mcts", "tpe"),  # Table 7
    "operator_space": ("full", "no_cross_feature"),        # Table 8
    "mode": ("full", "transfer_only", "search_only"),      # Table 9
}
CROSS_FEATURE_STEPS = ("feature_selection", "dimensionality_reduction")


@dataclass
class AblationConfig:
    """RQ2 variants; the defaults are the complete BETA. Each table changes one field."""
    similarity: str = "learned"        # metafeature_cosine: cosine on standardised meta-features, no encoder
    heuristic: str = "transferred"     # uniform: eta = 1 for every operator (Eq. 14 still compares with P_trans)
    optimizer: str = "aco"             # others search the same space with the same proxy and budget, without eta
    operator_space: str = "full"       # no_cross_feature: feature_selection and dimensionality_reduction fixed to none
    mode: str = "full"                 # transfer_only: P_trans, no search; search_only: uniform eta, P_search, no Eq. 14

    def validate(self) -> None:
        for name, allowed in ABLATION_CHOICES.items():
            value = getattr(self, name)
            if value not in allowed:
                raise ValueError(f"ablation.{name}={value!r}; expected one of {allowed}")


@dataclass
class BETAConfig:
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    transfer: TransferConfig = field(default_factory=TransferConfig)
    aco: ACOConfig = field(default_factory=ACOConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    store_predictions: bool = False  # keep the final model's test predictions (any metric can be computed later)
    proxy_model: str = "logreg"
    search_evaluator: str = "proxy"  # "proxy" | "autogluon"
    downstream_evaluator: str = "proxy"  # "proxy" | "autogluon"
    eval_metric: str = "accuracy"  # "accuracy" | "f1_macro"
    query_metafeatures: str = "non_test"  # "non_test" (train + validation rows only) | "lookup" (full-dataset table)
    autogluon_time_limit: int = 300  # seconds for downstream AutoGluon fit
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BETAConfig":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "BETAConfig":
        cfg = cls()
        cfg.similarity = SimilarityConfig(**{**asdict(cfg.similarity), **raw.get("similarity", {})})
        cfg.transfer = TransferConfig(**{**asdict(cfg.transfer), **raw.get("transfer", {})})
        cfg.aco = ACOConfig(**{**asdict(cfg.aco), **raw.get("aco", {})})
        cfg.split = SplitConfig(**{**asdict(cfg.split), **raw.get("split", {})})
        cfg.ablation = AblationConfig(**{**asdict(cfg.ablation), **raw.get("ablation", {})})
        cfg.store_predictions = raw.get("store_predictions", cfg.store_predictions)
        cfg.proxy_model = raw.get("proxy_model", cfg.proxy_model)
        cfg.search_evaluator = raw.get("search_evaluator", cfg.search_evaluator)
        cfg.downstream_evaluator = raw.get("downstream_evaluator", cfg.downstream_evaluator)
        cfg.eval_metric = raw.get("eval_metric", cfg.eval_metric)
        cfg.query_metafeatures = raw.get("query_metafeatures", cfg.query_metafeatures)
        cfg.autogluon_time_limit = raw.get("autogluon_time_limit", cfg.autogluon_time_limit)
        cfg.seed = raw.get("seed", cfg.seed)
        return cfg

    def override(self, dotted_key: str, value: str) -> None:
        section, _, name = dotted_key.partition(".")
        target = getattr(self, section, None) if name else None
        if target is None:
            setattr(self, dotted_key, _coerce(value))
            return
        current = getattr(target, name)
        setattr(target, name, _coerce(value, type(current)))


def _coerce(value: str, target_type: type | None = None) -> Any:
    if target_type is bool:
        return value.strip().lower() in {"1", "true", "yes"}
    if target_type in (int, float):
        return target_type(value)
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value
