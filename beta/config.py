"""Operator space, pipeline prototype, and config dataclasses."""
from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List

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


@dataclass
class SplitConfig:
    val_ratio: float = 0.2
    test_ratio: float = 0.2


@dataclass
class BETAConfig:
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    transfer: TransferConfig = field(default_factory=TransferConfig)
    aco: ACOConfig = field(default_factory=ACOConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    proxy_model: str = "logreg"
    downstream_evaluator: str = "proxy"  # "proxy" | "autogluon"
    autogluon_time_limit: int = 60  # seconds per candidate fit; ACO evaluates many candidates
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
        cfg.proxy_model = raw.get("proxy_model", cfg.proxy_model)
        cfg.downstream_evaluator = raw.get("downstream_evaluator", cfg.downstream_evaluator)
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
