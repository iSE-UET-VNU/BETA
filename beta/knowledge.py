"""Reference library D_ref: id normalization, eval-dataset holdout, alignment"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

# Paper Table 2's 30 evaluation datasets, name -> OpenML id. Held out from the reference library
# so pipeline-transfer knowledge is never learned from a dataset it will later be evaluated on.
EVAL_DATASETS: "dict[str, str]" = {
    "kc1-binary": "1066", "usp05": "1047", "sleuth-ex2016": "862", "calendarDOW": "40663",
    "mc2": "1054", "fri-c1": "876", "mfeat-morphological": "18", "robot-failures-lp5": "1520",
    "autoUniv-au4": "1548", "ipums-la-99": "378", "madelon": "1485", "mfeat-fourier": "14",
    "colic": "27", "abalone": "44956", "ada_prior": "1037", "avila": "42932",
    "connect-4": "40668", "eeg": "1471", "google": "100000", "house": "42165",
    "jungle_chess": "41001", "micro": "41671", "mozilla4": "1046", "obesity": "46597",
    "page-blocks": "30", "pbcseq": "802", "pol": "722", "run_or_walk": "40922",
    "uscensus": "1119", "wall-robot-nav": "1497",
}
EVAL_ID_SET = frozenset(EVAL_DATASETS.values())


def normalize_id(val: Any) -> str:
    """Map D_248 / 248 / 248.0 / dataset_248 to '248'."""
    if val is None:
        return ""
    if isinstance(val, float):
        return str(int(round(val))) if val == val and abs(val - round(val)) <= 1e-9 else str(val).strip()
    if isinstance(val, int):
        return str(val)
    s = str(val).strip()
    m = re.fullmatch(r"(?i)(?:d|dataset|openml)[_\-: ]*([0-9]+(?:\.[0-9]+)?)", s)
    if m:
        s = m.group(1)
    m = re.fullmatch(r"([0-9]+)\.[0-9]+", s)  # pandas' D_1037.1 dedup suffix
    return m.group(1) if m else s


def assert_disjoint(ids: Iterable[Any], context: str = "reference set") -> None:
    contaminated = sorted({normalize_id(x) for x in ids} & EVAL_ID_SET)
    if contaminated:
        raise AssertionError(f"held-out evaluation id(s) leaked into {context}: {contaminated}")


def holdout_reference(performance_matrix: pd.DataFrame, metafeatures_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    perf = performance_matrix.drop(
        columns=[c for c in performance_matrix.columns if normalize_id(c) in EVAL_ID_SET], errors="ignore"
    ).copy()
    meta = metafeatures_df.drop(
        index=[i for i in metafeatures_df.index if normalize_id(i) in EVAL_ID_SET], errors="ignore"
    ).copy()
    assert_disjoint(perf.columns, "performance matrix after holdout")
    assert_disjoint(meta.index, "metafeatures after holdout")
    return perf, meta


def _sanitize_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    return numeric.replace([np.inf, -np.inf], np.nan)


@dataclass
class ReferenceLibrary:
    performance_matrix: pd.DataFrame 
    metafeatures: pd.DataFrame  
    metafeatures_full: pd.DataFrame
    metafeatures_imputed: np.ndarray
    metafeatures_scaled: np.ndarray
    performance_matrix_imputed: pd.DataFrame
    imputer: SimpleImputer
    scaler: MinMaxScaler

    @classmethod
    def build(cls, performance_matrix: pd.DataFrame, metafeatures_df: pd.DataFrame) -> "ReferenceLibrary":
        full_meta = metafeatures_df.copy()
        full_meta.index = [normalize_id(i) for i in full_meta.index]
        perf, meta = holdout_reference(performance_matrix, metafeatures_df)

        perf_by_id = {normalize_id(c): c for c in perf.columns}
        meta_by_id = {normalize_id(i): i for i in meta.index}
        common = sorted(set(perf_by_id) & set(meta_by_id), key=lambda x: int(x) if x.isdigit() else x)
        if not common:
            raise ValueError("no dataset ids are common to the performance matrix and metafeature table")

        perf = _sanitize_numeric(perf.loc[:, [perf_by_id[c] for c in common]])
        meta = _sanitize_numeric(meta.loc[[meta_by_id[c] for c in common], :])
        perf.columns = common
        meta.index = common

        imputer, scaler = SimpleImputer(strategy="mean"), MinMaxScaler()
        meta_imputed = imputer.fit_transform(meta)
        meta_scaled = scaler.fit_transform(meta_imputed)

        perf_imputer = SimpleImputer(strategy="mean")
        perf_imputed = pd.DataFrame(perf_imputer.fit_transform(perf.T).T, index=perf.index, columns=perf.columns)

        return cls(perf, meta, full_meta, meta_imputed, meta_scaled, perf_imputed, imputer, scaler)
