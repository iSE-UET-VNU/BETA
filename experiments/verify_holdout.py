"""Check that the shipped similarity encoder was trained without the datasets in EVAL_DATASETS.

Retrains the encoder from the shipped matrix and meta-features exactly as `beta train-similarity`
does (every id in beta.knowledge.EVAL_DATASETS removed at load), then compares the weights with
assets/similarity_encoder.pt. Identical weights mean the shipped encoder saw only the held-out
reference library. The check is only as good as EVAL_DATASETS: a dataset listed under a wrong id
is not held out.

    python experiments/verify_holdout.py        # about 2 minutes on a laptop CPU
"""
from __future__ import annotations

import sys

import pandas as pd
import torch

from beta.config import DEFAULT_METAFEATURES, DEFAULT_PERFORMANCE_MATRIX, DEFAULT_SIMILARITY_ENCODER
from beta.knowledge import EVAL_ID_SET, ReferenceLibrary, normalize_id
from beta.similarity import train_similarity_encoder

TOLERANCE = 1e-5  # bitwise equality is expected on the same torch build; allow float noise across builds


def main() -> int:
    perf = pd.read_csv(DEFAULT_PERFORMANCE_MATRIX, index_col=0)
    meta = pd.read_csv(DEFAULT_METAFEATURES, index_col=0)
    dropped = sorted({normalize_id(c) for c in perf.columns} & EVAL_ID_SET, key=int)
    library = ReferenceLibrary.build(perf, meta)  # raises if any evaluation id survives the holdout
    print(f"evaluation datasets present in the shipped matrix and removed at load: {len(dropped)}")
    print(f"reference datasets the encoder is trained on: {len(library.metafeatures)}")

    retrained = train_similarity_encoder(library.metafeatures, library.performance_matrix_imputed, seed=42)
    payload = torch.load(DEFAULT_SIMILARITY_ENCODER, map_location="cpu")
    shipped = payload["state_dict"]
    trained_with = payload.get("torch_version", "not recorded")
    diff = max(float((shipped[k] - v).abs().max()) for k, v in retrained.embedder.state_dict().items())
    print(f"torch: encoder trained with {trained_with}, retrained here with {torch.__version__}")
    print(f"max |shipped - retrained| weight difference: {diff:.3g}")
    if diff > TOLERANCE:
        if trained_with != torch.__version__:
            print("INCONCLUSIVE: different torch builds give different weights; run this with the build above")
            return 2
        print("MISMATCH: the shipped encoder was not produced from the held-out library")
        return 1
    print("OK: the shipped encoder is reproduced from the library with every EVAL_DATASETS id held out")
    return 0


if __name__ == "__main__":
    sys.exit(main())
