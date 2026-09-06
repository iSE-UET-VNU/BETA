"""Phase 1 - Similarity Learning: the Siamese behavioral-similarity encoder."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


@dataclass
class SimilarityEncoder:
    embedder: Any
    input_dim: int
    hidden_dim: int
    embed_dim: int
    similarity_target: str
    projector: Optional[Any] = None
    metric_objective: str = "embedding_cosine"

    def embed(self, metafeatures_scaled: np.ndarray) -> np.ndarray:
        torch = _torch()
        with torch.no_grad():
            z = self.embedder(torch.tensor(metafeatures_scaled, dtype=torch.float32))
            z = z / (z.norm(dim=1, keepdim=True) + 1e-8)  # Eq. 6
        return z.numpy()

    def predict_similarity(self, reference_embeddings: np.ndarray, query_embedding: np.ndarray) -> np.ndarray:
        """Compute predicted behavioral similarities between reference embeddings and query embedding."""
        if self.metric_objective == "projector_product" and self.projector is not None:
            torch = _torch()
            with torch.no_grad():
                t_ref = torch.tensor(reference_embeddings, dtype=torch.float32)
                t_query = torch.tensor(query_embedding.reshape(1, -1), dtype=torch.float32)
                inter = t_ref * t_query  # element-wise product of normalized embeddings
                sims = self.projector(inter).squeeze(-1).numpy()
            return sims
        # Default: cosine similarity (dot product of l2-normalized embeddings, Eq. 7)
        return cosine_similarity(reference_embeddings, query_embedding.reshape(1, -1)).ravel()

    def save(self, path: str) -> None:
        torch = _torch()
        payload = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "embed_dim": self.embed_dim,
            "similarity_target": self.similarity_target,
            "metric_objective": self.metric_objective,
            "state_dict": self.embedder.state_dict(),
        }
        if self.projector is not None:
            payload["projector_state_dict"] = self.projector.state_dict()
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str) -> "SimilarityEncoder":
        torch = _torch()
        payload = torch.load(path, map_location="cpu")
        embedder = _build_embedder(payload["input_dim"], payload["hidden_dim"], payload["embed_dim"])
        embedder.load_state_dict(payload["state_dict"])
        embedder.eval()
        projector = None
        if "projector_state_dict" in payload:
            projector = _build_projector(payload["embed_dim"])
            projector.load_state_dict(payload["projector_state_dict"])
            projector.eval()
        return cls(
            embedder=embedder,
            input_dim=payload["input_dim"],
            hidden_dim=payload["hidden_dim"],
            embed_dim=payload["embed_dim"],
            similarity_target=payload["similarity_target"],
            projector=projector,
            metric_objective=payload.get("metric_objective", "embedding_cosine"),
        )


def _torch():
    import torch
    return torch


def _build_embedder(input_dim: int, hidden_dim: int, embed_dim: int):
    torch = _torch()
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, embed_dim), nn.ReLU(),
    )


def _build_projector(embed_dim: int):
    torch = _torch()
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(embed_dim, 1),
        nn.Tanh(),
    )


def _row_rank_normalize(profiles: np.ndarray) -> np.ndarray:
    n_rows, n_cols = profiles.shape
    if n_cols <= 1:
        return np.ones_like(profiles, dtype=float)
    out = np.zeros_like(profiles, dtype=float)
    for i in range(n_rows):
        ranks = pd.Series(profiles[i]).rank(method="average").to_numpy(dtype=float)
        out[i] = (ranks - 1.0) / (n_cols - 1)
    return out


def _row_zscore_normalize(profiles: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mean, std = profiles.mean(axis=1, keepdims=True), profiles.std(axis=1, keepdims=True)
    return (profiles - mean) / np.where(std > eps, std, 1.0)


def behavioral_similarity(performance_matrix_imputed: pd.DataFrame, similarity_target: str = "row_zscore_cosine") -> np.ndarray:
    """S(D_i, D_j) via the relative behavioral profile (Def. 1-3, Eq. 2-3)."""
    profiles = performance_matrix_imputed.T.values  # rows = datasets, cols = reference pipelines
    if similarity_target == "row_zscore_cosine":
        normalized = _row_zscore_normalize(profiles)
    elif similarity_target == "rank_cosine":
        normalized = _row_rank_normalize(profiles)
    else:
        raise ValueError(f"unknown similarity_target: {similarity_target}")
    return cosine_similarity(normalized)


def _pearson_loss(pred, target):
    """1 - Pearson correlation over the batch (Eq. 8): optimizes ranking agreement, not scale."""
    p, t = pred.reshape(-1) - pred.mean(), target.reshape(-1) - target.mean()
    return 1.0 - (p * t).sum() / ((p.norm() * t.norm()) + 1e-8)


def train_similarity_encoder(
    metafeatures_df: pd.DataFrame,
    performance_matrix_imputed: pd.DataFrame,
    hidden_dim: int = 64,
    embed_dim: int = 64,
    epochs: int = 100,
    lr: float = 1e-3,
    seed: int = 42,
    similarity_target: str = "row_zscore_cosine",
    metric_objective: str = "embedding_cosine",
) -> SimilarityEncoder:
    torch = _torch()
    import torch.optim as optim

    common = sorted(set(performance_matrix_imputed.columns) & set(metafeatures_df.index))
    if not common:
        raise ValueError("no common datasets between performance_matrix and metafeatures_df")
    meta = metafeatures_df.loc[common]
    perf = performance_matrix_imputed[common]

    mf_scaled = MinMaxScaler().fit_transform(SimpleImputer(strategy="mean").fit_transform(meta)).astype(np.float32)
    mf_scaled = np.nan_to_num(mf_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    target_sim = behavioral_similarity(perf, similarity_target)  # r_ij, Eq. 4

    torch.manual_seed(seed)
    embedder = _build_embedder(mf_scaled.shape[1], hidden_dim, embed_dim)
    projector = _build_projector(embed_dim) if metric_objective == "projector_product" else None
    params = list(embedder.parameters()) + (list(projector.parameters()) if projector is not None else [])
    optimizer = optim.Adam(params, lr=lr)

    n = mf_scaled.shape[0]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    X_i = torch.tensor(np.array([mf_scaled[i] for i, _ in pairs]), dtype=torch.float32)
    X_j = torch.tensor(np.array([mf_scaled[j] for _, j in pairs]), dtype=torch.float32)
    y_pairs = torch.tensor(np.array([target_sim[i, j] for i, j in pairs]), dtype=torch.float32).unsqueeze(1)

    for _ in range(epochs):
        emb_i, emb_j = embedder(X_i), embedder(X_j)
        emb_i = emb_i / (emb_i.norm(dim=1, keepdim=True) + 1e-8)
        emb_j = emb_j / (emb_j.norm(dim=1, keepdim=True) + 1e-8)
        if metric_objective == "projector_product":
            pred = projector(emb_i * emb_j)
        else:
            pred = (emb_i * emb_j).sum(dim=1, keepdim=True)  # Eq. 7
        loss = _pearson_loss(pred, y_pairs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    embedder.eval()
    if projector is not None:
        projector.eval()
    return SimilarityEncoder(
        embedder=embedder,
        input_dim=mf_scaled.shape[1],
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        similarity_target=similarity_target,
        projector=projector,
        metric_objective=metric_objective,
    )
