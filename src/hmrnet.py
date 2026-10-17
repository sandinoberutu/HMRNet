from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from scipy import sparse
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .preprocessing import set_global_seed


@dataclass
class HMRNetConfig:
    n_topics: int = 20
    hidden_dim: int = 256
    latent_dim: Optional[int] = None
    attention_heads: int = 4
    dropout: float = 0.20
    learning_rate: float = 2e-3
    weight_decay: float = 1e-5
    batch_size: int = 128
    epochs: int = 60
    patience: int = 8
    kl_weight: float = 0.10
    alignment_weight: float = 0.10
    diversity_weight: float = 0.05
    stability_weight: float = 0.05
    use_lexical: bool = True
    use_semantic: bool = True
    use_document: bool = True
    use_cra: bool = True
    fusion_mode: str = "adaptive"  # adaptive | concat | average
    use_alignment_loss: bool = True
    use_diversity_loss: bool = True
    use_stability_loss: bool = True
    branch_noise_std: float = 0.03
    gradient_clip: float = 5.0
    random_state: int = 42
    device: Optional[str] = None


class CrossRepresentationAttention(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(attended))
        return self.norm2(x + self.dropout(self.ffn(x)))


class AdaptiveFusion(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, branches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.scorer(branches).squeeze(-1), dim=1)
        fused = torch.sum(branches * weights.unsqueeze(-1), dim=1)
        return fused, weights


class HMRNetCore(nn.Module):
    def __init__(
        self,
        bow_dim: int,
        semantic_dim: int,
        document_dim: int,
        config: HMRNetConfig,
    ) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_dim
        latent = config.latent_dim or config.n_topics

        self.lexical_encoder = nn.Sequential(
            nn.Linear(bow_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.semantic_encoder = nn.Sequential(
            nn.Linear(semantic_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.document_encoder = nn.Sequential(
            nn.Linear(document_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        self.cra = CrossRepresentationAttention(
            hidden_dim=hidden,
            heads=config.attention_heads,
            dropout=config.dropout,
        )
        self.adaptive_fusion = AdaptiveFusion(hidden, config.dropout)
        self.concat_fusion = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        self.mu = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)
        self.topic_projection = (
            nn.Identity() if latent == config.n_topics else nn.Linear(latent, config.n_topics)
        )
        self.decoder = nn.Linear(config.n_topics, bow_dim)
        self.decoder_bn = nn.BatchNorm1d(bow_dim, affine=False)
        self.dropout = nn.Dropout(config.dropout)

    def _encode_branches(
        self,
        bow: torch.Tensor,
        semantic: torch.Tensor,
        document: torch.Tensor,
        add_noise: bool = False,
    ) -> torch.Tensor:
        lexical_h = self.lexical_encoder(bow)
        semantic_h = self.semantic_encoder(semantic)
        document_h = self.document_encoder(document)

        active = torch.tensor(
            [
                float(self.config.use_lexical),
                float(self.config.use_semantic),
                float(self.config.use_document),
            ],
            dtype=bow.dtype,
            device=bow.device,
        ).view(1, 3, 1)

        branches = torch.stack([lexical_h, semantic_h, document_h], dim=1)
        branches = branches * active

        if add_noise and self.config.branch_noise_std > 0:
            branches = branches + torch.randn_like(branches) * self.config.branch_noise_std * active

        if self.config.use_cra:
            branches = self.cra(branches) * active
        return branches

    def _fuse(self, branches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        active = torch.tensor(
            [
                float(self.config.use_lexical),
                float(self.config.use_semantic),
                float(self.config.use_document),
            ],
            dtype=branches.dtype,
            device=branches.device,
        ).view(1, 3)

        if active.sum() == 0:
            raise ValueError("Setidaknya satu cabang representasi harus aktif.")

        if self.config.fusion_mode == "adaptive":
            scores = self.adaptive_fusion.scorer(branches).squeeze(-1)
            scores = scores.masked_fill(active.eq(0), -1e9)
            weights = torch.softmax(scores, dim=1)
            fused = torch.sum(branches * weights.unsqueeze(-1), dim=1)
        elif self.config.fusion_mode == "concat":
            fused = self.concat_fusion(branches.flatten(start_dim=1))
            weights = active / active.sum()
            weights = weights.expand(branches.shape[0], -1)
        elif self.config.fusion_mode == "average":
            denominator = active.sum().clamp_min(1.0)
            fused = branches.sum(dim=1) / denominator
            weights = active / denominator
            weights = weights.expand(branches.shape[0], -1)
        else:
            raise ValueError(
                "fusion_mode harus salah satu dari: adaptive, concat, average."
            )

        return fused, weights

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not mu.requires_grad:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(
        self,
        bow: torch.Tensor,
        semantic: torch.Tensor,
        document: torch.Tensor,
        add_noise: bool = False,
    ) -> dict[str, torch.Tensor]:
        branches = self._encode_branches(bow, semantic, document, add_noise=add_noise)
        fused, fusion_weights = self._fuse(branches)

        mu = self.mu(fused)
        logvar = self.logvar(fused).clamp(-10.0, 10.0)
        z = self.reparameterize(mu, logvar)
        topic_logits = self.topic_projection(z)
        theta = torch.softmax(topic_logits, dim=-1)

        reconstruction_logits = self.decoder(self.dropout(theta))
        if reconstruction_logits.shape[0] > 1:
            reconstruction_logits = self.decoder_bn(reconstruction_logits)
        log_word_prob = torch.log_softmax(reconstruction_logits, dim=-1)

        return {
            "branches": branches,
            "fused": fused,
            "fusion_weights": fusion_weights,
            "mu": mu,
            "logvar": logvar,
            "theta": theta,
            "log_word_prob": log_word_prob,
        }

    def topic_word_distribution(self) -> torch.Tensor:
        # decoder.weight: [vocabulary, topics]
        return torch.softmax(self.decoder.weight.T, dim=1)


def _pairwise_alignment_loss(branches: torch.Tensor, active_flags: list[bool]) -> torch.Tensor:
    losses = []
    active_indices = [index for index, flag in enumerate(active_flags) if flag]
    for position, left in enumerate(active_indices):
        for right in active_indices[position + 1 :]:
            cosine = F.cosine_similarity(branches[:, left], branches[:, right], dim=-1)
            losses.append((1.0 - cosine).mean())
    if not losses:
        return branches.new_tensor(0.0)
    return torch.stack(losses).mean()


def _topic_diversity_loss(beta: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(beta, p=2, dim=1)
    similarity = normalized @ normalized.T
    mask = ~torch.eye(similarity.shape[0], dtype=torch.bool, device=similarity.device)
    if mask.sum() == 0:
        return similarity.new_tensor(0.0)
    return similarity[mask].pow(2).mean()


def _multinomial_reconstruction_loss(
    bow_counts: torch.Tensor,
    log_word_prob: torch.Tensor,
) -> torch.Tensor:
    document_lengths = bow_counts.sum(dim=1).clamp_min(1.0)
    negative_log_likelihood = -(bow_counts * log_word_prob).sum(dim=1)
    return (negative_log_likelihood / document_lengths).mean()


class HMRNet:
    """
    HMRNet experiment implementation.

    The class exposes a topic-model-like API:
    fit(), transform(), get_topic_word_distribution(), and get_topics().
    """

    def __init__(self, config: Optional[HMRNetConfig] = None) -> None:
        self.config = config or HMRNetConfig()
        self.model: Optional[HMRNetCore] = None
        self.history: list[dict[str, float]] = []
        self.vocabulary: Optional[np.ndarray] = None
        self.device = torch.device(
            self.config.device
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )

    @staticmethod
    def _to_dense_float32(matrix: Any) -> np.ndarray:
        if sparse.issparse(matrix):
            return matrix.toarray().astype(np.float32)
        return np.asarray(matrix, dtype=np.float32)

    def _make_loader(
        self,
        bow: Any,
        semantic: np.ndarray,
        document: np.ndarray,
        shuffle: bool,
    ) -> DataLoader:
        bow_array = self._to_dense_float32(bow)
        dataset = TensorDataset(
            torch.from_numpy(bow_array),
            torch.from_numpy(np.asarray(semantic, dtype=np.float32)),
            torch.from_numpy(np.asarray(document, dtype=np.float32)),
        )
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
        )

    def fit(
        self,
        bow: Any,
        semantic_embeddings: np.ndarray,
        document_embeddings: np.ndarray,
        vocabulary: np.ndarray | list[str],
        validation_fraction: float = 0.10,
        verbose: bool = True,
    ) -> "HMRNet":
        set_global_seed(self.config.random_state)
        bow_array = self._to_dense_float32(bow)
        semantic = np.asarray(semantic_embeddings, dtype=np.float32)
        document = np.asarray(document_embeddings, dtype=np.float32)

        if not (len(bow_array) == len(semantic) == len(document)):
            raise ValueError("Jumlah baris bow, semantic, dan document harus sama.")

        self.vocabulary = np.asarray(vocabulary)
        n_samples = len(bow_array)
        validation_size = int(round(n_samples * validation_fraction))
        validation_size = min(max(validation_size, 1), max(n_samples - 1, 1))

        rng = np.random.default_rng(self.config.random_state)
        indices = rng.permutation(n_samples)
        val_idx = indices[:validation_size]
        train_idx = indices[validation_size:]
        if len(train_idx) == 0:
            train_idx = val_idx

        train_loader = self._make_loader(
            bow_array[train_idx], semantic[train_idx], document[train_idx], shuffle=True
        )
        val_loader = self._make_loader(
            bow_array[val_idx], semantic[val_idx], document[val_idx], shuffle=False
        )

        self.model = HMRNetCore(
            bow_dim=bow_array.shape[1],
            semantic_dim=semantic.shape[1],
            document_dim=document.shape[1],
            config=self.config,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        best_state = None
        best_val = float("inf")
        epochs_without_improvement = 0
        self.history = []

        for epoch in range(1, self.config.epochs + 1):
            train_metrics = self._run_epoch(train_loader, optimizer)
            with torch.no_grad():
                val_metrics = self._run_epoch(val_loader, optimizer=None)

            epoch_metrics = {
                "epoch": float(epoch),
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
            self.history.append(epoch_metrics)

            if verbose and (epoch == 1 or epoch % 5 == 0):
                print(
                    f"Epoch {epoch:03d} | "
                    f"train={train_metrics['total']:.4f} | "
                    f"val={val_metrics['total']:.4f}"
                )

            if val_metrics["total"] < best_val - 1e-5:
                best_val = val_metrics["total"]
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.config.patience:
                    if verbose:
                        print(f"Early stopping pada epoch {epoch}.")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def _run_epoch(
        self,
        loader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer],
    ) -> dict[str, float]:
        assert self.model is not None
        training = optimizer is not None
        self.model.train(training)

        totals = {
            "total": 0.0,
            "reconstruction": 0.0,
            "kl": 0.0,
            "alignment": 0.0,
            "diversity": 0.0,
            "stability": 0.0,
        }
        n_documents = 0

        for bow, semantic, document in loader:
            bow = bow.to(self.device, non_blocking=True)
            semantic = semantic.to(self.device, non_blocking=True)
            document = document.to(self.device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            output = self.model(bow, semantic, document)
            reconstruction = _multinomial_reconstruction_loss(
                bow, output["log_word_prob"]
            )
            kl = -0.5 * torch.mean(
                1.0 + output["logvar"] - output["mu"].pow(2) - output["logvar"].exp()
            )

            alignment = bow.new_tensor(0.0)
            if self.config.use_alignment_loss:
                alignment = _pairwise_alignment_loss(
                    output["branches"],
                    [
                        self.config.use_lexical,
                        self.config.use_semantic,
                        self.config.use_document,
                    ],
                )

            diversity = bow.new_tensor(0.0)
            if self.config.use_diversity_loss:
                diversity = _topic_diversity_loss(
                    self.model.topic_word_distribution()
                )

            stability = bow.new_tensor(0.0)
            if self.config.use_stability_loss and training:
                noisy_output = self.model(
                    bow, semantic, document, add_noise=True
                )
                stability = F.mse_loss(output["theta"], noisy_output["theta"])

            total = (
                reconstruction
                + self.config.kl_weight * kl
                + self.config.alignment_weight * alignment
                + self.config.diversity_weight * diversity
                + self.config.stability_weight * stability
            )

            if training:
                total.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip
                )
                optimizer.step()

            batch_size = bow.shape[0]
            n_documents += batch_size
            values = {
                "total": total,
                "reconstruction": reconstruction,
                "kl": kl,
                "alignment": alignment,
                "diversity": diversity,
                "stability": stability,
            }
            for key, value in values.items():
                totals[key] += float(value.detach().cpu()) * batch_size

        denominator = max(n_documents, 1)
        return {key: value / denominator for key, value in totals.items()}

    def transform(
        self,
        bow: Any,
        semantic_embeddings: np.ndarray,
        document_embeddings: np.ndarray,
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model belum dilatih atau dimuat.")

        loader = self._make_loader(
            bow, semantic_embeddings, document_embeddings, shuffle=False
        )
        self.model.eval()
        thetas = []
        with torch.no_grad():
            for bow_batch, semantic, document in loader:
                output = self.model(
                    bow_batch.to(self.device),
                    semantic.to(self.device),
                    document.to(self.device),
                )
                thetas.append(output["theta"].cpu().numpy())
        return np.vstack(thetas)

    def get_topic_word_distribution(self) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model belum dilatih atau dimuat.")
        self.model.eval()
        with torch.no_grad():
            return self.model.topic_word_distribution().cpu().numpy()

    def get_topics(self, top_n: int = 10) -> list[list[str]]:
        if self.vocabulary is None:
            raise RuntimeError("Vocabulary belum tersedia.")
        beta = self.get_topic_word_distribution()
        top_indices = np.argsort(-beta, axis=1)[:, :top_n]
        return [[str(self.vocabulary[index]) for index in row] for row in top_indices]

    def get_fusion_weights(
        self,
        bow: Any,
        semantic_embeddings: np.ndarray,
        document_embeddings: np.ndarray,
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model belum dilatih atau dimuat.")
        loader = self._make_loader(
            bow, semantic_embeddings, document_embeddings, shuffle=False
        )
        self.model.eval()
        weights = []
        with torch.no_grad():
            for bow_batch, semantic, document in loader:
                output = self.model(
                    bow_batch.to(self.device),
                    semantic.to(self.device),
                    document.to(self.device),
                )
                weights.append(output["fusion_weights"].cpu().numpy())
        return np.vstack(weights)

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Tidak ada model untuk disimpan.")
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.config),
                "state_dict": self.model.state_dict(),
                "vocabulary": self.vocabulary,
                "history": self.history,
                "dimensions": {
                    "bow_dim": self.model.lexical_encoder[0].in_features,
                    "semantic_dim": self.model.semantic_encoder[0].in_features,
                    "document_dim": self.model.document_encoder[0].in_features,
                },
            },
            output_path,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: Optional[str] = None,
    ) -> "HMRNet":
        try:
            checkpoint = torch.load(
                path,
                map_location=device or "cpu",
                weights_only=False,
            )
        except TypeError:
            # Compatibility with PyTorch releases before weights_only existed.
            checkpoint = torch.load(path, map_location=device or "cpu")
        config = HMRNetConfig(**checkpoint["config"])
        if device is not None:
            config.device = device

        instance = cls(config)
        dimensions = checkpoint["dimensions"]
        instance.model = HMRNetCore(
            bow_dim=dimensions["bow_dim"],
            semantic_dim=dimensions["semantic_dim"],
            document_dim=dimensions["document_dim"],
            config=config,
        ).to(instance.device)
        instance.model.load_state_dict(checkpoint["state_dict"])
        instance.vocabulary = checkpoint["vocabulary"]
        instance.history = checkpoint.get("history", [])
        return instance
