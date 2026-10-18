from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from scipy import sparse
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.preprocessing import normalize
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .preprocessing import set_global_seed


@dataclass
class TopicModelResult:
    name: str
    topics: list[list[str]]
    topic_word: Optional[np.ndarray] = None
    document_topic: Optional[np.ndarray] = None
    model: Any = None
    metadata: Optional[dict[str, Any]] = None


def _top_words(
    matrix: np.ndarray,
    vocabulary: np.ndarray | list[str],
    top_n: int,
) -> list[list[str]]:
    vocab = np.asarray(vocabulary)
    indices = np.argsort(-matrix, axis=1)[:, :top_n]
    return [[str(vocab[index]) for index in row] for row in indices]


def fit_lda(
    bow: Any,
    vocabulary: np.ndarray | list[str],
    n_topics: int,
    top_n: int = 10,
    random_state: int = 42,
    max_iter: int = 30,
) -> TopicModelResult:
    model = LatentDirichletAllocation(
        n_components=n_topics,
        learning_method="batch",
        max_iter=max_iter,
        random_state=random_state,
        n_jobs=-1,
    )
    document_topic = model.fit_transform(bow)
    topic_word = normalize(model.components_, norm="l1", axis=1)
    return TopicModelResult(
        name="LDA",
        topics=_top_words(topic_word, vocabulary, top_n),
        topic_word=np.asarray(topic_word),
        document_topic=np.asarray(document_topic),
        model=model,
    )


def fit_nmf(
    tfidf: Any,
    vocabulary: np.ndarray | list[str],
    n_topics: int,
    top_n: int = 10,
    random_state: int = 42,
    max_iter: int = 400,
) -> TopicModelResult:
    model = NMF(
        n_components=n_topics,
        init="nndsvda",
        solver="cd",
        beta_loss="frobenius",
        max_iter=max_iter,
        random_state=random_state,
    )
    document_topic = model.fit_transform(tfidf)
    document_topic = normalize(document_topic, norm="l1", axis=1)
    topic_word = normalize(model.components_, norm="l1", axis=1)
    return TopicModelResult(
        name="NMF",
        topics=_top_words(topic_word, vocabulary, top_n),
        topic_word=np.asarray(topic_word),
        document_topic=np.asarray(document_topic),
        model=model,
    )


class _NeuralTopicVAE(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_topics: int,
        hidden_dim: int,
        model_type: str,
        dropout: float,
    ) -> None:
        super().__init__()
        self.model_type = model_type
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.Softplus(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
        )
        self.mu = nn.Linear(hidden_dim, n_topics)
        self.logvar = nn.Linear(hidden_dim, n_topics)
        self.decoder = nn.Linear(n_topics, vocab_size)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(-10.0, 10.0)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std

        if self.model_type == "prodlda":
            theta = torch.softmax(z, dim=1)
            word_logits = self.decoder(theta)
        elif self.model_type == "nvdm":
            theta = torch.softmax(z, dim=1)
            word_logits = self.decoder(z)
        else:
            raise ValueError("model_type harus prodlda atau nvdm.")

        return {
            "mu": mu,
            "logvar": logvar,
            "theta": theta,
            "log_word_prob": torch.log_softmax(word_logits, dim=1),
        }

    def beta(self) -> torch.Tensor:
        return torch.softmax(self.decoder.weight.T, dim=1)


def fit_neural_vae_topic_model(
    bow: Any,
    vocabulary: np.ndarray | list[str],
    n_topics: int,
    model_type: str,
    top_n: int = 10,
    hidden_dim: int = 256,
    epochs: int = 60,
    batch_size: int = 128,
    learning_rate: float = 2e-3,
    random_state: int = 42,
    device: Optional[str] = None,
    verbose: bool = True,
) -> TopicModelResult:
    set_global_seed(random_state)
    resolved_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    x = bow.toarray() if sparse.issparse(bow) else np.asarray(bow)
    x = x.astype(np.float32)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(x)),
        batch_size=batch_size,
        shuffle=True,
    )
    model = _NeuralTopicVAE(
        vocab_size=x.shape[1],
        n_topics=n_topics,
        hidden_dim=hidden_dim,
        model_type=model_type,
        dropout=0.20,
    ).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(1, epochs + 1):
        running = 0.0
        for (batch,) in loader:
            batch = batch.to(resolved_device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            lengths = batch.sum(dim=1).clamp_min(1.0)
            reconstruction = (
                -(batch * output["log_word_prob"]).sum(dim=1) / lengths
            ).mean()
            kl = -0.5 * torch.mean(
                1 + output["logvar"] - output["mu"].pow(2) - output["logvar"].exp()
            )
            loss = reconstruction + 0.10 * kl
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.detach().cpu()) * len(batch)

        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(f"{model_type.upper()} epoch {epoch:03d}: {running / len(x):.4f}")

    model.eval()
    document_topic = []
    inference_loader = DataLoader(
        TensorDataset(torch.from_numpy(x)),
        batch_size=batch_size,
        shuffle=False,
    )
    with torch.no_grad():
        for (batch,) in inference_loader:
            document_topic.append(
                model(batch.to(resolved_device))["theta"].cpu().numpy()
            )
        topic_word = model.beta().cpu().numpy()

    name = "ProdLDA" if model_type == "prodlda" else "NVDM"
    return TopicModelResult(
        name=name,
        topics=_top_words(topic_word, vocabulary, top_n),
        topic_word=topic_word,
        document_topic=np.vstack(document_topic),
        model=model,
    )


def fit_prodlda(**kwargs: Any) -> TopicModelResult:
    return fit_neural_vae_topic_model(model_type="prodlda", **kwargs)


def fit_nvdm(**kwargs: Any) -> TopicModelResult:
    return fit_neural_vae_topic_model(model_type="nvdm", **kwargs)


def fit_combined_tm(
    raw_texts: list[str],
    processed_texts: list[str],
    n_topics: int,
    top_n: int = 10,
    contextual_model: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ),
    epochs: int = 50,
    batch_size: int = 64,
    random_state: int = 42,
) -> TopicModelResult:
    """Fit CombinedTM through contextualized-topic-models."""
    set_global_seed(random_state)
    from contextualized_topic_models.models.ctm import CombinedTM
    from contextualized_topic_models.utils.data_preparation import (
        TopicModelDataPreparation,
    )

    preparation = TopicModelDataPreparation(contextualized_model=contextual_model)
    training_dataset = preparation.fit(
        text_for_contextual=raw_texts,
        text_for_bow=processed_texts,
    )
    model = CombinedTM(
        bow_size=len(preparation.vocab),
        contextual_size=training_dataset.X_contextual.shape[1],
        n_components=n_topics,
        num_epochs=epochs,
        batch_size=batch_size,
    )
    model.fit(training_dataset)
    topics_dict = model.get_topics(top_n)
    topics = [topics_dict[index] for index in sorted(topics_dict)]
    document_topic = model.get_doc_topic_distribution(training_dataset, n_samples=20)

    topic_word = None
    for attribute in ("get_topic_word_matrix", "get_topic_word_distribution"):
        if hasattr(model, attribute):
            try:
                topic_word = np.asarray(getattr(model, attribute)())
                break
            except Exception:
                pass

    return TopicModelResult(
        name="CombinedTM",
        topics=topics,
        topic_word=topic_word,
        document_topic=np.asarray(document_topic),
        model=model,
        metadata={"preparation": preparation, "training_dataset": training_dataset},
    )


def fit_bertopic(
    raw_texts: list[str],
    n_topics: Optional[int] = None,
    top_n: int = 10,
    embedding_model: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ),
    random_state: int = 42,
) -> TopicModelResult:
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP

    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=random_state,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=max(10, len(raw_texts) // 100),
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        nr_topics=n_topics,
        calculate_probabilities=True,
        verbose=True,
    )
    assignments, probabilities = model.fit_transform(raw_texts)

    topic_ids = [
        int(topic_id)
        for topic_id in model.get_topic_info()["Topic"].tolist()
        if int(topic_id) != -1
    ]
    topics = [
        [word for word, _ in model.get_topic(topic_id)[:top_n]]
        for topic_id in topic_ids
    ]

    document_topic = None
    if probabilities is not None:
        document_topic = np.asarray(probabilities)
    else:
        document_topic = np.zeros((len(raw_texts), len(topic_ids)), dtype=np.float32)
        topic_to_column = {topic_id: i for i, topic_id in enumerate(topic_ids)}
        for row, topic_id in enumerate(assignments):
            if topic_id in topic_to_column:
                document_topic[row, topic_to_column[topic_id]] = 1.0

    topic_word = None
    if getattr(model, "c_tf_idf_", None) is not None:
        info = model.get_topic_info()
        valid_rows = [
            row_index
            for row_index, topic_id in enumerate(info["Topic"].tolist())
            if int(topic_id) != -1
        ]
        try:
            topic_word = model.c_tf_idf_[valid_rows].toarray()
            topic_word = normalize(topic_word, norm="l1", axis=1)
        except Exception:
            topic_word = None

    return TopicModelResult(
        name="BERTopic",
        topics=topics,
        topic_word=topic_word,
        document_topic=document_topic,
        model=model,
        metadata={"assignments": assignments, "topic_ids": topic_ids},
    )


def fit_top2vec(
    raw_texts: list[str],
    top_n: int = 10,
    embedding_model: str = "doc2vec",
    speed: str = "learn",
    workers: Optional[int] = None,
) -> TopicModelResult:
    from top2vec import Top2Vec

    model = Top2Vec(
        documents=raw_texts,
        embedding_model=embedding_model,
        speed=speed,
        workers=workers,
        min_count=5,
        verbose=True,
    )
    topic_words, word_scores, topic_nums = model.get_topics()
    topics = [
        [str(word) for word in words[:top_n]]
        for words in topic_words
    ]

    document_topic = None
    if hasattr(model, "document_vectors") and hasattr(model, "topic_vectors"):
        document_vectors = np.asarray(model.document_vectors)
        topic_vectors = np.asarray(model.topic_vectors)
        document_vectors = normalize(document_vectors)
        topic_vectors = normalize(topic_vectors)
        similarities = document_vectors @ topic_vectors.T
        shifted = similarities - similarities.max(axis=1, keepdims=True)
        document_topic = np.exp(shifted)
        document_topic /= document_topic.sum(axis=1, keepdims=True).clip(min=1e-12)

    return TopicModelResult(
        name="Top2Vec",
        topics=topics,
        topic_word=None,
        document_topic=document_topic,
        model=model,
        metadata={
            "topic_nums": np.asarray(topic_nums),
            "word_scores": word_scores,
        },
    )
