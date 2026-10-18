from __future__ import annotations

from itertools import combinations
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd


def sanitize_topics(
    topics: Iterable[Iterable[str]],
    vocabulary: Optional[set[str]] = None,
) -> list[list[str]]:
    cleaned = []
    for topic in topics:
        words = []
        seen = set()
        for word in topic:
            token = str(word).strip()
            if not token or token in seen:
                continue
            if vocabulary is not None and token not in vocabulary:
                continue
            seen.add(token)
            words.append(token)
        if words:
            cleaned.append(words)
    return cleaned


def compute_coherence_scores(
    topics: list[list[str]],
    processed_texts: list[str],
) -> dict[str, float]:
    """Compute C_v, C_NPMI, and UMass with Gensim."""
    from gensim.corpora import Dictionary
    from gensim.models import CoherenceModel

    tokenized = [text.split() for text in processed_texts if text.strip()]
    dictionary = Dictionary(tokenized)
    corpus = [dictionary.doc2bow(tokens) for tokens in tokenized]
    vocabulary = set(dictionary.token2id)

    valid_topics = sanitize_topics(topics, vocabulary)
    if not valid_topics:
        return {"c_v": np.nan, "c_npmi": np.nan, "u_mass": np.nan}

    scores = {}
    for coherence in ("c_v", "c_npmi"):
        model = CoherenceModel(
            topics=valid_topics,
            texts=tokenized,
            dictionary=dictionary,
            coherence=coherence,
            processes=1,
        )
        scores[coherence] = float(model.get_coherence())

    umass_model = CoherenceModel(
        topics=valid_topics,
        corpus=corpus,
        dictionary=dictionary,
        coherence="u_mass",
        processes=1,
    )
    scores["u_mass"] = float(umass_model.get_coherence())
    return scores


def topic_diversity(topics: list[list[str]], top_n: Optional[int] = None) -> float:
    if not topics:
        return np.nan
    truncated = [topic[:top_n] if top_n else topic for topic in topics]
    flattened = [word for topic in truncated for word in topic]
    return len(set(flattened)) / max(len(flattened), 1)


def topic_redundancy(topics: list[list[str]], top_n: Optional[int] = None) -> float:
    if len(topics) < 2:
        return 0.0
    values = []
    truncated = [set(topic[:top_n] if top_n else topic) for topic in topics]
    for left, right in combinations(truncated, 2):
        union = left | right
        values.append(len(left & right) / max(len(union), 1))
    return float(np.mean(values))


def topic_uniqueness(topics: list[list[str]], top_n: Optional[int] = None) -> float:
    """
    Mean inverse topic frequency of top words.
    A value near 1 means words rarely repeat across topics.
    """
    if not topics:
        return np.nan
    truncated = [topic[:top_n] if top_n else topic for topic in topics]
    frequency: dict[str, int] = {}
    for topic in truncated:
        for word in set(topic):
            frequency[word] = frequency.get(word, 0) + 1

    values = [
        1.0 / frequency[word]
        for topic in truncated
        for word in topic
        if word in frequency
    ]
    return float(np.mean(values)) if values else np.nan


def evaluate_topics(
    model_name: str,
    topics: list[list[str]],
    processed_texts: list[str],
    top_n: int = 10,
) -> dict[str, Any]:
    truncated = [topic[:top_n] for topic in topics]
    coherence = compute_coherence_scores(truncated, processed_texts)
    return {
        "Model": model_name,
        "C_v": coherence["c_v"],
        "C_NPMI": coherence["c_npmi"],
        "UMass": coherence["u_mass"],
        "Topic_Diversity": topic_diversity(truncated),
        "Topic_Redundancy": topic_redundancy(truncated),
        "Topic_Uniqueness": topic_uniqueness(truncated),
        "Number_of_Topics": len(truncated),
    }


def evaluate_many_models(
    model_topics: dict[str, list[list[str]]],
    processed_texts: list[str],
    top_n: int = 10,
) -> pd.DataFrame:
    rows = [
        evaluate_topics(name, topics, processed_texts, top_n=top_n)
        for name, topics in model_topics.items()
    ]
    return pd.DataFrame(rows).sort_values("C_v", ascending=False).reset_index(drop=True)
