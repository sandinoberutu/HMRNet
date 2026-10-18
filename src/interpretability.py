from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def encode_topic_words(
    topics: list[list[str]],
    encoder: object,
) -> list[np.ndarray]:
    return [
        np.asarray(
            encoder.encode(
                topic,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        )
        for topic in topics
    ]


def semantic_compactness(
    topics: list[list[str]],
    encoder: object,
) -> tuple[pd.DataFrame, float]:
    rows = []
    for topic_index, words in enumerate(topics):
        embeddings = np.asarray(
            encoder.encode(
                words,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        )
        centroid = embeddings.mean(axis=0, keepdims=True)
        centroid /= np.clip(np.linalg.norm(centroid, axis=1, keepdims=True), 1e-12, None)
        score = float(cosine_similarity(embeddings, centroid).mean())
        rows.append({"Topic": topic_index, "Semantic_Compactness": score})
    frame = pd.DataFrame(rows)
    return frame, float(frame["Semantic_Compactness"].mean())


def topic_centroids(
    topics: list[list[str]],
    encoder: object,
) -> np.ndarray:
    centroids = []
    for words in topics:
        embeddings = np.asarray(
            encoder.encode(
                words,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        )
        centroid = embeddings.mean(axis=0)
        centroid /= max(np.linalg.norm(centroid), 1e-12)
        centroids.append(centroid)
    return np.vstack(centroids)


def inter_topic_separation(
    topics: list[list[str]],
    encoder: object,
) -> float:
    centroids = topic_centroids(topics, encoder)
    similarity = cosine_similarity(centroids)
    upper = np.triu_indices_from(similarity, k=1)
    if len(upper[0]) == 0:
        return 1.0
    return float(1.0 - similarity[upper].mean())


def automatic_intruder_word_test(
    topics: list[list[str]],
    encoder: object,
    repetitions_per_topic: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, float]:
    """
    Add a word from another topic and identify the word farthest from
    the centroid of the remaining candidate set.
    """
    rng = np.random.default_rng(random_state)
    rows = []
    for topic_index, words in enumerate(topics):
        other_topics = [index for index in range(len(topics)) if index != topic_index]
        if not other_topics or len(words) < 2:
            continue

        for repetition in range(repetitions_per_topic):
            source_topic = int(rng.choice(other_topics))
            intruder = str(rng.choice(topics[source_topic]))
            candidates = list(words) + [intruder]
            embeddings = np.asarray(
                encoder.encode(
                    candidates,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
            )

            # Leave-one-out cohesion: an intruder should have the lowest
            # similarity to the centroid formed by all other candidates.
            scores = []
            for candidate_index in range(len(candidates)):
                mask = np.ones(len(candidates), dtype=bool)
                mask[candidate_index] = False
                centroid = embeddings[mask].mean(axis=0, keepdims=True)
                centroid /= np.clip(
                    np.linalg.norm(centroid, axis=1, keepdims=True),
                    1e-12,
                    None,
                )
                score = float(
                    cosine_similarity(
                        embeddings[candidate_index : candidate_index + 1],
                        centroid,
                    )[0, 0]
                )
                scores.append(score)

            detected = candidates[int(np.argmin(scores))]
            rows.append(
                {
                    "Topic": topic_index,
                    "Repetition": repetition,
                    "Intruder": intruder,
                    "Detected": detected,
                    "Correct": int(detected == intruder),
                }
            )

    frame = pd.DataFrame(rows)
    score = float(frame["Correct"].mean()) if len(frame) else np.nan
    return frame, score


def topic_intrusion_test(
    document_embeddings: np.ndarray,
    topic_centroid_embeddings: np.ndarray,
    gold_topics: np.ndarray,
    candidate_count: int = 4,
    random_state: int = 42,
) -> tuple[pd.DataFrame, float]:
    """
    Automatic topic intrusion:
    compare the gold dominant topic with sampled distractor topics.
    """
    rng = np.random.default_rng(random_state)
    document_embeddings = np.asarray(document_embeddings)
    topic_centroid_embeddings = np.asarray(topic_centroid_embeddings)
    gold_topics = np.asarray(gold_topics, dtype=int)

    rows = []
    all_topics = np.arange(len(topic_centroid_embeddings))
    for index, gold in enumerate(gold_topics):
        distractor_pool = all_topics[all_topics != gold]
        number_to_sample = min(candidate_count - 1, len(distractor_pool))
        distractors = (
            rng.choice(distractor_pool, size=number_to_sample, replace=False)
            if number_to_sample > 0
            else np.array([], dtype=int)
        )
        candidates = np.concatenate([[gold], distractors])
        similarities = cosine_similarity(
            document_embeddings[index : index + 1],
            topic_centroid_embeddings[candidates],
        )[0]
        prediction = int(candidates[int(np.argmax(similarities))])
        rows.append(
            {
                "Document": index,
                "Gold_Topic": int(gold),
                "Predicted_Topic": prediction,
                "Correct": int(prediction == gold),
            }
        )

    frame = pd.DataFrame(rows)
    score = float(frame["Correct"].mean()) if len(frame) else np.nan
    return frame, score
