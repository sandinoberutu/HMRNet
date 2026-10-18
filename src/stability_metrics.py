from __future__ import annotations

from itertools import combinations
from typing import Literal, Optional

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jensenshannon
from scipy.stats import t


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    matrix = np.clip(matrix, 0.0, None)
    denominator = matrix.sum(axis=1, keepdims=True)
    return matrix / np.clip(denominator, 1e-12, None)


def topic_similarity_matrix(
    beta_left: np.ndarray,
    beta_right: np.ndarray,
    metric: Literal["js", "cosine"] = "js",
) -> np.ndarray:
    left = _normalize_rows(beta_left)
    right = _normalize_rows(beta_right)
    similarity = np.zeros((left.shape[0], right.shape[0]), dtype=np.float64)

    if metric == "js":
        for i in range(left.shape[0]):
            for j in range(right.shape[0]):
                # scipy returns sqrt(JS divergence); squaring gives JS divergence.
                js_divergence = float(jensenshannon(left[i], right[j]) ** 2)
                similarity[i, j] = 1.0 - js_divergence
    elif metric == "cosine":
        left_norm = left / np.clip(np.linalg.norm(left, axis=1, keepdims=True), 1e-12, None)
        right_norm = right / np.clip(np.linalg.norm(right, axis=1, keepdims=True), 1e-12, None)
        similarity = left_norm @ right_norm.T
    else:
        raise ValueError("metric harus 'js' atau 'cosine'.")

    return np.clip(similarity, -1.0, 1.0)


def hungarian_topic_stability(
    beta_left: np.ndarray,
    beta_right: np.ndarray,
    metric: Literal["js", "cosine"] = "js",
) -> tuple[float, np.ndarray, np.ndarray]:
    similarity = topic_similarity_matrix(beta_left, beta_right, metric=metric)
    row_indices, column_indices = linear_sum_assignment(-similarity)
    matched = similarity[row_indices, column_indices]
    return float(np.mean(matched)), row_indices, column_indices


def confidence_interval(
    values: list[float] | np.ndarray,
    confidence: float = 0.95,
) -> tuple[float, float]:
    values_array = np.asarray(values, dtype=float)
    values_array = values_array[np.isfinite(values_array)]
    if len(values_array) < 2:
        value = float(values_array[0]) if len(values_array) else np.nan
        return value, value

    mean = float(values_array.mean())
    standard_error = float(values_array.std(ddof=1) / np.sqrt(len(values_array)))
    critical = float(t.ppf((1 + confidence) / 2, df=len(values_array) - 1))
    return mean - critical * standard_error, mean + critical * standard_error


def evaluate_stability_runs(
    beta_by_run: dict[int | str, np.ndarray],
    metric: Literal["js", "cosine"] = "js",
    confidence: float = 0.95,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    for run_left, run_right in combinations(beta_by_run.keys(), 2):
        score, _, _ = hungarian_topic_stability(
            beta_by_run[run_left],
            beta_by_run[run_right],
            metric=metric,
        )
        rows.append(
            {
                "Run_1": run_left,
                "Run_2": run_right,
                "Topic_Stability": score,
            }
        )

    pairwise = pd.DataFrame(rows)
    values = pairwise["Topic_Stability"].to_numpy() if len(pairwise) else np.array([])
    ci_low, ci_high = confidence_interval(values, confidence)
    summary = {
        "Mean_TS": float(np.mean(values)) if len(values) else np.nan,
        "Std_TS": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "CI_Lower": float(ci_low),
        "CI_Upper": float(ci_high),
        "Number_of_Pairs": int(len(values)),
    }
    return pairwise, summary


def align_topic_matrices(
    reference_beta: np.ndarray,
    candidate_beta: np.ndarray,
    metric: Literal["js", "cosine"] = "js",
) -> np.ndarray:
    """Reorder candidate topics to match a reference run."""
    _, reference_indices, candidate_indices = hungarian_topic_stability(
        reference_beta, candidate_beta, metric=metric
    )
    order = np.argsort(reference_indices)
    return candidate_beta[candidate_indices[order]]
