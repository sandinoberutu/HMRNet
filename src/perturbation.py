from __future__ import annotations

import random
import string
from typing import Iterable, Optional

import numpy as np
import pandas as pd


DEFAULT_SLANG = {
    "tidak": ["gak", "nggak", "ga"],
    "sudah": ["udah"],
    "sangat": ["banget"],
    "dengan": ["dgn"],
    "karena": ["krn"],
    "untuk": ["utk"],
    "yang": ["yg"],
    "makan": ["mkn"],
    "pemerintah": ["pemerintahh"],
}


def subsample_dataframe(
    df: pd.DataFrame,
    proportion: float,
    random_state: int = 42,
) -> pd.DataFrame:
    if not 0 < proportion <= 1:
        raise ValueError("proportion harus berada dalam interval (0, 1].")
    if proportion == 1.0:
        return df.copy().reset_index(drop=True)
    return df.sample(frac=proportion, random_state=random_state).reset_index(drop=True)


def random_token_deletion(
    text: str,
    noise_level: float,
    rng: np.random.Generator,
) -> str:
    tokens = str(text).split()
    if len(tokens) <= 1:
        return str(text)
    kept = [token for token in tokens if rng.random() > noise_level]
    return " ".join(kept or [tokens[int(rng.integers(0, len(tokens)))]])


def synthetic_typo(
    text: str,
    noise_level: float,
    rng: np.random.Generator,
) -> str:
    tokens = str(text).split()
    output = []
    for token in tokens:
        if len(token) >= 4 and rng.random() < noise_level:
            operation = int(rng.integers(0, 3))
            position = int(rng.integers(1, len(token) - 1))
            if operation == 0:
                token = token[:position] + token[position + 1 :]
            elif operation == 1:
                chars = list(token)
                chars[position - 1], chars[position] = chars[position], chars[position - 1]
                token = "".join(chars)
            else:
                replacement = random.choice(string.ascii_lowercase)
                token = token[:position] + replacement + token[position + 1 :]
        output.append(token)
    return " ".join(output)


def stopword_removal(
    text: str,
    noise_level: float,
    rng: np.random.Generator,
    stopwords: set[str],
) -> str:
    tokens = str(text).split()
    output = [
        token
        for token in tokens
        if not (token.lower() in stopwords and rng.random() < noise_level)
    ]
    return " ".join(output or tokens[:1])


def slang_injection(
    text: str,
    noise_level: float,
    rng: np.random.Generator,
    slang_map: Optional[dict[str, list[str]]] = None,
) -> str:
    mapping = slang_map or DEFAULT_SLANG
    output = []
    for token in str(text).split():
        key = token.lower()
        if key in mapping and rng.random() < noise_level:
            output.append(str(rng.choice(mapping[key])))
        else:
            output.append(token)
    return " ".join(output)


def perturb_texts(
    texts: Iterable[str],
    noise_type: str,
    noise_level: float,
    random_state: int = 42,
    stopwords: Optional[set[str]] = None,
    slang_map: Optional[dict[str, list[str]]] = None,
) -> list[str]:
    if not 0 <= noise_level <= 1:
        raise ValueError("noise_level harus berada dalam interval [0, 1].")
    rng = np.random.default_rng(random_state)

    output = []
    for text in texts:
        if noise_type == "token_deletion":
            value = random_token_deletion(text, noise_level, rng)
        elif noise_type == "typo":
            value = synthetic_typo(text, noise_level, rng)
        elif noise_type == "stopword_removal":
            value = stopword_removal(text, noise_level, rng, stopwords or set())
        elif noise_type == "slang":
            value = slang_injection(text, noise_level, rng, slang_map)
        else:
            raise ValueError(
                "noise_type harus token_deletion, typo, stopword_removal, atau slang."
            )
        output.append(value)
    return output


def remove_documents(
    texts: list[str],
    noise_level: float,
    random_state: int = 42,
) -> list[str]:
    if not 0 <= noise_level < 1:
        raise ValueError("noise_level document removal harus dalam [0, 1).")
    rng = np.random.default_rng(random_state)
    keep_count = max(1, int(round(len(texts) * (1 - noise_level))))
    indices = np.sort(rng.choice(len(texts), size=keep_count, replace=False))
    return [texts[index] for index in indices]


def relative_performance_degradation(
    full_performance: float,
    perturbed_performance: float,
    higher_is_better: bool = True,
) -> float:
    denominator = abs(full_performance)
    if denominator < 1e-12:
        return np.nan
    if higher_is_better:
        return (full_performance - perturbed_performance) / denominator
    return (perturbed_performance - full_performance) / denominator


def create_temporal_phases(
    df: pd.DataFrame,
    date_column: str,
    labels: tuple[str, str, str] = (
        "Fase Awal Peluncuran",
        "Fase Implementasi",
        "Fase Evaluasi Publik",
    ),
) -> pd.DataFrame:
    result = df.copy()
    dates = pd.to_datetime(result[date_column], errors="coerce")
    valid = dates.notna()
    if valid.sum() < 3:
        raise ValueError("Data tanggal valid terlalu sedikit untuk tiga fase temporal.")

    ranks = dates[valid].rank(method="first", pct=True)
    phase = pd.Series(index=result.index, dtype="object")
    phase.loc[ranks.index] = pd.cut(
        ranks,
        bins=[0.0, 1 / 3, 2 / 3, 1.0],
        labels=list(labels),
        include_lowest=True,
    ).astype(str)
    result["temporal_phase"] = phase
    return result
