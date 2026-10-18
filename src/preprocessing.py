from __future__ import annotations

import html
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import joblib
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize


@dataclass
class PreprocessConfig:
    text_column: str = "text"
    label_column: Optional[str] = "sentiment"
    date_column: Optional[str] = "created_at"
    language: str = "indonesian"
    lowercase: bool = True
    remove_urls: bool = True
    remove_mentions: bool = True
    remove_hashtag_symbol: bool = True
    remove_numbers: bool = False
    min_token_length: int = 2
    min_df: int = 3
    max_df: float = 0.95
    max_features: int = 12000
    ngram_range: tuple[int, int] = (1, 2)
    semantic_model_name: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    semantic_batch_size: int = 64
    document_dim: int = 256
    random_state: int = 42


URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
MULTISPACE_RE = re.compile(r"\s+")
NON_TEXT_RE = re.compile(r"[^\w\s#]", flags=re.UNICODE)
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")


def set_global_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set common random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


class TextPreprocessor:
    """Conservative social-media text cleaning for topic modeling."""

    def __init__(
        self,
        config: Optional[PreprocessConfig] = None,
        stopwords: Optional[Iterable[str]] = None,
    ) -> None:
        self.config = config or PreprocessConfig()
        self.stopwords = set(stopwords or [])

    def clean(self, text: Any) -> str:
        if text is None or (isinstance(text, float) and np.isnan(text)):
            return ""

        value = html.unescape(str(text))
        if self.config.lowercase:
            value = value.lower()
        if self.config.remove_urls:
            value = URL_RE.sub(" ", value)
        if self.config.remove_mentions:
            value = MENTION_RE.sub(" ", value)
        if self.config.remove_hashtag_symbol:
            value = value.replace("#", " ")
        if self.config.remove_numbers:
            value = NUMBER_RE.sub(" ", value)

        value = NON_TEXT_RE.sub(" ", value)
        tokens = [
            token
            for token in value.split()
            if len(token) >= self.config.min_token_length
            and token not in self.stopwords
        ]
        return MULTISPACE_RE.sub(" ", " ".join(tokens)).strip()

    def transform(self, texts: Iterable[Any]) -> list[str]:
        return [self.clean(text) for text in texts]


def infer_column(
    df: pd.DataFrame,
    candidates: list[str],
    required: bool = False,
) -> Optional[str]:
    lower_map = {str(column).lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return str(lower_map[candidate.lower()])
    if required:
        raise ValueError(
            f"Tidak menemukan kolom yang sesuai. Kandidat: {candidates}. "
            f"Kolom tersedia: {list(df.columns)}"
        )
    return None


def load_text_dataset(
    path: str | Path,
    text_column: Optional[str] = None,
    label_column: Optional[str] = None,
    date_column: Optional[str] = None,
) -> tuple[pd.DataFrame, dict[str, Optional[str]]]:
    """Load CSV/XLSX/JSON/Parquet and infer common columns."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(file_path)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(file_path)
    elif suffix in {".json", ".jsonl"}:
        df = pd.read_json(file_path, lines=suffix == ".jsonl")
    elif suffix == ".parquet":
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Format dataset belum didukung: {suffix}")

    resolved_text = text_column or infer_column(
        df,
        ["text", "tweet", "content", "document", "sentence", "review"],
        required=True,
    )
    resolved_label = label_column or infer_column(
        df,
        ["sentiment", "label", "class", "target"],
        required=False,
    )
    resolved_date = date_column or infer_column(
        df,
        ["created_at", "date", "timestamp", "time", "datetime"],
        required=False,
    )

    df = df.copy()
    df[resolved_text] = df[resolved_text].fillna("").astype(str)
    df = df[df[resolved_text].str.strip().ne("")].reset_index(drop=True)

    if resolved_date is not None:
        df[resolved_date] = pd.to_datetime(df[resolved_date], errors="coerce")

    return df, {
        "text_column": resolved_text,
        "label_column": resolved_label,
        "date_column": resolved_date,
    }


def encode_semantic_embeddings(
    texts: list[str],
    model_name: str,
    batch_size: int = 64,
    normalize_embeddings: bool = True,
    device: Optional[str] = None,
) -> np.ndarray:
    """Create multilingual sentence embeddings."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )
    return embeddings.astype(np.float32)


def build_feature_bundle(
    raw_texts: list[str],
    processed_texts: list[str],
    config: Optional[PreprocessConfig] = None,
    semantic_embeddings: Optional[np.ndarray] = None,
    device: Optional[str] = None,
) -> dict[str, Any]:
    """
    Build the shared representations used by HMRNet:
    lexical = normalized count vector,
    semantic = sentence-transformer embedding,
    document = TF-IDF reduced with TruncatedSVD.
    """
    cfg = config or PreprocessConfig()

    count_vectorizer = CountVectorizer(
        min_df=cfg.min_df,
        max_df=cfg.max_df,
        max_features=cfg.max_features,
        ngram_range=cfg.ngram_range,
        token_pattern=r"(?u)\b\w\w+\b",
    )
    bow = count_vectorizer.fit_transform(processed_texts).astype(np.float32)
    if bow.shape[1] == 0:
        raise ValueError(
            "Vocabulary kosong. Kurangi min_df, naikkan max_df, atau periksa preprocessing."
        )

    tfidf_vectorizer = TfidfVectorizer(
        vocabulary=count_vectorizer.vocabulary_,
        ngram_range=cfg.ngram_range,
        token_pattern=r"(?u)\b\w\w+\b",
        sublinear_tf=True,
    )
    tfidf = tfidf_vectorizer.fit_transform(processed_texts).astype(np.float32)

    max_svd = max(2, min(cfg.document_dim, tfidf.shape[0] - 1, tfidf.shape[1] - 1))
    svd = TruncatedSVD(n_components=max_svd, random_state=cfg.random_state)
    document_embeddings = svd.fit_transform(tfidf).astype(np.float32)
    document_embeddings = normalize(document_embeddings).astype(np.float32)

    if semantic_embeddings is None:
        semantic_embeddings = encode_semantic_embeddings(
            raw_texts,
            model_name=cfg.semantic_model_name,
            batch_size=cfg.semantic_batch_size,
            device=device,
        )

    id_to_token = np.asarray(count_vectorizer.get_feature_names_out())
    bow_normalized = normalize(bow, norm="l1", axis=1).astype(np.float32)

    return {
        "raw_texts": list(raw_texts),
        "processed_texts": list(processed_texts),
        "bow": bow,
        "bow_normalized": bow_normalized,
        "tfidf": tfidf,
        "semantic_embeddings": np.asarray(semantic_embeddings, dtype=np.float32),
        "document_embeddings": document_embeddings,
        "vocabulary": id_to_token,
        "count_vectorizer": count_vectorizer,
        "tfidf_vectorizer": tfidf_vectorizer,
        "svd": svd,
        "config": asdict(cfg),
    }


def save_feature_bundle(bundle: dict[str, Any], directory: str | Path) -> None:
    """Persist sparse matrices, dense arrays, metadata, and transformers."""
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    sparse.save_npz(output_dir / "bow.npz", bundle["bow"])
    sparse.save_npz(output_dir / "bow_normalized.npz", bundle["bow_normalized"])
    sparse.save_npz(output_dir / "tfidf.npz", bundle["tfidf"])

    np.save(output_dir / "semantic_embeddings.npy", bundle["semantic_embeddings"])
    np.save(output_dir / "document_embeddings.npy", bundle["document_embeddings"])
    np.save(output_dir / "vocabulary.npy", bundle["vocabulary"], allow_pickle=True)

    pd.DataFrame(
        {
            "raw_text": bundle["raw_texts"],
            "processed_text": bundle["processed_texts"],
        }
    ).to_parquet(output_dir / "texts.parquet", index=False)

    joblib.dump(bundle["count_vectorizer"], output_dir / "count_vectorizer.joblib")
    joblib.dump(bundle["tfidf_vectorizer"], output_dir / "tfidf_vectorizer.joblib")
    joblib.dump(bundle["svd"], output_dir / "svd.joblib")
    joblib.dump(bundle["config"], output_dir / "preprocess_config.joblib")


def load_feature_bundle(directory: str | Path) -> dict[str, Any]:
    input_dir = Path(directory)
    texts = pd.read_parquet(input_dir / "texts.parquet")

    return {
        "raw_texts": texts["raw_text"].astype(str).tolist(),
        "processed_texts": texts["processed_text"].astype(str).tolist(),
        "bow": sparse.load_npz(input_dir / "bow.npz"),
        "bow_normalized": sparse.load_npz(input_dir / "bow_normalized.npz"),
        "tfidf": sparse.load_npz(input_dir / "tfidf.npz"),
        "semantic_embeddings": np.load(input_dir / "semantic_embeddings.npy"),
        "document_embeddings": np.load(input_dir / "document_embeddings.npy"),
        "vocabulary": np.load(input_dir / "vocabulary.npy", allow_pickle=True),
        "count_vectorizer": joblib.load(input_dir / "count_vectorizer.joblib"),
        "tfidf_vectorizer": joblib.load(input_dir / "tfidf_vectorizer.joblib"),
        "svd": joblib.load(input_dir / "svd.joblib"),
        "config": joblib.load(input_dir / "preprocess_config.joblib"),
    }
