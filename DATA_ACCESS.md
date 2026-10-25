# Data Access and Research Artifacts

**Project:** HMRNet — A Hybrid Multi-Representation Network with Cross-Representation Attention and Adaptive Fusion for Stable Neural Topic Modeling  
**Repository:** https://github.com/sandinoberutu/HMRNet

## 1. Primary Dataset: MBG Public-Discourse Corpus

The primary study uses **23,370 preprocessed public X/Twitter posts** concerning Indonesia's Free Nutritious Meal (*Makan Bergizi Gratis*, MBG) policy. Posts were collected for the period **August 1, 2024–July 16, 2026**, using 20 predefined MBG-related search terms. The initial collection contained **29,269 records** before preprocessing and filtering.

The corpus is a sample of observable public discourse on X/Twitter; it is **not statistically representative** of the Indonesian population. Platform-user composition, search terms, collection periods, and repeated posting may affect the observed topic distribution. Repeated or synchronized text patterns were audited, but **no accounts were classified as bots** on that basis.

### Public access restrictions

The repository **does not distribute** the original X/Twitter posts, associated user identifiers, or potentially identifiable source content. Individual human-evaluation records and evaluator identities are also excluded from the public release. These restrictions reflect privacy considerations, applicable platform conditions, and institutional requirements.

Researchers may contact the **corresponding author using the contact details in the associated manuscript** to inquire about:

- the data-collection and preprocessing protocols;
- metadata definitions, search-query design, and evaluation procedures; or
- whether any form of restricted access to research data is permissible.

**Submitting a request does not guarantee access.** Any potential sharing must be evaluated against applicable platform terms, privacy protections, and institutional rules. No raw corpus download link is offered by this repository.

## 2. Preprocessing and Feature Construction

The manuscript describes lowercasing, URL and user-mention removal, hashtag-symbol normalization, punctuation and whitespace normalization, and token filtering. The main HMRNet input representations are:

| Representation | Configuration |
|---|---|
| Lexical | Bag-of-words, 12,000-term vocabulary |
| Contextual semantic | 384-dimensional `paraphrase-multilingual-MiniLM-L12-v2` sentence embeddings |
| Corpus-level | 256-dimensional TF-IDF/Truncated SVD features |

See [`src/preprocessing.py`](src/preprocessing.py), [`configs/final.json`](configs/final.json), and the available setup/experiment notebooks for implementation details. Since the primary corpus and its complete feature bundle are not distributed publicly, a third party cannot reconstruct identical MBG inputs solely from this repository.

## 3. Public Benchmark Datasets

The manuscript additionally evaluates HMRNet on **20 Newsgroups** (18,728 documents after the reported preprocessing) and **StackOverflow** (19,965 documents after the reported preprocessing). Obtain these datasets from their respective original distribution sources; they are not redistributed as part of the MBG corpus release. See the experiment notebooks for the available preparation and evaluation procedures. The public-benchmark experiments use a fixed **K = 20**; dataset-specific topic-number tuning was not conducted.

## 4. Publicly Available Research Artifacts

At the repository snapshot audited on **October 25, 2026** (`main`, commit `3221626f1c81e81b1cb8f0d3726044523b04f26c`), the public file inventory included:

| Category | Representative paths |
|---|---|
| Core model | [`src/hmrnet.py`](src/hmrnet.py) |
| Preprocessing | [`src/preprocessing.py`](src/preprocessing.py) |
| Evaluation | [`src/topic_metrics.py`](src/topic_metrics.py), [`src/stability_metrics.py`](src/stability_metrics.py), [`src/interpretability.py`](src/interpretability.py) |
| Configuration | [`configs/final.json`](configs/final.json) |
| Dependency specifications | [`requirements.txt`](requirements.txt), [`requirements-baselines.txt`](requirements-baselines.txt) |
| Notebooks | Root-level and [`notebooks/`](notebooks/) experiment notebooks |
| Final checkpoint | [`checkpoints/HMRNet-Stage8-v2-CV-GT070.pt`](checkpoints/HMRNet-Stage8-v2-CV-GT070.pt) |
| Archived results | [`results/reference/`](results/reference/) |
| Verification records | [`docs/original_runtime_verification.json`](docs/original_runtime_verification.json), other records in [`docs/`](docs/) |

This inventory confirms **file presence** on the public default branch; it is not itself evidence that every workflow runs successfully from a clean environment.

## 5. Scope of Reproducibility and Verification

A **separate local verification** of the original HMRNet-Final checkpoint confirmed successful source imports, feature-shape and vocabulary checks, checkpoint loading, and inference over the archived **23,370-document** corpus. The inferred document-topic matrix had shape **23,370 × 20**, and the topic-word matrix had shape **20 × 12,000**. The reproduced document-topic output matched the archived result with a maximum absolute difference of approximately **8.94 × 10⁻⁷**.

**Important limitations:**

- This was **saved-checkpoint inference**, **not** retraining of the model or complete end-to-end reproduction of every published experiment.
- Independently recomputed coherence metrics included non-finite results. Single-checkpoint metrics must **not** be substituted for the manuscript's ten-seed estimates.
- The public-repository audit checked metadata and file presence; it did **not** execute training scripts or verify a clean-install reproduction command.
- Baseline comparisons did not use fully matched random-seed protocols across every model. Performance differences should therefore not be treated as a statistically established ranking across all baselines.
- Topic-number sensitivity on MBG was limited to **K ∈ {10, 15, 20, 25, 30}**; the public datasets used fixed **K = 20**.
- Because the underlying MBG corpus and corresponding fixed features are not public, exact end-to-end reproduction of the primary-corpus results cannot be guaranteed using public artifacts alone.

See [`docs/original_runtime_verification.json`](docs/original_runtime_verification.json) for the recorded local verification. The reference result tables under [`results/reference/`](results/reference/) are **archived reported outputs**, not proof of an independently rerun experiment.

## 6. Human-Evaluation Data

Four voluntary evaluators assessed a subset of ten HMRNet topics through a structured form. Their identities and individual responses are **not publicly released**. The manuscript reports aggregated ratings, inter-rater concordance, and diagnostic-task accuracy. The use of model-derived topic assignments as diagnostic answer keys does not constitute independent validation of the semantic correctness of every document-topic relationship.

## 7. Questions and Responsible Use

For methodological clarification or inquiries about the feasibility of restricted data access, please contact the **corresponding author through the associated manuscript**. Do not attempt to reconstruct, republish, or disclose social-media user identities or human-evaluator records from any research artifact.

**Citation:** If you use the released code or reported reference results, cite the associated HMRNet manuscript and identify the repository version or commit used. Check the repository for its current license and reuse terms; this document does not grant a software or data license.
