# HMRNet research code

HMRNet: A Hybrid Multi-Representation Network with Cross-Representation Attention and Adaptive Fusion for Stable Neural Topic Modeling.

This release includes the original verified final checkpoint and restores the original preprocessing, feature persistence, topic evaluation, stability evaluation, baseline, robustness and interpretability modules recovered from the author's project. The original `src/` files are unchanged. The model file is identical to the earlier release. Historical reference results are kept separately from newly generated outputs.

## Load the included historical checkpoint

```bash
python scripts/inspect_checkpoint.py
python scripts/inspect_checkpoint.py --output results/new_runs/checkpoint_inspection
```

The included file `checkpoints/HMRNet-Stage8-v2-CV-GT070.pt` has SHA-256 `a359a0406a60686059b16ea39b13e7fce41a5cc821ce09fbafa66a1fabf13278`. It was loaded using the original source; its topic-word matrix is 20 x 12,000 and its saved training history has 34 epochs (maximum configured epochs: 60). No training is required to inspect its topics.

For inference on the original corpus, supply the separately stored original feature bundle:
```bash
python scripts/inspect_checkpoint.py --features /path/to/original/features --output results/new_runs/original_checkpoint_inference
```

The author-provided runtime report confirms full-corpus inference on 23,370 documents and agreement with archived document-topic output within maximum absolute difference 8.940696716308594e-7. This full-corpus check ran in the author environment; local packaging independently checked the checkpoint hash, loading and topic-word outputs.

## What is included
- `src/`: all eight recovered original Python modules, including the original seed implementation and feature loader.
- Root notebooks `00`–`07`: original setup and experiment procedures.
- `notebooks/`: previously recovered Stage8v2 calibration, calibrated ablation, downstream/automatic interpretability, duplicate sensitivity and public-baseline procedures.
- `configs/final.json`: archived calibrated final configuration.
- `results/reference/`: selected archived article results.
- `scripts/run_final_pipeline.py`: new portable orchestration using the original feature/model/evaluation implementations; supports the final model and calibrated CRA/AF ablations.
- `tests/smoke_recovered.py`: artificial-data functional test.
- `docs/original_recovery_hashes.json`: original export hashes; `docs/validation.json`: validation scope and limits.

## Install
Use Python 3.10 or later. Install PyTorch for your CPU/CUDA environment, then:
```bash
pip install -r requirements.txt
```
For the optional CombinedTM/BERTopic/Top2Vec original baseline notebooks:
```bash
pip install -r requirements-baselines.txt
```
Those optional external baselines and SentenceTransformer model downloading were not exercised in the recovery tests. Dependency pins in historical notebooks are retained as original experiment instructions; they are not a guarantee of cross-version compatibility.

## Check the installation and archived results
```bash
python tests/smoke_recovered.py
python scripts/check_reference.py
```
The functional test uses artificial documents and supplied synthetic 384-dimensional embeddings, so it needs no pretrained model download. It tests the original feature save/load, two training epochs, inference, fusion weights, checkpoint reload equivalence, coherence, topic-permutation matching and four baseline implementations. Artificial-data metrics are not article results.

## Use the original fixed feature bundle
The restored loader requires these exact original filenames in one directory:
```text
bow.npz
bow_normalized.npz
tfidf.npz
semantic_embeddings.npy
document_embeddings.npy
vocabulary.npy
texts.parquet
count_vectorizer.joblib
tfidf_vectorizer.joblib
svd.joblib
preprocess_config.joblib
```
All rows and vocabulary columns must preserve the original alignment. Only load serialized features and checkpoints from a trusted project.

Train and evaluate the calibrated final model over ten seeds:
```bash
python scripts/run_final_pipeline.py --features /path/to/original/features --output results/new_runs/final --seeds 1 2 3 4 5 6 7 8 9 10
```
`--output` must be a new directory. Outputs include per-seed checkpoints, topic distributions, topic words, configuration, history, metrics, aggregate stability, package versions and feature hashes. CPU/GPU is selected automatically; `--device cpu` overrides it. Full-corpus runs need substantial RAM because the unchanged original model densifies the sparse BoW matrix.

## Regenerate features from authorized text data
```bash
python scripts/run_final_pipeline.py --dataset /path/to/dataset_mbg.csv --text-column full_text --label-column sentimen --date-column created_at --output results/new_runs/regenerated --seeds 1 2 3 4 5 6 7 8 9 10
```
This invokes the original cleaning, CountVectorizer, TF-IDF/SVD and multilingual SentenceTransformer feature builder. It uses the setup notebook's seed-42 sample of at most 25,000 documents and removes documents empty after cleaning. The SentenceTransformer model may require downloading. Generated features are a new run; exact agreement with the historical bundle must be checked using its hashes and ordering. Labels/dates are metadata only in this topic-model runner.

## Calibrated architecture ablations
Use a fresh output directory for each variant:
```bash
python scripts/run_final_pipeline.py --features /path/to/original/features --variant no-cra --output results/new_runs/no_cra
python scripts/run_final_pipeline.py --features /path/to/original/features --variant no-af --output results/new_runs/no_af
python scripts/run_final_pipeline.py --features /path/to/original/features --variant no-cra-no-af --output results/new_runs/no_cra_no_af
```
The final configuration retains CRA and adaptive fusion, sets alignment=0.10, diversity=0.07 and KL=0.10, disables the explicit stability pathway, uses no branch noise, and sets early-stopping patience to 16. Original initial-configuration ablations are a different experiment and remain in the archived notebooks.

## Historical notebooks
Original notebooks retain Google Drive paths. Set their `PROJECT_ROOT` / `PROJECT_CANDIDATES` to the extracted project location before running. The original preparation and baseline notebooks can produce features and the initial checkpoint. Stage8v2 and downstream notebooks require their documented previous-stage outputs. Public-baseline notebooks also require the preceding public-dataset feature stages; the full public HMRNet preparation/training pipeline has not been recovered. The portable runner covers MBG HMRNet feature construction, final-model training/evaluation and calibrated architecture ablations, not all nine article evaluation blocks.

Original metrics are preserved: coherence sanitizes topic terms against the whitespace-tokenized corpus dictionary, and JS stability uses SciPy's default logarithm base. No evaluation formula was changed during packaging.

## Validation and remaining materials
Recovered original file hashes and all eight Python modules were checked. Artificial-data training/inference and checkpoint save/load equivalence passed. LDA, NMF, ProdLDA and NVDM ran on artificial inputs. Archived ten-seed and 45-pair stability records reproduce C_v = 0.7095306832 +/- 0.0257027280 and TS = 0.9894068105 when checked.

The original final checkpoint is included and verified. The fixed feature bundle and the other historical checkpoints remain outside this package. Full-corpus inference was verified in the author environment (see `docs/original_runtime_verification.json`), but original-data retraining and every article experiment have not been reproduced.

For the original final seed-42 checkpoint, the reported C_v is 0.7138034902, TD is 0.985 and TR is 0.0008618036. Original Gensim C_NPMI and UMass returned non-finite values; the report preserves these as null with an explicit serialization note. No metric was replaced by zero or by a different definition. These single-checkpoint metrics must not be presented as the ten-seed article mean. Providing regeneration code or a loadable checkpoint does not establish end-to-end reproduction of all reported experiments.

Raw social-media text, participant identities, other historical checkpoints and full features are not included. Do not publish private feature archives containing corpus text without reviewing the contents and applicable sharing permissions. No open-source license has been assigned; the authors should choose a license before granting reuse rights.

## Updating GitHub
Extract the release ZIP and upload the contents of its project folder, including `checkpoints/`, the restored `src/`, root notebooks, updated `scripts/`, README and dependency files. This archive does not publish or modify the remote repository automatically.
