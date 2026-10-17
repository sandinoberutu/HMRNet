# HMRNet research code

HMRNet: A Hybrid Multi-Representation Network with Cross-Representation Attention and Adaptive Fusion for Stable Neural Topic Modeling.

Recovered research source and archived experiment evidence associated with JCTA_Rev06.docx. This package is a partial recovery, not a verified end-to-end reproduction of every article experiment.

## Contents
- `src/hmrnet.py`: recovered original model implementation, unchanged.
- `configs/final.json`: final calibrated configuration extracted from the saved experiment configuration.
- `notebooks/`: original experiment code and markdown, with outputs cleared for sharing.
- `results/reference/`: selected original aggregate results, not newly generated results.
- `scripts/train_final.py`: newly added standalone training entry point using supplied fixed features.
- `scripts/check_reference.py`: checks saved ten-seed metrics against the article.
- `docs/source_manifest.json`: hashes identifying recovered original files.

## Verified reference results
HMRNet-Final: C_v = 0.7095306832, sample SD = 0.0257027280; TD = 0.9585; TR = 0.0042796635; TS = 0.9894068105.
Final settings: K=20, hidden=256, 4 attention heads, AdamW learning rate 0.002, batch 128, dropout 0.20, weight decay 1e-5, max 60 epochs, patience 16. KL=0.10, alignment=0.10, diversity=0.07. Explicit stability pathway disabled; stability weight and branch noise both zero.

## Setup and reference check
Use Python 3.10 or later. Install PyTorch suitable for your CPU/CUDA environment, then:
```bash
pip install -r requirements.txt
python scripts/check_reference.py
```
Dependency versions other than Gensim are not a historical environment lock.

## Standalone training (new packaging entry point)
Provide the original aligned feature arrays in a directory:
- `bow_counts.npz`: SciPy sparse raw count matrix, N x V.
- `semantic_embeddings.npy`: N x 384.
- `document_embeddings.npy`: N x 256.
- `vocabulary.json`: JSON list of V terms in the exact BoW column order.

```bash
python scripts/train_final.py --features /path/to/features --seeds 1 2 3 4 5 6 7 8 9 10
```
This trains fresh models; it does not recover the historical checkpoint or guarantee identical article scores. Feature identity, row order, original preprocessing, seed helper, environment and evaluation implementation affect reproduction. The original model densifies the BoW matrix, so full-corpus training requires substantial RAM. Training batches of size one are incompatible with its decoder BatchNorm.

## Original notebook prerequisites
The notebooks retain their original Google Drive paths. They target the existing research project `/content/drive/MyDrive/UDB/Riset/2026`. They are archival experiment procedures and are not standalone in this recovered package. Required original modules `src.topic_metrics`, `src.stability_metrics`, and `src.preprocessing.load_feature_bundle`, the feature bundle, and checkpoints must be restored before running them. The public-baseline notebook additionally needs its preceding public-dataset feature stages. Stage8 starts from `HMRNet-Full.pt`; downstream evaluation needs `HMRNet-Stage8-v2-CV-GT070.pt`.

## Recovery limits
The original checkpoint files, fixed feature bundle, complete preprocessing/evaluation source, full public HMRNet training pipeline and human-evaluation analysis source were not recovered. `src/preprocessing.py` is a newly written seed helper only, not the original preprocessing module. It must not be presented as the exact historical seed implementation. Training and inference were not executed in the packaging environment because PyTorch and the original features were unavailable. Only syntax and archived-result integrity were checked.

Raw social-media text, respondent identities, trained checkpoints and labels are not included. No open-source license is assigned by this package; the authors should choose one before granting reuse rights.

## GitHub upload
Extract the ZIP, create a repository, then upload the contents of this directory. Do not upload the ZIP as the only repository file. Keep the recovery limits visible in the README until the missing original materials are supplied.
