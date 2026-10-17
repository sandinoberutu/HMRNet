"""New packaging runner; historical notebooks are retained separately."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--seeds', nargs='+', type=int, default=[42])
    p.add_argument('--output', type=Path, default=ROOT/'results/new_runs')
    args = p.parse_args()
    from src.hmrnet import HMRNet, HMRNetConfig
    bow = sparse.load_npz(args.features/'bow_counts.npz')
    sem = np.load(args.features/'semantic_embeddings.npy', allow_pickle=False)
    doc = np.load(args.features/'document_embeddings.npy', allow_pickle=False)
    vocab = json.loads((args.features/'vocabulary.json').read_text())
    if not (bow.shape[0] == sem.shape[0] == doc.shape[0]):
        raise ValueError('Feature row counts differ')
    if bow.shape[1] != len(vocab) or sem.shape[1] != 384 or doc.shape[1] != 256:
        raise ValueError('Expected matching vocabulary, semantic dimension 384 and document dimension 256')
    if not isinstance(vocab, list) or not all(isinstance(t,str) for t in vocab):
        raise ValueError('Vocabulary must be a JSON list of strings')
    if bow.shape[0] < 4 or np.any(bow.data < 0) or not all(np.isfinite(x).all() for x in [bow.data, sem, doc]):
        raise ValueError('Features must be finite, BoW nonnegative and at least four rows')
    cfg = json.loads((ROOT/'configs/final.json').read_text())
    n_val = min(max(round(bow.shape[0]*0.1),1),bow.shape[0]-1)
    if (bow.shape[0]-n_val) % cfg['batch_size'] == 1:
        raise ValueError('Training split produces a singleton BatchNorm batch; restore original training procedure')
    args.output.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        target = args.output/f'seed_{seed}'
        if target.exists():
            raise FileExistsError(f'Refusing to overwrite {target}')
        target.mkdir()
        model = HMRNet(HMRNetConfig(**(cfg | {'random_state':seed})))
        model.fit(bow,sem,doc,vocab)
        model.save(target/'model.pt')
        np.save(target/'document_topic.npy',model.transform(bow,sem,doc))
        np.save(target/'topic_word.npy',model.get_topic_word_distribution())
        (target/'topics.json').write_text(json.dumps(model.get_topics(),ensure_ascii=False,indent=2))
        (target/'history.json').write_text(json.dumps(model.history,indent=2))

if __name__ == '__main__':
    main()
