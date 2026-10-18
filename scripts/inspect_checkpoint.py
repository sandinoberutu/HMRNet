"""Load the included historical checkpoint; optionally infer on original features."""
from pathlib import Path
from dataclasses import asdict
import argparse,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
EXPECTED_SHA256='a359a0406a60686059b16ea39b13e7fce41a5cc821ce09fbafa66a1fabf13278'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/HMRNet-Stage8-v2-CV-GT070.pt')
    p.add_argument('--features',type=Path,help='Original trusted feature bundle, with bow.npz and texts.parquet')
    p.add_argument('--output',type=Path,help='New directory for topics, beta and optional theta')
    p.add_argument('--device',default='cpu')
    a=p.parse_args()
    import numpy as np
    from src.hmrnet import HMRNet
    if a.output and a.output.exists():raise FileExistsError('Output directory already exists')
    if a.features and not a.output:p.error('--features requires a new --output directory')
    digest=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    if a.checkpoint.resolve()==(ROOT/'checkpoints/HMRNet-Stage8-v2-CV-GT070.pt').resolve() and digest!=EXPECTED_SHA256:
        raise ValueError('Included checkpoint hash does not match verified historical file')
    model=HMRNet.load(a.checkpoint,device=a.device)
    beta=model.get_topic_word_distribution()
    if not np.isfinite(beta).all() or (beta<0).any() or not np.allclose(beta.sum(1),1,atol=1e-5):raise ValueError('Invalid topic-word distribution')
    result={'status':'PASS','checkpoint_sha256':digest,'matches_verified_checkpoint':digest==EXPECTED_SHA256,
            'topic_word_shape':list(beta.shape),'vocabulary_size':len(model.vocabulary),'history_epochs':len(model.history),'config':asdict(model.config)}
    theta=None
    if a.features:
        from src.preprocessing import load_feature_bundle
        b=load_feature_bundle(a.features);bow=b['bow'];sem=b['semantic_embeddings'];doc=b['document_embeddings'];n=bow.shape[0]
        if not (n==len(sem)==len(doc)==len(b['raw_texts'])==len(b['processed_texts'])):raise ValueError('Feature row counts differ')
        if list(map(str,b['vocabulary']))!=list(map(str,model.vocabulary)):raise ValueError('Vocabulary order mismatch')
        theta=np.vstack([model.transform(bow[i:i+256],sem[i:i+256],doc[i:i+256]) for i in range(0,n,256)])
        if not np.isfinite(theta).all() or (theta<0).any() or not np.allclose(theta.sum(1),1,atol=1e-4):raise ValueError('Invalid document-topic output')
        result['document_topic_shape']=list(theta.shape)
    if a.output:
        a.output.mkdir(parents=True)
        (a.output/'topics.json').write_text(json.dumps(model.get_topics(10),ensure_ascii=False,indent=2))
        np.save(a.output/'topic_word.npy',beta)
        if theta is not None:np.save(a.output/'document_topic.npy',theta)
        (a.output/'checkpoint_check.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
