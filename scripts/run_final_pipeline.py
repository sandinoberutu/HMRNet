"""Portable runner around recovered original HMRNet modules.
New orchestration only; original source remains unchanged.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    inputs=p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--dataset',type=Path)
    inputs.add_argument('--features',type=Path,help='Original save_feature_bundle directory containing bow.npz and texts.parquet')
    p.add_argument('--text-column',default='full_text')
    p.add_argument('--label-column',default=None)
    p.add_argument('--date-column',default=None)
    p.add_argument('--output',type=Path,required=True,help='New output directory; must not already exist')
    p.add_argument('--seeds',nargs='+',type=int,default=list(range(1,11)))
    p.add_argument('--variant',choices=['final','no-cra','no-af','no-cra-no-af'],default='final')
    p.add_argument('--device',default=None)
    args=p.parse_args()
    import numpy as np
    import pandas as pd
    import torch
    from src.preprocessing import PreprocessConfig,TextPreprocessor,build_feature_bundle,load_text_dataset,save_feature_bundle,load_feature_bundle,set_global_seed
    from src.hmrnet import HMRNet,HMRNetConfig
    from src.topic_metrics import evaluate_topics
    from src.stability_metrics import evaluate_stability_runs
    device=args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if len(set(args.seeds))!=len(args.seeds):raise ValueError('Seeds must be distinct')
    if args.output.exists():raise FileExistsError('Use a new output directory; existing outputs will not be overwritten')
    args.output.mkdir(parents=True)
    if args.dataset:
        set_global_seed(42)
        df,columns=load_text_dataset(args.dataset,text_column=args.text_column,label_column=args.label_column,date_column=args.date_column)
        # Matches original setup: seeded sampling also occurs when the corpus has fewer than 25,000 rows.
        df=df.sample(n=min(25000,len(df)),random_state=42).reset_index(drop=True)
        cfg=PreprocessConfig(text_column=columns['text_column'],label_column=columns['label_column'],date_column=columns['date_column'])
        raw=df[columns['text_column']].astype(str).tolist();processed=TextPreprocessor(cfg).transform(raw)
        keep=np.array([bool(t.strip()) for t in processed]);raw=[t for t,k in zip(raw,keep) if k];processed=[t for t,k in zip(processed,keep) if k]
        if len(processed)<4:raise ValueError('At least four nonempty processed documents are required')
        b=build_feature_bundle(raw,processed,cfg,device=device)
        feature_dir=args.output/'features';save_feature_bundle(b,feature_dir)
        df.loc[keep].reset_index(drop=True).to_parquet(args.output/'dataset_metadata.parquet',index=False)
        (args.output/'resolved_columns.json').write_text(json.dumps(columns,indent=2))
    else:
        feature_dir=args.features;b=load_feature_bundle(feature_dir)
    bow=b['bow'];sem=np.asarray(b['semantic_embeddings']);doc=np.asarray(b['document_embeddings']);vocab=b['vocabulary']
    n=bow.shape[0]
    if not (n==len(sem)==len(doc)==len(b['raw_texts'])==len(b['processed_texts'])):raise ValueError('Feature rows are not aligned')
    if len(vocab)!=bow.shape[1]:raise ValueError('Vocabulary order/width must match original BoW')
    if not all(np.isfinite(x).all() for x in [bow.data,sem,doc]) or (bow.data<0).any():raise ValueError('Invalid feature values')
    config=json.loads((ROOT/'configs/final.json').read_text())
    if args.variant in ['no-cra','no-cra-no-af']:config['use_cra']=False
    if args.variant in ['no-af','no-cra-no-af']:config['fusion_mode']='average'
    config['device']=device
    n_val=min(max(round(n*.1),1),max(n-1,1))
    if n<4 or (n-n_val)%config['batch_size']==1:raise ValueError('Insufficient rows or singleton training BatchNorm batch; use the original valid feature bundle')
    feature_hashes={}
    for f in sorted(feature_dir.iterdir()):
        if f.is_file():
            h=hashlib.sha256()
            with f.open('rb') as stream:
                for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
            feature_hashes[f.name]=h.hexdigest()
    provenance={'feature_hashes':feature_hashes,'seeds':args.seeds,'variant':args.variant,'new_runner':True,'historical_reproduction_verified':False,
                'feature_shapes':{'bow':list(bow.shape),'semantic':list(sem.shape),'document':list(doc.shape)},
                'versions':{name:importlib.metadata.version(name) for name in ['numpy','pandas','torch','scipy','gensim','scikit-learn']}}
    (args.output/'run_provenance.json').write_text(json.dumps(provenance,indent=2))
    metrics=[];betas={}
    for seed in args.seeds:
        run=args.output/f'seed_{seed}';run.mkdir()
        model=HMRNet(HMRNetConfig(**(config|{'random_state':seed})))
        model.fit(bow,sem,doc,vocab)
        model.save(run/'model.pt')
        theta=model.transform(bow,sem,doc);beta=model.get_topic_word_distribution();topics=model.get_topics(10)
        np.save(run/'document_topic.npy',theta);np.save(run/'topic_word.npy',beta)
        (run/'topics.json').write_text(json.dumps(topics,ensure_ascii=False,indent=2))
        (run/'config.json').write_text(json.dumps(asdict(model.config),indent=2))
        (run/'history.json').write_text(json.dumps(model.history,indent=2))
        row=evaluate_topics('HMRNet-'+args.variant,topics,b['processed_texts'],top_n=10)|{'Seed':seed}
        metrics.append(row);betas[seed]=beta
        pd.DataFrame(metrics).to_csv(args.output/'run_metrics.csv',index=False)
        del model
        if device.startswith('cuda'):torch.cuda.empty_cache()
    pairwise,stability=evaluate_stability_runs(betas)
    pairwise.to_csv(args.output/'stability_pairs.csv',index=False)
    summary={'variant':args.variant,'n_seeds':len(args.seeds),**stability}
    frame=pd.DataFrame(metrics)
    for column in ['C_v','C_NPMI','UMass','Topic_Diversity','Topic_Redundancy']:
        summary['Mean_'+column]=float(frame[column].mean());summary['Std_'+column]=float(frame[column].std(ddof=1))
    pd.DataFrame([summary]).to_csv(args.output/'summary.csv',index=False)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
