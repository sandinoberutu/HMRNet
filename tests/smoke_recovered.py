"""Functional tests on artificial data, not a reproduction of article results."""
from pathlib import Path
import sys,tempfile,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.preprocessing import PreprocessConfig,TextPreprocessor,build_feature_bundle,save_feature_bundle,load_feature_bundle,set_global_seed
from src.hmrnet import HMRNet,HMRNetConfig
from src.topic_metrics import evaluate_topics,topic_diversity,topic_redundancy
from src.stability_metrics import hungarian_topic_stability,evaluate_stability_runs
from src.baseline_models import fit_lda,fit_nmf,fit_prodlda,fit_nvdm
from src.perturbation import subsample_dataframe,random_token_deletion
import src.interpretability

def main():
    torch.set_num_threads(2);set_global_seed(42)
    themes=[
        'school student teacher classroom education lesson reading science knowledge learning',
        'energy electric charging battery solar voltage power current grid renewable',
        'health doctor patient hospital medicine treatment clinic care therapy recovery',
        'market finance business revenue budget trading company growth investment profit']
    raw=[themes[i%4]+(' research study evidence analysis' if i%2 else ' practice method system design') for i in range(80)]
    cfg=PreprocessConfig(min_df=1,max_df=1.,max_features=100,ngram_range=(1,1),document_dim=16)
    pre=TextPreprocessor(cfg);processed=pre.transform(raw)
    assert pre.clean('@name Hello #Grid https://example.org')=='hello grid'
    rng=np.random.default_rng(42);semantic=rng.normal(size=(80,384)).astype('float32')
    semantic/=np.linalg.norm(semantic,axis=1,keepdims=True)
    b=build_feature_bundle(raw,processed,cfg,semantic_embeddings=semantic)
    reports=[]
    with tempfile.TemporaryDirectory() as tmp:
        temp=Path(tmp);save_feature_bundle(b,temp/'features');loaded=load_feature_bundle(temp/'features')
        assert np.array_equal(loaded['vocabulary'],b['vocabulary'])
        assert np.allclose(loaded['bow'].toarray(),b['bow'].toarray())
        assert loaded['processed_texts']==processed
        reports.append('Original feature build/save/load round trip')
        final=json.loads((ROOT/'configs/final.json').read_text())|{'epochs':2,'batch_size':16,'device':'cpu'}
        model=HMRNet(HMRNetConfig(**final)).fit(b['bow'],semantic,b['document_embeddings'],b['vocabulary'],verbose=False)
        theta=model.transform(b['bow'],semantic,b['document_embeddings']);beta=model.get_topic_word_distribution()
        assert theta.shape==(80,20) and np.isfinite(theta).all() and (theta>=0).all()
        assert np.allclose(theta.sum(1),1,atol=1e-5)
        assert len(model.history)==2
        assert all(row['train_stability']==0 for row in model.history)
        weights=model.get_fusion_weights(b['bow'],semantic,b['document_embeddings'])
        assert weights.shape==(80,3) and np.allclose(weights.sum(1),1,atol=1e-5)
        model.save(temp/'model.pt');restored=HMRNet.load(temp/'model.pt',device='cpu')
        theta2=restored.transform(b['bow'],semantic,b['document_embeddings'])
        assert np.allclose(theta,theta2,atol=1e-6)
        assert restored.config.diversity_weight==.07 and restored.config.use_stability_loss is False
        reports.append('Final-config model: 2-epoch train, inference, adaptive weights, save/load equivalence')
        score,_,_=hungarian_topic_stability(beta,beta[::-1]);assert np.isclose(score,1.)
        pairs,summary=evaluate_stability_runs({1:beta,2:beta[::-1],3:beta})
        assert len(pairs)==3 and np.isclose(summary['Mean_TS'],1.)
        reports.append('Hungarian matching invariant to topic permutation')
        metric=evaluate_topics('SMOKE_ARTIFICIAL',restored.get_topics(10),processed)
        assert all(np.isfinite(metric[k]) for k in ['C_v','C_NPMI','UMass','Topic_Diversity','Topic_Redundancy'])
        assert topic_diversity([['a','b'],['b','c']])==.75
        assert np.isclose(topic_redundancy([['a','b'],['b','c']]),1/3)
        reports.append('Original Gensim coherence, diversity and redundancy evaluation')
        for fn,matrix_name in [(fit_lda,'bow'),(fit_nmf,'tfidf')]:
            kwargs={'bow':b['bow']} if matrix_name=='bow' else {'tfidf':b['tfidf']}
            if fn==fit_lda:kwargs['max_iter']=2
            result=fn(**kwargs,vocabulary=b['vocabulary'],n_topics=4)
            assert result.document_topic.shape==(80,4) and len(result.topics)==4
        for fn in [fit_prodlda,fit_nvdm]:
            result=fn(bow=b['bow'],vocabulary=b['vocabulary'],n_topics=4,epochs=1,batch_size=16,device='cpu')
            assert result.document_topic.shape==(80,4) and np.isfinite(result.document_topic).all()
        reports.append('LDA, NMF, ProdLDA and NVDM fit/inference on artificial features')
    report={'status':'PASS','scope':'Recovered original modules exercised on artificial corpus; no pretrained SentenceTransformer download or article-data retraining',
            'checks':reports,'torch_version':torch.__version__,'article_reproduction':'NOT TESTED: original fixed features and trained checkpoints not supplied'}
    print(json.dumps(report,indent=2));return report

if __name__=='__main__':main()
