import glob, json, os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
# Folders are set by environment variables. RES_DIR holds the five unzipped arm folders
# (z42/final10x5_odir42, z7/..., z123/..., z384/..., zr50/...); OUT_DIR receives the outputs.
R=os.environ.get('RES_DIR','results_final/'); OUT=os.environ.get('OUT_DIR','analysis_out/')
os.makedirs(OUT,exist_ok=True)
ARMS={'odir42':R+'z42/final10x5_odir42','odir7':R+'z7/final10x5_odir7','odir123':R+'z123/final10x5_odir123',
      'eff384':R+'z384/final10x5_eff384','resnet50':R+'zr50/final10x5_resnet50'}
SEEDS=[101,202,303,404,505]; NF=10
info=pd.read_csv(ARMS['resnet50']+'/folds.csv',dtype={'group_id':str}).set_index('filename')
files=list(info.index); fidx={f:i for i,f in enumerate(files)}
y=info['label'].values; fold=info['fold'].values; src=info['source'].values; grp=info['group_id'].values
N=len(files)
def load(arm,kind):
    P=np.full((len(SEEDS),N),np.nan)
    for si,s in enumerate(SEEDS):
        for k in range(NF):
            d=pd.read_csv(f"{ARMS[arm]}/fold{k}_seed{s}/{kind}_probs.csv")
            idx=[fidx[f] for f in d.filename]
            P[si,idx]=d.prob.values
    return P
H={a:load(a,'heldout') for a in ARMS}
assert all(not np.isnan(v).any() for v in H.values())
ENS={a:H[a].mean(0) for a in ARMS}
ODIR=['odir42','odir7','odir123']
FINAL=np.mean([ENS[a] for a in ODIR+['eff384']],axis=0)        # 20 models, equal weight per model
ODIRAVG=np.mean([ENS[a] for a in ODIR],axis=0)
def auc(p,m=None):
    m=slice(None) if m is None else m
    return roc_auc_score(y[m],p[m])
def fold_aucs(p): return np.array([roc_auc_score(y[fold==k],p[fold==k]) for k in range(NF)])
rng=np.random.RandomState(2026)
ug=pd.unique(grp); gidx=pd.Series(np.arange(N)).groupby(grp).apply(list).to_dict()
BOOT=[]
for b in range(1000):
    pick=rng.choice(len(ug),len(ug),replace=True)
    ix=np.concatenate([gidx[ug[i]] for i in pick])
    if len(set(y[ix]))<2: continue
    BOOT.append(ix)
print('bootstrap resamples',len(BOOT))
def ci(f):
    v=np.array([f(ix) for ix in BOOT]); return np.percentile(v,[2.5,97.5])
def pooled(name,p):
    a=auc(p); lo,hi=ci(lambda ix: roc_auc_score(y[ix],p[ix]))
    pr=average_precision_score(y,p); plo,phi=ci(lambda ix: average_precision_score(y[ix],p[ix]))
    fa=fold_aucs(p)
    return dict(name=name,pooled=a,lo=lo,hi=hi,pr=pr,pr_lo=plo,pr_hi=phi,fold_mean=fa.mean(),fold_sd=fa.std(ddof=1))
rows=[pooled(a,ENS[a]) for a in ARMS]
rows+= [pooled('ODIR avg (3 arms, 15 models)',ODIRAVG), pooled('FINAL (3 ODIR + eff384, 20 models)',FINAL)]
tab=pd.DataFrame(rows); pd.set_option('display.width',250); pd.set_option('display.max_columns',20)
print(tab.round(4).to_string(index=False))
base=ENS['resnet50']; bf=fold_aucs(base)
def paired(name,p):
    pf=fold_aucs(p); d=pf-bf
    dd=np.array([roc_auc_score(y[ix],p[ix])-roc_auc_score(y[ix],base[ix]) for ix in BOOT])
    est=auc(p)-auc(base)
    return dict(comparison=name+' minus resnet50',pooled_diff=est,lo=np.percentile(dd,2.5),hi=np.percentile(dd,97.5),
                fold_mean_diff=d.mean(),fold_sd=d.std(ddof=1),folds_won=int((d>0).sum()),
                share_boot_pos=float((dd>0).mean()))
pc=pd.DataFrame([paired(a,ENS[a]) for a in ['odir42','odir7','odir123','eff384']]+
                [paired('ODIR avg (3 arms)',ODIRAVG),paired('FINAL (20 models)',FINAL)])
print(pc.round(4).to_string(index=False))
single={}
for a in ARMS:
    single[a]=float(np.mean([fold_aucs(H[a][s]) for s in range(5)]))
    print(f'{a}: mean single-seed fold AUC {single[a]:.4f}')
rows=[]
for s in ['ORIGA','REFUGE_train','REFUGE_val','G1020']:
    m=(src==s)
    for nm,p in [('FINAL',FINAL),('resnet50',base)]:
        a=roc_auc_score(y[m],p[m])
        gs=[ix[m[ix]] for ix in BOOT]; v=np.array([roc_auc_score(y[g],p[g]) for g in gs if len(set(y[g]))==2])
        rows.append(dict(source=s,model=nm,n=int(m.sum()),pos=int(y[m].sum()),auc=a,lo=np.percentile(v,2.5),hi=np.percentile(v,97.5)))
m=np.isin(src,['REFUGE_train','REFUGE_val'])
for nm,p in [('FINAL',FINAL),('resnet50',base)]:
    a=roc_auc_score(y[m],p[m]); v=np.array([roc_auc_score(y[ix[m[ix]]],p[ix[m[ix]]]) for ix in BOOT if len(set(y[ix[m[ix]]]))==2])
    rows.append(dict(source='REFUGE (all)',model=nm,n=int(m.sum()),pos=int(y[m].sum()),auc=a,lo=np.percentile(v,2.5),hi=np.percentile(v,97.5)))
ps=pd.DataFrame(rows); print(ps.round(4).to_string(index=False))
def inner(arm,k):
    out=None
    for s in SEEDS:
        d=pd.read_csv(f"{ARMS[arm]}/fold{k}_seed{s}/innerval_probs.csv").set_index('filename')
        out=d if out is None else out.assign(prob=out.prob+d.loc[out.index,'prob'])
    out=out.assign(prob=out.prob/len(SEEDS)); return out
def inner_ens(arms,k):
    ds=[inner(a,k) for a in arms]
    base_=ds[0]
    for d in ds[1:]: assert set(d.index)==set(base_.index)
    return base_.assign(prob=np.mean([d.loc[base_.index,'prob'].values for d in ds],axis=0))
def op_table(arms,heldout_p,label):
    res={}
    for tgt in ['youden',0.7,0.8,0.9]:
        TP=FP=TN=FN=0
        for k in range(NF):
            iv=inner_ens(arms,k); yi=iv.label.values; pi=iv.prob.values
            if tgt=='youden':
                ths=np.unique(pi); best=None
                for t in ths:
                    sens=((pi>=t)&(yi==1)).sum()/(yi==1).sum(); spec=((pi<t)&(yi==0)).sum()/(yi==0).sum()
                    j=sens+spec-1
                    if best is None or j>best[0]: best=(j,t)
                t=best[1]
            else:
                t=np.quantile(pi[yi==0],tgt)
            mm=fold==k; pk=heldout_p[mm]; yk=y[mm]
            TP+=((pk>=t)&(yk==1)).sum(); FP+=((pk>=t)&(yk==0)).sum(); TN+=((pk<t)&(yk==0)).sum(); FN+=((pk<t)&(yk==1)).sum()
        res[str(tgt)]=dict(sens=TP/(TP+FN),spec=TN/(TN+FP),prec=TP/(TP+FP),acc=(TP+TN)/(TP+TN+FP+FN))
    print(label); print(pd.DataFrame(res).T.round(3).to_string())
    return res
OPF=op_table(ODIR+['eff384'],FINAL,'FINAL ensemble: operating points')
OPB=op_table(['resnet50'],base,'resnet50 baseline: operating points')
json.dump(dict(table=tab.to_dict('records'),paired=pc.to_dict('records'),per_source=ps.to_dict('records'),op_final=OPF,op_base=OPB,single=single),
          open(OUT+'final_numbers.json','w'),indent=1,default=float)
np.savez(OUT+'probs.npz',FINAL=FINAL,base=base,ODIRAVG=ODIRAVG,y=y,fold=fold,src=src,**{a:ENS[a] for a in ARMS})
