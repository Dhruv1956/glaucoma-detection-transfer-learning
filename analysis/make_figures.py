# Makes fig_roc and fig_source from the outputs of analyze_final.py.
# Needs analysis_out/probs.npz (per-image predictions, NOT included in this repository) and
# analysis_out/final_numbers.json; run from the folder that contains analysis_out/ (rename it to final/ or edit the paths).
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
plt.rcParams.update({'font.family':'serif','font.size':8,'axes.linewidth':0.6,'pdf.fonttype':42})
d=np.load('analysis_out/probs.npz',allow_pickle=True); y=d['y']
J=json.load(open('analysis_out/final_numbers.json'))
# ---- ROC
fig,ax=plt.subplots(figsize=(3.4,3.1))
spec=[('resnet50 baseline','base','#6b6b6b','--',1.1),('EfficientNet-B0 384 px','eff384','#0072B2','-.',1.1),
      ('ODIR average (15 models)','ODIRAVG','#D55E00',':',1.4),('Final ensemble (20 models)','FINAL','#000000','-',1.6)]
names={'base':'ResNet50 baseline (5 models)'}
for lab,k,c,ls,lw in spec:
    p=d[k]; fpr,tpr,_=roc_curve(y,p); a=roc_auc_score(y,p)
    ax.plot(fpr,tpr,color=c,ls=ls,lw=lw,label=f"{names.get(k,lab)} ({a:.3f})")
ax.plot([0,1],[0,1],color='#bbbbbb',lw=0.6,ls='-')
ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate')
ax.set_xlim(0,1); ax.set_ylim(0,1.0); ax.set_aspect('equal')
ax.grid(True,color='#e6e6e6',lw=0.4); ax.legend(loc='lower right',fontsize=6.3,frameon=False)
for s in ['top','right']: ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig('figures/fig_roc.pdf'); plt.close(fig)
# ---- per source
rows=[r for r in J['per_source'] if r['source'] in ('ORIGA','REFUGE (all)','G1020')]
order=['ORIGA','REFUGE (all)','G1020']; lab={'ORIGA':'ORIGA','REFUGE (all)':'REFUGE','G1020':'G1020'}
fig,ax=plt.subplots(figsize=(3.4,2.3))
for i,s in enumerate(order):
    for j,(m,c,mk,off) in enumerate([('FINAL','#000000','o',-0.13),('resnet50','#6b6b6b','s',0.13)]):
        r=[x for x in rows if x['source']==s and x['model']==m][0]
        ax.errorbar(r['auc'],i+off,xerr=[[r['auc']-r['lo']],[r['hi']-r['auc']]],fmt=mk,color=c,ms=3.6,lw=0.9,capsize=2,
                    label=('Final ensemble' if m=='FINAL' else 'ResNet50 baseline') if i==0 else None)
        ax.text(r['hi']+0.008,i+off,f"{r['auc']:.3f}",va='center',fontsize=6.5,color=c)
ax.axvline(0.5,color='#bbbbbb',lw=0.7,ls='--'); 
ax.set_yticks(range(3)); ax.set_yticklabels([f"{lab[s]}" for s in order]); ax.invert_yaxis()
ax.set_xlim(0.40,1.03); ax.set_xlabel('Within-source ROC-AUC (95% CI)')
ax.grid(True,axis='x',color='#e6e6e6',lw=0.4); ax.legend(loc='lower center',fontsize=6.3,frameon=False,ncol=2,bbox_to_anchor=(0.5,1.0))
for s in ['top','right']: ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig('figures/fig_source.pdf'); plt.close(fig)
print('ok')
