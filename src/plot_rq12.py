"""Render archived descriptive figures; no experiment or score changes."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('results');OUT=ROOT/'analysis_v1'
d=pd.read_csv(ROOT/'factorial_validation_v1/metrics_by_seed.csv',dtype={'channels':str});d=d[(d.stratum=='all')&(d.regime==-1)]
# A fixed diagnostic panel: all pairs, both lags, strong bias. No favorable-pair selection.
fig,axes=plt.subplots(1,3,figsize=(12,3.7));labels=[]
for ax,metric,title in zip(axes,['point_recall','bystander_fpr','mode_error'],['Point recall','Normal-channel false alarms','Regime classification error']):
 sub=d[(d.channels.isin(['01','02','12']))&(d.magnitude==4)]
 groups=list(sub.groupby(['lag','channels']));xs=np.arange(len(groups))
 for ai,arm in enumerate(['joint12','factorial24']):
  means=[g[g.arm==arm][metric].mean()*100 for _,g in groups];sds=[g[g.arm==arm][metric].std()*100 for _,g in groups]
  ax.errorbar(xs+(ai-.5)*.16,means,yerr=sds,fmt='o-',label=arm,capsize=3)
 labels=[f'{ch}/d{lag}' for (lag,ch),_ in groups];ax.set_xticks(xs,labels,rotation=35);ax.set_title(title);ax.set_ylabel('% (mean ± seed SD)');ax.grid(alpha=.2)
axes[0].legend();fig.tight_layout();fig.savefig(OUT/'compound_fault_validation.png',dpi=170);fig.savefig(OUT/'compound_fault_validation.pdf');plt.close(fig)
rq=pd.read_csv(ROOT/'rq1_confirmation_v1/metrics_by_seed.csv');sel=rq[(rq.channel==-1)&(rq.regime=='all')]
fig,axes=plt.subplots(1,2,figsize=(8,3.5))
for ax,mag,met,title in [(axes[0],0,'clean_fpr','Clean observation false alarms'),(axes[1],2,'recall','2 SD bias recall')]:
 g=sel[sel.magnitude==mag].groupby('method')[met];means=g.mean()*100;sds=g.std()*100;order=['single_marginal','pooled_marginal','regime_marginal'];ax.bar(range(3),means[order],yerr=sds[order],capsize=4);ax.set_xticks(range(3),['Single','Pooled','Regime']);ax.set_title(title);ax.set_ylabel('% (mean ± seed SD)')
fig.tight_layout();fig.savefig(OUT/'rq1_confirmation.png',dpi=170);fig.savefig(OUT/'rq1_confirmation.pdf');plt.close(fig)
