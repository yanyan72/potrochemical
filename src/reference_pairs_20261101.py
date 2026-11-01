"""Descriptive paired contrasts from existing five-seed E1b; no new experiments."""
import csv,json,statistics,argparse
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('results/paper_closeout/20261101'));a=parser.parse_args();root=a.output;root.mkdir(parents=True,exist_ok=True)
source=root if (root/'e1b_metrics_by_seed.csv').exists() else Path('results/logistics_e1b/logistics_e1b_20260918_v1')
prefix='e1b_' if source==root else ''
rows=list(csv.DictReader((source/(prefix+'metrics_by_seed.csv')).open()));summary=list(csv.DictReader((source/(prefix+'summary.csv')).open()));out=[]
for a,b,scenario,metric in [('pooled_marginal','single_marginal','clean','clean_channel_fpr'),('regime_marginal','pooled_marginal','clean','clean_channel_fpr'),('regime_marginal','pooled_marginal','observed','recall'),('regime_marginal','pooled_marginal','observed','f1'),('regime_marginal','regime_conditional','observed','bystander_fpr')]:
 for regime in ['all','warehouse','transit_smooth','transit_rough']:
  vals=[]
  for seed in ['42','43','44','45','46']:
   get=lambda method:float(next(r for r in rows if r['seed']==seed and r['method']==method and r['scenario']==scenario and r['regime']==regime)[metric])
   vals.append(get(a)-get(b))
  for method in [a,b]:
   expected=float(next(r for r in summary if r['method']==method and r['scenario']==scenario and r['regime']==regime)[metric+'_mean'])
   actual=statistics.mean(float(r[metric]) for r in rows if r['method']==method and r['scenario']==scenario and r['regime']==regime);assert abs(expected-actual)<1e-12
  out.append(dict(comparison=a+'-'+b,scenario=scenario,regime=regime,metric=metric,mean=statistics.mean(vals),sd=statistics.stdev(vals),seed_differences=vals,evidence='posthoc descriptive reanalysis of original validation; no confirmatory p value'))
(root/'reference_pairs.json').write_text(json.dumps(out,indent=2)+'\n')
for r in out:
 if r['regime']=='all':print(json.dumps(r))
