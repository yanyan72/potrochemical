"""Descriptive paired contrasts from existing five-seed E1b; no new experiments."""
import csv,json,statistics
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'results';rows=list(csv.DictReader((root/'e1b_metrics_by_seed.csv').open()));summary=list(csv.DictReader((root/'e1b_summary.csv').open()));out=[]
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
