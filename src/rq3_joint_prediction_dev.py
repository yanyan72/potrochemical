"""Fixed 21-fit development comparison of scoring and reference-based fusion."""
from __future__ import annotations
import argparse,csv,hashlib,itertools,json,math,platform,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import rq3_uncertainty_a0 as a0
from rq3_shared_mode_fault_development import score_raw

class Predictor(nn.Module):
    """Reuse the original single-layer LSTM and mean-hidden readout."""
    def __init__(self,inputs:int=18,hidden:int=32)->None:
        super().__init__();self.lstm=nn.LSTM(inputs,hidden,batch_first=True);self.head=nn.Linear(hidden,1)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        """Map a visible prefix to a scalar standardized target."""
        h,_=self.lstm(x);return self.head(h.mean(1)).squeeze(-1)

def ah(x:np.ndarray)->str:
    """Hash dtype, shape, and values for fairness checks."""
    return hashlib.sha256(str((x.shape,x.dtype.str)).encode()+x.tobytes()).hexdigest()

def events(ids:np.ndarray,r:np.ndarray,old:list,cfg:dict,seed:int,split:str)->list:
    """Retain A0 events and extend its balanced design without new latent data."""
    out=[dict(e) for e in old];assert list(ids[:len(old)])==[e['sequence_id'] for e in old]
    base=list(itertools.product(range(3),(-1,1),(-6,18)));remaining=len(ids)-len(out)
    table=(base*math.ceil(remaining/len(base)))[:remaining]
    rng=np.random.default_rng(np.random.SeedSequence([cfg['event_seed'],seed,0 if split=='train' else 1]))
    for i,k in enumerate(rng.permutation(remaining),start=len(out)):
        ch,sign,offset=table[k];departure=int(np.flatnonzero(r[i,1:]!=r[i,:-1])[0]+1);s=departure+offset
        if not 0<=s<s+12<=r.shape[1]:raise ValueError('event outside visible prefix')
        out.append(dict(sequence_id=str(ids[i]),channel=ch,sign=sign,departure=departure,offset=offset,start=s,end=s+12))
    return out

def score_bundle(x:np.ndarray,r:np.ndarray,ev:list,scales:np.ndarray,old:dict,new:dict,candidate:dict,scorecfg:dict,viewcfg:dict)->dict:
    """Build common observations, scores and references; labels never enter."""
    xs=[];rs=[];ss=[];olds=[];abl=[];qs=[];centers=[];acenters=[];masks=[]
    m=new['modes'];physical=np.asarray(m['mu'])*m['scale']+m['center']
    for view in viewcfg['views']:
        rec,kind=view.split('_');xv,rv,mask=a0.paired_view(x,r,ev,scales,kind,0 if rec=='correct' else 12,viewcfg)
        raw,_,p=score_raw(xv,rv,m,candidate,scorecfg,'joint_full');ratio=raw/new['thresholds']['joint_full']
        ra,_,pa=score_raw(xv,rv,m,candidate,scorecfg,'no_fault_hypothesis');ra=ra/new['thresholds']['no_fault_hypothesis']
        ro,_=a0.legacy_score(xv,rv,old['legacy'])
        xs.append(xv);rs.append(rv);masks.append(mask);ss.append(np.exp(-.5*ratio**2));olds.append(np.exp(-.5*ro**2));abl.append(np.exp(-.5*ra**2))
        qs.append(p['mode']);centers.append(p['mode']@physical);acenters.append(pa['mode']@physical)
    return {k:np.stack(v) for k,v in dict(x=xs,r=rs,mask=masks,S=ss,oldS=olds,ablationS=abl,q=qs,center=centers,ablation_center=acenters).items()}

def draw_random(ids:np.ndarray,train_ids:np.ndarray,bank:np.ndarray,seed:int,control:int)->tuple:
    """Use train trajectories only, excluding every view of the receiver ID."""
    nv,n=bank.shape[:2];lookup={str(v):i for i,v in enumerate(train_ids)};indices=[]
    for sid in ids:
        value=int.from_bytes(hashlib.sha256(f'{control}:{seed}:{sid}'.encode()).digest()[:8],'big');own=lookup.get(str(sid))
        if own is None:j=value%(nv*n);indices.append((j//n,j%n))
        else:
            j=value%(nv*(n-1));b=j%(n-1);indices.append((j//(n-1),b+(b>=own)))
    ix=np.asarray(indices);return bank[ix[:,0],ix[:,1]].copy(),ix

def base_features(bundle:dict)->np.ndarray:
    """Fifteen common visible channels with shared candidate mode probabilities."""
    x=bundle['x'];return np.concatenate([x,np.zeros_like(x),np.eye(3)[bundle['r']],np.ones_like(x),bundle['q']],axis=-1)

def scaler_fit(base:np.ndarray)->dict:
    """Fit a single common scaler from training views only."""
    a=base.mean((0,1,2));b=base.std((0,1,2));constant=b<1e-12;b[constant]=1
    return dict(center=a.tolist(),scale=b.tolist(),constant=constant.tolist())

def inputs_for(bundle:dict,arm:str,scaler:dict,random:np.ndarray)->np.ndarray:
    """Concatenate scores or apply fixed gate/center correction in common units."""
    z=(base_features(bundle)-scaler['center'])/scaler['scale'];z[...,np.asarray(scaler['constant'])]=0
    extra=np.zeros_like(bundle['S']);s=bundle['S'];ref=bundle['center']
    if arm=='OT':extra=bundle['oldS']
    elif arm=='NT':extra=s
    elif arm in ['ZG','NG','RG','AG']:
        if arm=='RG':s=np.broadcast_to(random,bundle['S'].shape)
        if arm=='AG':s=bundle['ablationS'];ref=bundle['ablation_center']
        center=np.zeros_like(ref) if arm=='ZG' else (ref-np.asarray(scaler['center'])[:3])/np.asarray(scaler['scale'])[:3]
        z[...,:3]=s*z[...,:3]+(1-s)*center
    elif arm!='N':raise ValueError(arm)
    result=np.concatenate([z,extra],axis=-1).astype(np.float32)
    if not np.isfinite(result).all():raise ValueError('nonfinite input')
    return result

def predict(model:Predictor,x:np.ndarray,mean:float,scale:float,batch:int)->np.ndarray:
    """CPU batched inference in original target units."""
    model.eval()
    with torch.no_grad():p=torch.cat([model(torch.from_numpy(x[i:i+batch])) for i in range(0,len(x),batch)]).numpy()
    return p.astype(float)*scale+mean

def fit(x:np.ndarray,y:np.ndarray,vx:np.ndarray,vy:np.ndarray,schedule:np.ndarray,cfg:dict)->tuple:
    """Same initialization, order and fixed updates; select earliest best epoch."""
    torch.manual_seed(cfg['model_seed']);model=Predictor(x.shape[-1],cfg['hidden_size'])
    initial=hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.state_dict().values())).hexdigest()
    mean,scale=float(y.mean()),float(y.std());assert scale>0
    tx=torch.from_numpy(x);ty=torch.from_numpy(((y-mean)/scale).astype(np.float32))
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    shuffle=np.random.default_rng(cfg['shuffle_seed']);hist=[];orders=[];best=float('inf');state=None;bestepoch=None
    flat=vx.reshape(-1,*vx.shape[2:]);target=np.tile(vy,len(vx));started=time.monotonic()
    for epoch in range(cfg['epochs']):
        order=shuffle.permutation(len(y));orders.append(order);model.train();loss_sum=0.
        for start in range(0,len(y),cfg['batch_size']):
            ix=order[start:start+cfg['batch_size']];optimizer.zero_grad(set_to_none=True)
            loss=((model(tx[schedule[epoch,ix],ix])-ty[ix])**2).mean()
            if not torch.isfinite(loss):raise ValueError('nonfinite loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip'],error_if_nonfinite=True);optimizer.step();loss_sum+=float(loss.detach())*len(ix)
        pred=predict(model,flat,mean,scale,cfg['batch_size']);mae=float(np.abs(pred-target).mean())
        if mae<best:best=mae;bestepoch=epoch+1;state={k:p.detach().clone() for k,p in model.state_dict().items()}
        hist.append(dict(epoch=epoch+1,train_mse=loss_sum/len(y),val_suite_mae=mae))
    model.load_state_dict(state);pred=predict(model,flat,mean,scale,cfg['batch_size']).reshape(len(vx),len(vy))
    for h in hist:h['selected']=h['epoch']==bestepoch
    params=dict(target_center=mean,target_scale=scale,selected_epoch=bestepoch,val_mae=best,model_seed=cfg['model_seed'],shuffle_seed=cfg['shuffle_seed'],initial_sha256=initial,shuffle_sha256=ah(np.stack(orders)),schedule_sha256=ah(schedule),optimizer_updates=cfg['epochs']*math.ceil(len(y)/cfg['batch_size']),parameter_count=sum(p.numel() for p in model.parameters()),input_channels=x.shape[-1],elapsed_seconds=time.monotonic()-started)
    return {k:p.numpy() for k,p in state.items()},params,hist,pred

def labels(data:Path,seed:int,split:str,ids:np.ndarray,target:str)->np.ndarray:
    """Load ID-aligned train or val target only."""
    if split not in ['train','val']:raise ValueError(split)
    rows=list(csv.DictReader((data/f'seed_{seed}/{split}/targets.csv').open()))
    assert [r['sequence_id'] for r in rows]==list(ids)
    return np.asarray([float(r[target]) for r in rows])

def seal(root:Path)->None:
    """Write an immutable content manifest after a stage completes."""
    if (root/'manifest.json').exists():raise FileExistsError(root)
    a0.dump(root/'manifest.json',{str(p.relative_to(root)):a0.digest(p) for p in sorted(root.rglob('*')) if p.is_file()})

def run_seed(data:Path,a0root:Path,scorer:Path,cfg:dict,out:Path,commit:str,seed:int)->None:
    """Run seven models and persist each immediately, keeping input sources intact."""
    torch.set_num_threads(cfg['threads']);torch.use_deterministic_algorithms(True)
    if seed not in cfg['seeds']:raise ValueError(seed)
    if a0.digest(data/'manifest.json')!=cfg['source_dataset_manifest']:raise ValueError('dataset identity')
    paths=[data/'manifest.json',a0root/'config.json',scorer/'config.json',scorer/'candidate_config.json',a0root/f'seed_{seed}/model.json',scorer/f'seed_{seed}/model.json']
    paths += [data/f'seed_{seed}/{s}/{p}' for s in ['train','val'] for p in ['inputs.npz','targets.csv']]
    paths += [data/f'seed_{seed}/fault_scales.json']+[a0root/f'seed_{seed}/{s}_events.json' for s in ['train','val']]
    inputhash={str(p):a0.digest(p) for p in paths};dm=json.loads((data/'manifest.json').read_text())
    for p in paths:
        if p.is_relative_to(data) and p!=data/'manifest.json':assert inputhash[str(p)]==dm[str(p.relative_to(data))]
    for r in [a0root,scorer]:
        manifest=json.loads((r/'manifest.json').read_text())
        for p in paths:
            if p.is_relative_to(r):assert inputhash[str(p)]==manifest[str(p.relative_to(r))]
    if not out.exists():
        out.mkdir(parents=True);a0.dump(out/'config.json',cfg);a0.dump(out/'identity.json',dict(implementation_commit=commit,python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,source='synthetic_rebuilt_reused',test_used=False,new_latent_processes=0,source_sha256={p.name:a0.digest(p) for p in [Path(__file__),Path(a0.__file__),Path(__file__).with_name('rq3_shared_mode_fault.py'),Path(__file__).with_name('rq3_shared_mode_fault_development.py')]}))
    else:
        assert json.loads((out/'config.json').read_text())==cfg;assert json.loads((out/'identity.json').read_text())['implementation_commit']==commit
    sd=out/f'seed_{seed}';sd.mkdir(exist_ok=False);a0.dump(sd/'source_hashes.json',inputhash)
    old=json.loads((a0root/f'seed_{seed}/model.json').read_text());new=json.loads((scorer/f'seed_{seed}/model.json').read_text())
    candidate=json.loads((scorer/'candidate_config.json').read_text());scorecfg=json.loads((scorer/'config.json').read_text());viewcfg=json.loads((a0root/'config.json').read_text())
    scales=np.asarray(json.loads((data/f'seed_{seed}/fault_scales.json').read_text())['scales']);bundles={};ys={};ids={};evs={};r0={}
    for split in ['train','val']:
        v=a0.npz(data/f'seed_{seed}/{split}/inputs.npz');ids[split]=v['sequence_id'];r0[split]=v['recorded_regime'];assert len(ids[split])==cfg['counts'][split]
        evs[split]=events(ids[split],r0[split],json.loads((a0root/f'seed_{seed}/{split}_events.json').read_text()),cfg,seed,split)
        a0.dump(sd/f'{split}_events.json',evs[split]);ys[split]=labels(data,seed,split,ids[split],cfg['target'])
        bundles[split]=score_bundle(v['observations'],r0[split],evs[split],scales,old,new,candidate,scorecfg,viewcfg)
    assert not set(ids['train'])&set(ids['val'])
    tb,vb=bundles['train'],bundles['val'];scaler=scaler_fit(base_features(tb));a0.dump(sd/'scaler.json',scaler)
    random={};indices={}
    for split in ['train','val']:random[split],indices[split]=draw_random(ids[split],ids['train'],tb['S'],seed,cfg['random_control_seed'])
    assert np.all(indices['train'][:,1]!=np.arange(len(ids['train'])))
    schedule=np.random.default_rng(np.random.SeedSequence([cfg['schedule_seed'],seed])).integers(0,len(cfg['views']),size=(cfg['epochs'],len(ys['train'])),dtype=np.int64)
    np.savez_compressed(sd/'schedule_and_random.npz',schedule=schedule,train_random=indices['train'],val_random=indices['val'],train_ids=ids['train'],val_ids=ids['val'])
    np.savez_compressed(sd/'evaluation_metadata.npz',y=ys['val'],sequence_id=ids['val'],end_regime=r0['val'][:,-1],channel=np.array([e['channel'] for e in evs['val']]),offset=np.array([e['offset'] for e in evs['val']]))
    fairness=[];predictions=[];first=None;common={}
    for arm in cfg['arms']:
        x=inputs_for(tb,arm,scaler,random['train']);vx=inputs_for(vb,arm,scaler,random['val'])
        commonhash=(ah(x[...,3:15]),ah(vx[...,3:15]))
        if common:assert commonhash==common['all']
        common['all']=commonhash
        if arm in ['N','OT','NT']:
            if 'raw' in common:assert (ah(x[...,:15]),ah(vx[...,:15]))==common['raw']
            common['raw']=(ah(x[...,:15]),ah(vx[...,:15]))
        state,params,hist,pred=fit(x,ys['train'],vx,ys['val'],schedule,cfg)
        signature=tuple(params[k] for k in ['initial_sha256','shuffle_sha256','schedule_sha256','optimizer_updates','parameter_count'])
        if first is not None:assert signature==first
        first=signature
        np.savez_compressed(sd/f'{arm}_model.npz',**state);a0.dump(sd/f'{arm}_parameters.json',params);a0.csv_write(sd/f'{arm}_history.csv',hist);np.savez_compressed(sd/f'{arm}_predictions.npz',pred=pred)
        replay=Predictor(cfg['input_channels'],cfg['hidden_size']);replay.load_state_dict({k:torch.from_numpy(v) for k,v in a0.npz(sd/f'{arm}_model.npz').items()})
        rp=predict(replay,vx.reshape(-1,*vx.shape[2:]),params['target_center'],params['target_scale'],cfg['batch_size']).reshape(pred.shape);assert np.array_equal(rp,pred)
        fairness.append(dict(arm=arm,train_features_sha256=ah(x),val_features_sha256=ah(vx),shared_other_features_sha256=commonhash,replay_error=0.,**params));predictions.append(pred)
        print(json.dumps(dict(seed=seed,arm=arm,selected_epoch=params['selected_epoch'],val_mae=params['val_mae'],seconds=params['elapsed_seconds'])),flush=True)
    np.savez_compressed(sd/'predictions.npz',pred=np.stack(predictions),arms=cfg['arms'],views=cfg['views'],y=ys['val'],sequence_id=ids['val'])
    # Batch-level paired signal attenuation: no hidden clean process used.
    diag=[]
    for vi,view in enumerate(cfg['views']):
        clean_vi=cfg['views'].index(view.split('_')[0]+'_clean')
        for i,e in enumerate(evs['val']):
            s,end,ch=e['start'],e['end'],e['channel'];mask=vb['mask'][vi,i];w=vb['S'][vi,i];x=vb['x'][vi,i];clean=vb['x'][clean_vi,i];c=vb['center'][vi,i];corrected=w*x+(1-w)*c
            diag.append(dict(seed=seed,view=view,sequence_id=str(ids['val'][i]),channel=ch,offset=e['offset'],event_score=float(w[s:end,ch].mean()),normal_downweight=float((1-w)[~mask].mean()),observed_vs_paired_clean=float(np.abs(x-clean).mean()),corrected_vs_paired_clean=float(np.abs(corrected-clean).mean()),temperature_shift=float((corrected[:,0]-x[:,0]).mean())))
    a0.csv_write(sd/'fusion_diagnostics.csv',diag);a0.dump(sd/'fairness.json',fairness)
    for p,h in inputhash.items():assert a0.digest(p)==h
    a0.dump(sd/'completion.json',dict(models=len(predictions),source_unchanged=True,complete=True));seal(sd)

def summarize(out:Path)->None:
    """Create all strata and paired seed comparisons from archived predictions."""
    cfg=json.loads((out/'config.json').read_text());rows=[];paired=[];suite=[]
    for seed in cfg['seeds']:
        sd=out/f'seed_{seed}';p=a0.npz(sd/'predictions.npz');meta=a0.npz(sd/'evaluation_metadata.npz');errors=np.abs(p['pred']-p['y'][None,None,:])
        for ai,arm in enumerate(cfg['arms']):
            suite.append(dict(seed=seed,arm=arm,mae=float(errors[ai].mean())))
            for vi,view in enumerate(cfg['views']):
                strata={'all':np.ones(len(p['y']),bool)}
                strata.update({f'channel_{j}':meta['channel']==j for j in range(3)});strata.update({f'offset_{o}':meta['offset']==o for o in [-6,18]});strata.update({f'end_regime_{j}':meta['end_regime']==j for j in range(3)})
                for name,mask in strata.items():
                    if mask.any():rows.append(dict(seed=seed,arm=arm,view=view,stratum=name,n=int(mask.sum()),mae=float(errors[ai,vi,mask].mean()),rmse=float(np.sqrt(np.square(p['pred'][ai,vi,mask]-p['y'][mask]).mean()))))
        for arm,base in cfg['comparisons']:
            delta=errors[cfg['arms'].index(arm)]-errors[cfg['arms'].index(base)]
            paired.append(dict(seed=seed,comparison=f'{arm}-{base}',view='suite',mae_difference=float(delta.mean())))
            for vi,view in enumerate(cfg['views']):paired.append(dict(seed=seed,comparison=f'{arm}-{base}',view=view,mae_difference=float(delta[vi].mean())))
    agg=[]
    for a,b in cfg['comparisons']:
        for view in ['suite']+cfg['views']:
            values=[r['mae_difference'] for r in paired if r['comparison']==f'{a}-{b}' and r['view']==view]
            agg.append(dict(comparison=f'{a}-{b}',view=view,mean=float(np.mean(values)),sd=float(np.std(values,ddof=1)),negative_seeds=int(np.sum(np.array(values)<0)),outer_seeds=len(values)))
    a0.csv_write(out/'metrics_by_seed.csv',rows);a0.csv_write(out/'suite_by_seed.csv',suite);a0.csv_write(out/'paired_by_seed.csv',paired);a0.csv_write(out/'paired_summary.csv',agg)
    screen=[]
    for s in cfg['seeds']:
        get=lambda arm:next(r['mae'] for r in suite if r['seed']==s and r['arm']==arm)
        clean=lambda arm:next(r['mae'] for r in rows if r['seed']==s and r['arm']==arm and r['view']=='correct_clean' and r['stratum']=='all')
        screen.append(dict(seed=s,NG_minus_N=get('NG')-get('N'),NG_minus_RG=get('NG')-get('RG'),relative_gain_vs_N=(get('N')-get('NG'))/get('N'),clean_relative_change=(clean('NG')-clean('N'))/clean('N')))
    a0.dump(out/'development_screen_descriptive.json',dict(rows=screen,role='Descriptive original thresholds, scientific follow-up based on trends and mechanism, not automatic rejection',new_test_evidence=False,models=21));seal(out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','summarize']);p.add_argument('--data',type=Path);p.add_argument('--a0-root',type=Path);p.add_argument('--scorer-root',type=Path);p.add_argument('--config',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--commit');p.add_argument('--seed',type=int);a=p.parse_args()
    if a.command=='summarize':summarize(a.output)
    else:run_seed(a.data,a.a0_root,a.scorer_root,json.loads(a.config.read_text()),a.output,a.commit,a.seed)
