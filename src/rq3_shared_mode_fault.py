"""One lightweight candidate: marginalize a shared mode/fault posterior.

Normal Gaussian references stay frozen. Fault covariance inflation is a working
likelihood, not an identified physical law or a calibrated failure probability.
"""
import numpy as np
from scipy.special import logsumexp


def joint_filter(observations, records, mode_model, config, *, kappa=None, fault_memory=True):
    """Causal 3 modes x (no fault or 3 locations); no truth or label arguments."""
    x=np.asarray(observations,float);r=np.asarray(records)
    if x.ndim!=3 or x.shape[-1]!=3 or r.shape!=x.shape[:2] or not np.isfinite(x).all():
        raise ValueError('finite [batch,time,3] observations and matching records required')
    if x.shape[1]<1 or not np.isin(r,[0,1,2]).all():raise ValueError('invalid times/records')
    if kappa is None:kappa=config['variance_inflation_kappa']
    if not np.isfinite(kappa) or kappa<0:raise ValueError('nonnegative kappa required')
    mu=np.asarray(mode_model['mu']);cov=np.asarray(mode_model['cov'])
    center=np.asarray(mode_model['center']);scale=np.asarray(mode_model['scale'])
    A=np.asarray(mode_model['transition']);initial=np.asarray(mode_model['initial'])
    f0=np.asarray(config['fault_initial']);B=np.asarray(config['fault_transition'])
    for prob in (A,B):
        if (prob<0).any() or not np.allclose(prob.sum(axis=1),1):raise ValueError('transition is not stochastic')
    if f0.shape!=(4,) or (f0<0).any() or not np.isclose(f0.sum(),1):raise ValueError('invalid fault prior')
    if not fault_memory:B=np.tile(f0,(4,1))
    inflation=[]
    for z in range(3):
        for f in range(4):
            c=cov[z].copy()
            if f:c[f-1,f-1]+=kappa*kappa*cov[z,f-1,f-1]
            inflation.append(c)
    covariance=np.asarray(inflation);sign,logdet=np.linalg.slogdet(covariance)
    if (sign<=0).any() or (scale<=0).any():raise ValueError('positive covariance/scales required')
    precision=np.linalg.inv(covariance);means=np.repeat(mu,4,axis=0)
    v=(x-center)/scale;delta=v[:,:,None,:]-means[None,None,:,:]
    ll=-.5*(np.einsum('ntki,kij,ntkj->ntk',delta,precision,delta)+logdet+3*np.log(2*np.pi))
    transitions=np.kron(A,B);prior=np.outer(initial,f0).ravel()
    if not np.isclose(prior.sum(),1) or (prior<0).any():raise ValueError('invalid mode initial')
    q=np.broadcast_to(prior,(len(x),12)).copy();out=np.empty((len(x),x.shape[1],3,4))
    record_p=config['record_correct_likelihood']
    if not 0<record_p<1:raise ValueError('record likelihood must be interior')
    modes=np.repeat(np.arange(3),4)
    for t in range(x.shape[1]):
        pred=q if t==0 else q@transitions
        emission=np.where(modes[None,:]==r[:,t,None],record_p,(1-record_p)/2)
        logprior=np.full_like(pred,-np.inf);np.log(pred,out=logprior,where=pred>0)
        logq=logprior+np.log(emission)+ll[:,t]
        q=np.exp(logq-logsumexp(logq,axis=1,keepdims=True));out[:,t]=q.reshape(len(x),3,4)
    return {'joint':out,'mode':out.sum(-1),'fault':out.sum(-2)}
