"""Fixed factorial fault-mask extension; no new fitted likelihood parameters."""
from __future__ import annotations
import itertools
import numpy as np
from scipy.special import logsumexp


def factorial_filter(observations: np.ndarray,records: np.ndarray,model: dict,config: dict) -> dict[str,np.ndarray]:
    """Causal 3-regime x 8-mask filter; independent channel fault Markov chains.

    Channel initial/onset/recovery probabilities inherit .01/.001/.1 from
    the original single-fault configuration. Covariance inflation is additive
    across active bits. This enlarges allowed joint hypotheses, not the score.
    """
    x=np.asarray(observations,float);r=np.asarray(records)
    if x.ndim!=3 or x.shape[-1]!=3 or r.shape!=x.shape[:2] or not np.isfinite(x).all() or not np.isin(r,[0,1,2]).all():raise ValueError('finite 3-channel observations and matching records required')
    bits=np.array(list(itertools.product([0,1],repeat=3)));nf=len(bits)
    f0=np.asarray(config['fault_initial']);old=np.asarray(config['fault_transition']);chains=np.array([[[1-old[0,j+1],old[0,j+1]],[old[j+1,0],1-old[j+1,0]]] for j in range(3)])
    pri=np.prod(np.where(bits,f0[1:],1-f0[1:]),axis=1)
    B=np.ones((nf,nf))
    for j in range(3):B*=chains[j,bits[:,j,None],bits[None,:,j]]
    A=np.asarray(model['transition']);initial=np.asarray(model['initial']);mu=np.asarray(model['mu']);cov=np.asarray(model['cov']);k=config['variance_inflation_kappa']
    if k<0 or not np.allclose(B.sum(-1),1) or not np.allclose(A.sum(-1),1):raise ValueError('invalid transition or inflation')
    cs=np.array([c+np.diag(k*k*np.diag(c)*mask) for c in cov for mask in bits]);sign,ld=np.linalg.slogdet(cs)
    if (sign<=0).any():raise ValueError('positive covariance required')
    v=(x-model['center'])/model['scale'];delta=v[:,:,None,:]-np.repeat(mu,nf,axis=0)[None,None]
    ll=-.5*(np.einsum('ntki,kij,ntkj->ntk',delta,np.linalg.inv(cs),delta)+ld+3*np.log(2*np.pi))
    transition=np.kron(A,B);post=np.broadcast_to(np.outer(initial,pri).ravel(),(len(x),3*nf)).copy();out=np.empty((len(x),x.shape[1],3,nf));modes=np.repeat(np.arange(3),nf);rec_p=config['record_correct_likelihood']
    for t in range(x.shape[1]):
        pred=post if not t else post@transition;record=np.where(modes[None,:]==r[:,t,None],rec_p,(1-rec_p)/2)
        logprior=np.full_like(pred,-np.inf);np.log(pred,out=logprior,where=pred>0)
        logq=logprior+np.log(record)+ll[:,t];post=np.exp(logq-logsumexp(logq,axis=-1,keepdims=True));out[:,t]=post.reshape(len(x),3,nf)
    return {'mode':out.sum(-1),'fault_mask':out.sum(-2),'fault_channel':out.sum(-2)@bits,'joint':out,'masks':bits}
