"""Audit one fixed binary-choice loss and a released fitted reference.

No downloaded research script is executed. Only numeric NumPy pickle globals
are accepted. This is a convergence audit, not a replication of GAI's claims.
"""
from pathlib import Path
import argparse, io, json, pickle, hashlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits
try:
    from numpy._core import multiarray, numeric
except ImportError:
    from numpy.core import multiarray, numeric


class NumericOnly(pickle.Unpickler):
    def find_class(self, module, name):
        allowed={('numpy','ndarray'):np.ndarray, ('numpy','dtype'):np.dtype}
        for prefix in ['numpy.core.multiarray','numpy._core.multiarray']:
            allowed[(prefix,'_reconstruct')]=multiarray._reconstruct
            allowed[(prefix,'scalar')]=multiarray.scalar
        for prefix in ['numpy.core.numeric','numpy._core.numeric']:
            allowed[(prefix,'_frombuffer')]=numeric._frombuffer
        if (module,name) not in allowed:
            raise ValueError('Unapproved numeric pickle global: '+module+'.'+name)
        return allowed[(module,name)]


def save(path, obj):
    def convert(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=convert,allow_nan=False)+'\n')


def load_data(root):
    manifest=json.loads((root/'source_manifest.json').read_text())
    for f in manifest['files']:
        raw=(root/f['path']).read_bytes()
        assert len(raw)==f['size']
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==f['git_blob_sha1']
    folder=root/'GAI_Conjoint_Study/label_z/data'
    dataset=NumericOnly(io.BytesIO((folder/'train_cot-gpt-4o_11_1200.pkl').read_bytes())).load()
    reference=NumericOnly(io.BytesIO((folder/'ground_truth.pkl').read_bytes())).load()['params'].astype(float)
    assert isinstance(dataset,list) and len(dataset)==1
    data=dataset[0];menus=np.asarray(data['X'],float)
    y=np.asarray(data['y'],int);z=np.asarray(data['y_aug'],float)
    assert menus.shape==(4605,2,12) and y.shape==(4605,) and z.shape==(1200,) and reference.shape==(11,)
    assert np.isfinite(menus).all() and np.isfinite(z).all() and np.isfinite(reference).all()
    assert set(np.unique(y))=={0,1} and set(np.unique(z))=={0.,1.}
    assert np.max(abs(menus[:,1,0]-menus[:,0,0]))<1e-12
    x=menus[:,1,1:]-menus[:,0,1:]
    assert np.linalg.matrix_rank(x)==11 and np.linalg.matrix_rank(x[:1200])==11
    check={'source':manifest,'human_outcome_rows':len(y),'feature_menus':menus.shape,
        'AI_outcome_rows':len(z),'common_AI_prefix_rows':1200,
        'full_human_label_counts':np.bincount(y,minlength=2),
        'prefix_human_label_counts':np.bincount(y[:1200],minlength=2),
        'AI_label_counts':np.bincount(z.astype(int),minlength=2),
        'constant_first_column_difference_max':float(np.max(abs(menus[:,1,0]-menus[:,0,0]))),
        'full_feature_rank':int(np.linalg.matrix_rank(x)),
        'block_information':'Five-row blocks are the source-script convention; no respondent IDs are released in this file',
        'raw_research_code_executed':False,'numeric_restricted_reader_used':True,
        'underlying_population_scope':'The source study excludes respondents who never select a vaccine; this is not all original1971 respondents',
        'reference_scope':'The released fitted11-vector, not known structural truth'}
    return x,y,z,reference,check


class Logistic:
    def __init__(self,x,y):self.x=x;self.y=y;self.n=len(y)
    def value(self,b):
        u=self.x@b;p=expit(u)
        return float(np.mean(np.logaddexp(0,u)-self.y*u)), self.x.T@(p-self.y)/self.n
    def hessian(self,b):
        p=expit(self.x@b)
        return self.x.T@((p*(1-p))[:,None]*self.x)/self.n


def adam(loss,steps):
    # Matches the released helper's convention: float32 parameter/buffers,
    # float64 data, ones start and the default Adam moment constants.
    b=np.ones(loss.x.shape[1],dtype=np.float32);m=np.zeros_like(b);v=np.zeros_like(b)
    for step in range(1,steps+1):
        g=loss.value(b.astype(float))[1].astype(np.float32)
        m=np.float32(.9)*m+np.float32(.1)*g
        v=np.float32(.999)*v+np.float32(.001)*g*g
        denom=np.sqrt(v)/np.sqrt(1-.999**step)+1e-8
        b-=np.float32(.01/(1-.9**step))*m/denom
    return b.astype(float)


def converged(loss):
    k=loss.x.shape[1]
    opts=[minimize(loss.value,np.full(k,s),jac=True,method='BFGS',
        options={'gtol':1e-10,'maxiter':3000}) for s in [-1.,0.,1.]]
    b=min(opts,key=lambda x:x.fun).x.copy()
    for _ in range(8):
        val,g=loss.value(b)
        if max(abs(g))<1e-12:break
        delta=np.linalg.solve(loss.hessian(b),g);step=1.
        while loss.value(b-step*delta)[0]>val+1e-13:
            step/=2
            if step<1e-8:raise ValueError('Newton refinement failed')
        b-=step*delta
    val,g=loss.value(b)
    assert max(abs(g))<1e-9
    eig=np.linalg.eigvalsh(loss.hessian(b));assert min(eig)>1e-8
    return b,{'start_losses':[float(o.fun) for o in opts],
        'start_loss_range':float(np.ptp([o.fun for o in opts])),
        'mean_gradient_max':float(max(abs(g))),
        'mean_information_min_eigenvalue':float(min(eig)),
        'mean_information_condition_number':float(max(eig)/min(eig))}


def derivative_check(loss):
    test=np.linspace(-.13,.17,loss.x.shape[1]);eps=1e-5;eye=np.eye(len(test))
    fd=np.array([(loss.value(test+eps*e)[0]-loss.value(test-eps*e)[0])/(2*eps) for e in eye])
    fh=np.column_stack([(loss.value(test+eps*e)[1]-loss.value(test-eps*e)[1])/(2*eps) for e in eye])
    eg=float(max(abs(fd-loss.value(test)[1])));eh=float(np.max(abs(fh-loss.hessian(test))))
    assert max(eg,eh)<1e-6
    return {'gradient_max_error':eg,'hessian_max_error':eh}


def main():
    p=argparse.ArgumentParser();p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    x,y,z,reference,preflight=load_data(a.input_dir);save(a.out/'data_preflight.json',preflight)
    rows=[];fits={};checks={}
    for name,n in [('all_released_human_rows',len(y)),('AI_aligned_prefix',len(z))]:
        loss=Logistic(x[:n],y[:n]);checks[name]=derivative_check(loss)
        optimum,diag=converged(loss);checks[name]['converged_fit']=diag
        estimates={'released_reference':reference,'Adam_200':adam(loss,200),
                   'Adam_4000':adam(loss,4000),'converged_MLE':optimum}
        fits[name]=estimates;best=loss.value(optimum)[0]
        for label,b in estimates.items():
            val,g=loss.value(b);h=loss.hessian(b)
            rows.append({'cohort':name,'human_outcome_rows':n,'estimate':label,
                'mean_negative_loglik':val,'excess_total_negative_loglik':n*(val-best),
                'mean_score_max':float(max(abs(g))),'total_score_max':float(n*max(abs(g))),
                'local_score_information_norm':float(np.sqrt(n*g@np.linalg.solve(h,g))),
                'parameter_L2_from_converged':float(np.linalg.norm(b-optimum)),
                'parameter_max_from_converged':float(max(abs(b-optimum))),
                'parameter_max_from_released_reference':float(max(abs(b-reference)))})
    frame=pd.DataFrame(rows);frame.to_csv(a.out/'optimizer_diagnostics.csv',index=False)
    save(a.out/'optimizer_parameters.json',fits);save(a.out/'numeric_checks.json',checks)
    print(frame.to_string(index=False),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
