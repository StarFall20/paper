"""Call the pinned, unmodified official package on predeclared frozen inputs.

Module-level delegates observe optimizer results and selected powers. They pass
all arguments and return values through unchanged. There is no source rewrite.
"""
from pathlib import Path
import argparse,hashlib,importlib,json,sys,warnings
import numpy as np
from scipy.optimize import linprog
from scipy.special import expit

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def clean(x):
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    return x
def write(p,x):p.write_text(json.dumps(clean(x),indent=2,allow_nan=False)+'\n')

def loss_terms(data,power):
    X,Y,g,U,h=[data[k] for k in ['X','Y','Yhat','X_unlabeled','Yhat_unlabeled']]
    n,N=len(Y),len(h);xx=np.vstack([X,U]);w=np.r_[np.full(n,(1-power)/n),np.full(N,power/N)]
    b=X.T@(Y-power*g)/n+power*U.T@h/N
    def val(beta):return float(w@np.logaddexp(0,xx@beta)-b@beta)
    def grad(beta):return xx.T@(w*expit(xx@beta))-b
    return xx,w,b,val,grad

def support(data,power):
    xx,w,b,val,grad=loss_terms(data,power);keep=w>0;z=xx[keep];ww=w[keep];A=(ww[:,None]*z).T;n=len(z);d=z.shape[1]
    rank=int(np.linalg.matrix_rank(z))
    # Maximize common interior probability margin using the original rows.
    inequalities=np.zeros((2*n,n+1));inequalities[:n,:n]=-np.eye(n);inequalities[n:,:n]=np.eye(n);inequalities[:,-1]=1
    r=linprog(np.r_[np.zeros(n),-1.],A_ub=inequalities,b_ub=np.r_[np.zeros(n),np.ones(n)],A_eq=np.column_stack([A,np.zeros(d)]),b_eq=b,bounds=[(0,1)]*n+[(0,.5)],method='highs')
    result=dict(rank=rank,dimensions=d,margin_LP_status=int(r.status),margin=None if r.x is None else float(r.x[-1]),state='unresolved')
    if rank<d:result['state']='rank_deficient'
    elif r.status==0:result['state']='interior' if r.x[-1]>1e-8 else 'boundary_or_near_boundary'
    elif r.status==2:
        result['state']='outside'
        # Direct recession minimization, with a compact direction bound.
        rr=linprog(np.r_[-b,ww],A_ub=np.column_stack([z,-np.eye(n)]),b_ub=np.zeros(n),bounds=[(-1,1)]*d+[(0,None)]*n,method='highs')
        assert rr.status==0 and rr.fun < -1e-9,(rr.status,rr.fun)
        direction=rr.x[:d];recession=float(w@np.maximum(xx@direction,0)-b@direction)
        result.update(direction=direction,recession=recession,large_ray_secant=(val(2e7*direction)-val(1e7*direction))/1e7)
        assert recession < -1e-9 and abs(result['large_ray_secant']-recession)<1e-6
    return result

def main(a):
    out=a.out;out.mkdir(parents=True,exist_ok=True)
    parent=Path(__file__).resolve().parents[1];plan=json.loads((parent/'plan.json').read_text());manifest=json.loads((parent/'frozen_input_manifest.json').read_text())
    assert sha(parent/'plan.json')==manifest['plan_sha256'] and sha(a.input_dir/'cases.json')==manifest['case_list_sha256']
    source_manifest=json.loads((parent/'upstream_source_manifest.json').read_text())
    for f in source_manifest['files']:assert sha(a.upstream/f['path'])==f['sha256'],f['path']
    for c in manifest['cases']:assert sha(a.input_dir/(c['name']+'.npz'))==c['sha256']
    sys.path.insert(0,str(a.upstream.resolve()));up=importlib.import_module('ppi_py.ppi')
    assert Path(up.__file__).resolve()==(a.upstream/'ppi_py/ppi.py').resolve()
    original_minimize=up.minimize;original_lambda=up._calc_lam_glm
    calls=[];powers=[]
    def observe_minimize(*args,**kwargs):
        result=original_minimize(*args,**kwargs)
        calls.append(dict(method=kwargs.get('method'),options=dict(kwargs.get('options',{})),success=bool(result.success),status=int(result.status),message=str(result.message),nit=int(result.nit),nfev=int(result.nfev),fun=float(result.fun),x=result.x.copy(),jac=result.jac.copy()))
        return result
    def observe_lambda(*args,**kwargs):
        result=original_lambda(*args,**kwargs);powers.append(float(result));return result
    up.minimize=observe_minimize;up._calc_lam_glm=observe_lambda
    rows=[];cases=json.loads((a.input_dir/'cases.json').read_text())
    try:
        for c in cases:
            with np.load(a.input_dir/c['input_file'],allow_pickle=False) as file:data={k:file[k].copy() for k in file.files}
            for requested in c['powers']:
                calls.clear();powers.clear();record=dict(case=c['name'],kind=c['kind'],requested_power=requested,exception=None,return_finite_array=False)
                beta=None
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    try:
                        beta=up.ppi_logistic_pointestimate(**data,lam=requested,optimizer_options=dict(plan['optimizer_options']))
                        record.update(return_finite_array=bool(np.isfinite(beta).all()),beta=beta,beta_norm=float(np.linalg.norm(beta)))
                    except Exception as e:record['exception']=dict(type=type(e).__name__,message=str(e))
                    record['warnings']=[dict(category=w.category.__name__,message=str(w.message)) for w in caught]
                record['optimizer_calls']=list(calls);record['selected_powers']=list(powers)
                final_power=float(requested) if requested is not None else powers[-1] if powers else None
                record['effective_power']=final_power
                # Even an API exception retains the unit-power initialization state.
                record['initial_unit_support']=support(data,1.) if requested is None else None
                if final_power is not None:
                    record['support']=support(data,final_power)
                    if beta is not None:
                        _,_,_,val,grad=loss_terms(data,final_power)
                        record.update(stable_original_objective=val(beta),original_score_max=float(np.max(abs(grad(beta)))),numerical_stationary_at_1e_6=bool(np.max(abs(grad(beta)))<1e-6))
                rows.append(record);write(out/'api_records.json',rows)
                print(json.dumps(dict(completed=len(rows),case=c['name'],power=requested,returned=record['return_finite_array'],exception=record['exception']['type'] if record['exception'] else None,state=record.get('support',{}).get('state'),score=record.get('original_score_max'))),flush=True)
    finally:up.minimize=original_minimize;up._calc_lam_glm=original_lambda
    assert len(rows)==plan['planned_API_calls']==68
    for f in source_manifest['files']:assert sha(a.upstream/f['path'])==f['sha256']
    from importlib.metadata import version
    import platform
    versions={k:version(k) for k in ['numpy','scipy','scikit-learn','numba','llvmlite','statsmodels','pandas','tqdm','patsy','threadpoolctl']}
    write(out/'environment.json',dict(python=sys.version,platform=platform.platform(),versions=versions,official_source_unchanged_before_after=True,normal_package_import=True,observational_wrappers_only=True,run_source_sha256=sha(Path(__file__)),plan_sha256=sha(parent/'plan.json')))
    write(out/'status.json',dict(state='complete',API_calls=len(rows),returned_finite_arrays=sum(r['return_finite_array'] for r in rows),exceptions=sum(r['exception'] is not None for r in rows),original_source_unchanged=True,no_unplanned_cases_added=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--upstream',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
