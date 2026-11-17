"""Separate original-row and residual-box verification; imports no primary audit.

The frozen inputs and upstream trace are shared data. Objective, score,
support LP, recession and variance-power calculations are written separately.
"""
from pathlib import Path
from fractions import Fraction as F
import argparse,hashlib,json
import numpy as np
from scipy.optimize import linprog
from scipy.special import expit

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify_support(d,lam,reported):
    X,Y,G,U,H=[d[k] for k in ['X','Y','Yhat','X_unlabeled','Yhat_unlabeled']];m,N=len(Y),len(H)
    A=np.column_stack([X.T/m,U.T/N]);low=np.r_[-Y+lam*G,-lam*H];high=np.r_[1-lam-Y+lam*G,lam*(1-H)];span=np.maximum(high-low,0)
    raw=linprog(np.zeros(len(low)),A_eq=A,b_eq=np.zeros(X.shape[1]),bounds=list(zip(low,high)),method='highs-ipm')
    contracted=linprog(np.zeros(len(low)),A_eq=A,b_eq=np.zeros(X.shape[1]),bounds=list(zip(low+1e-6*span,high-1e-6*span)),method='highs-ipm')
    expected='outside' if raw.status==2 else 'interior' if contracted.status==0 else 'boundary_or_near_boundary'
    assert raw.status in {0,2} and contracted.status in {0,2}
    assert expected==reported['state'],(expected,reported)
    z=np.vstack([X] if lam==0 else [U] if lam==1 else [X,U]);assert np.linalg.matrix_rank(z)==reported['rank']==X.shape[1]
    rec=dict(state=expected,residual_box_status=int(raw.status),contracted_residual_status=int(contracted.status))
    if expected=='outside':
        v=np.array(reported['direction']);r=(np.maximum(U@v,0)-H*(U@v)).mean()*lam+(np.maximum(X@v,0)-Y*(X@v)).mean()-lam*(np.maximum(X@v,0)-G*(X@v)).mean()
        assert r < -1e-9 and abs(r-reported['recession'])<1e-12
        rec['recession']=float(r)
    return rec

def original_score(d,lam,beta):
    X,Y,G,U,H=[d[k] for k in ['X','Y','Yhat','X_unlabeled','Yhat_unlabeled']]
    p=expit(X@beta);q=expit(U@beta)
    # Three empirical terms in the original PPI objective, no weighted support code.
    return np.mean(X*(p-Y)[:,None],axis=0)+lam*np.mean(U*(q-H)[:,None],axis=0)-lam*np.mean(X*(p-G)[:,None],axis=0)

def main(a):
    root=Path(__file__).resolve().parents[1];out=a.out;out.mkdir(parents=True,exist_ok=True)
    plan=json.loads((root/'plan.json').read_text());manifest=json.loads((root/'frozen_input_manifest.json').read_text());rows=json.loads((root/'results/api_records.json').read_text());env=json.loads((root/'results/environment.json').read_text())
    assert sha(root/'plan.json')==manifest['plan_sha256']==env['plan_sha256']
    assert env['run_source_sha256']==sha(root/'analysis/run_official_api.py')
    assert len(rows)==plan['planned_API_calls']==68
    inputs={}
    for c in manifest['cases']:
        path=a.input_dir/(c['name']+'.npz');assert sha(path)==c['sha256']
        with np.load(path,allow_pickle=False) as f:inputs[c['name']]={k:f[k].copy() for k in f.files}
    score_errors=[];LPs=[];power_errors=[];singular=[];analytic=[]
    for i,r in enumerate(rows):
        d=inputs[r['case']];lam=r['effective_power']
        if r['initial_unit_support'] is not None:LPs.append(dict(record=i,stage='initial',**verify_support(d,1.,r['initial_unit_support'])))
        if lam is not None:
            LPs.append(dict(record=i,stage='effective',**verify_support(d,lam,r['support'])))
            if r['return_finite_array']:
                b=np.array(r['beta']);s=float(np.max(abs(original_score(d,lam,b))));score_errors.append(abs(s-r['original_score_max']));assert abs(s-r['original_score_max'])<2e-12
                assert r['numerical_stationary_at_1e_6']==(s<1e-6)
        if r['requested_power'] is None:
            b=np.array(r['optimizer_calls'][0]['x']);X,Y,G,U,H=[d[k] for k in ['X','Y','Yhat','X_unlabeled','Yhat_unlabeled']];m,N=len(Y),len(H)
            p,q=expit(X@b),expit(U@b);design=np.vstack([X,U]);prob=np.r_[p,q]
            hessian=(design.T*(prob*(1-prob)))@design/(m+N)
            if r['exception']:
                assert r['exception']['type']=='LinAlgError' and not r['selected_powers'] and len(r['optimizer_calls'])==1
                assert np.linalg.matrix_rank(hessian)<X.shape[1]
                singular.append(dict(case=r['case'],hessian_rank=int(np.linalg.matrix_rank(hessian)),hessian_max_abs=float(np.max(abs(hessian)))))
            else:
                inverse=np.linalg.inv(hessian);sg=X*(p-Y)[:,None]@inverse;sp=X*(p-G)[:,None]@inverse;su=U*(q-H)[:,None]@inverse
                covariance=np.mean(np.sum((sg-sg.mean(0))*(sp-sp.mean(0)),axis=1));variance=np.var(np.vstack([sp,su]),axis=0,ddof=1).sum()
                chosen=float(np.clip(covariance/((1+m/N)*variance),0,1));power_errors.append(abs(chosen-lam));assert abs(chosen-lam)<2e-10
                assert len(r['optimizer_calls'])==2 and len(r['selected_powers'])==1
        if r['kind']=='analytic':
            l=F(str(r['requested_power']));limit=(F(11)-20*l)/80
            # Exact underlying rational family; machine inputs are rounded.
            moment=np.mean(d['X']*(d['Y']-float(l)*d['Yhat'])[:,None],axis=0)+float(l)*np.mean(d['X_unlabeled']*d['Yhat_unlabeled'][:,None],axis=0)
            assert max(abs(moment-np.array([.5,29/80])))<2e-16
            state='interior' if limit>0 else 'boundary_or_near_boundary' if limit==0 else 'outside';assert r['support']['state']==state
            analytic.append(dict(case=r['case'],exact_positive_slope_recession=str(limit),state=state,optimizer_success=r['optimizer_calls'][-1]['success'],returned_score_max=r['original_score_max']))
    assert max(score_errors)<2e-12 and max(power_errors)<2e-10
    result=dict(state='passed',API_records_checked=68,returned_coefficients_checked=len(score_errors),original_residual_box_LPs=len(LPs),all_power_selections_independently_reconstructed=len(power_errors),initial_hessian_singularities_independently_verified=len(singular),max_original_score_discrepancy=max(score_errors),max_selected_power_discrepancy=max(power_errors),analytical_cases=analytic,singular_initializations=singular,shared_with_primary=['frozen inputs','official API trace'],primary_objective_support_or_tuning_imported=False,upstream_package_imported_by_verifier=False,raw_input_files_shipped=False,source_hashes={str(f.relative_to(root)):sha(f) for f in [root/'plan.json',root/'frozen_input_manifest.json',root/'analysis/run_official_api.py',Path(__file__),root/'results/api_records.json',root/'results/environment.json',root/'results/status.json']})
    (out/'residual_LP_checks.json').write_text(json.dumps(LPs,indent=2)+'\n');(out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['source_hashes','analytical_cases','singular_initializations']}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
