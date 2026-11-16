"""Independent audit of retained roots, support extremes and probability losses.

Original records are preserved. A probability-coordinate primal and a
direction-coordinate dual check the support endpoints. Coercivity nesting
then covers intermediate powers for the SAME frozen prediction.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.integrate import quad
from scipy.optimize import linprog
from scipy.special import expit,ndtr
from threadpoolctl import threadpool_limits
from public_conjoint_optimizer_audit import load_data,save
from public_conjoint_matched_audit import nuisance
from power_tuned_choice_audit import objective

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def independent_support(x,y,g,m,power,kind):
    n,p=x.shape;weights=np.r_[np.full(m,(1-power)/m),np.full(n-m,power/(n-m))]
    moment=(x[:m].T@(y[:m]-power*g[:m]))/m+power*(x[m:].T@g[m:])/(n-m)
    if kind=='interior':
        scale=np.maximum(np.sqrt(np.mean(x*x,axis=0)),1e-12)
        matrix=(x*weights[:,None]/scale).T;eye=sparse.eye(n,format='csr');one=sparse.csr_matrix(np.ones((n,1)))
        aub=sparse.vstack([sparse.hstack([-eye,one]),sparse.hstack([eye,one])],format='csr')
        r=linprog(np.r_[np.zeros(n),-1.],A_eq=sparse.hstack([sparse.csr_matrix(matrix),sparse.csr_matrix((p,1))]),b_eq=moment/scale,A_ub=aub,b_ub=np.r_[np.zeros(n),np.ones(n)],bounds=[(0.,1.)]*n+[(0.,.5)],method='highs-ipm')
        assert r.success and r.x[-1]>1e-8,(power,kind,r.message)
        residual=float(np.max(abs(matrix@r.x[:-1]-moment/scale)))
        assert residual<1e-7
        return dict(status='interior',probability_margin=float(r.x[-1]),primal_residual=residual)
    aub=sparse.hstack([sparse.csr_matrix(x),-sparse.eye(n,format='csr')])
    r=linprog(np.r_[-moment,weights],A_ub=aub,b_ub=np.zeros(n),bounds=[(-1.,1.)]*p+[(0.,None)]*n,method='highs-ipm')
    assert r.success,(power,kind,r.message)
    v=r.x[:p];cost=float(weights@np.maximum(x@v,0)-moment@v)
    assert cost< -1e-9,(power,kind,cost)
    return dict(status='outside_mean_support',direction=v.tolist(),recession_cost=cost)

def normal_risks(b):
    a=float(b[0]);r=float(np.linalg.norm(b[1:]));assert r>0
    z=abs(a)/r
    absolute=2*r*np.exp(-z*z/2)/np.sqrt(2*np.pi)+abs(a)*(2*ndtr(z)-1)
    density=lambda u:np.exp(-((u-a)/r)**2/2)/(r*np.sqrt(2*np.pi))
    # Integration is in logit coordinates, unlike the original Gaussian-z
    # integration. Omitted smoothing/variance tails are bounded by exp(-50).
    smooth=quad(lambda u:np.logaddexp(0,-abs(u))*density(u),-50,50,epsabs=1e-11,epsrel=1e-11,points=[0],limit=250)[0]
    variance=quad(lambda u:expit(u)*expit(-u)*density(u),-50,50,epsabs=1e-11,epsrel=1e-11,points=[0],limit=250)[0]
    return float(absolute/2+smooth),float(.5-variance)

def run(a,mode):
    folder=a.root/'results'/mode;plan=json.loads((a.root/('public_plan.json' if mode=='public' else 'gaussian_plan.json')).read_text())
    raw_path=folder/'path_records.csv';raw=pd.read_csv(raw_path);before=sha(raw_path);details=json.loads((folder/'coefficient_records.json').read_text());by_key={d['case']:d for d in details}
    if mode=='public':
        xall,yall,z,_,_=load_data(a.input_dir);prefix=plan['input_prefix_rows'];x=xall[:prefix];y=yall[:prefix];m=plan['label_rows'];rng=np.random.RandomState(plan['seed']);cases=[]
        for trial in range(plan['allocations']):
            order=(rng.permutation(240)[:,None]*5+np.arange(5)[None]).reshape(-1)
            xx=x[order];yy=y[order];zz=z[order];predictions=[]
            for name in plan['prediction_routines']:
                g,fallback=nuisance(xx,yy,zz,m,name=='AI');predictions.append((name,g))
            cases.append((f'public_{trial}',xx,yy,m,predictions))
    else:
        rng=np.random.default_rng(plan['seed']);cases=[]
        for m in plan['label_rows']:
            for trial in range(plan['replications_per_label_budget']):
                n=m*(1+plan['auxiliary_ratio']);q=int(np.floor(plan['noise_slope_ratio']*m));xx=np.c_[np.ones(n),rng.normal(size=(n,q+1))];yy=rng.integers(0,2,n).astype(float);zz=np.logical_xor(yy,rng.random(n)<plan['proxy_flip_probability']).astype(float);e=plan['proxy_flip_probability']
                cases.append((f'gaussian_{m}_{trial}',xx,yy,m,[('oracle',e+(1-2*e)*zz)]))
    maxima=dict(gradient_difference=0.,replayed_gradient_over_power=0.,bound_violation=0.,test_log_loss_difference=0.,test_Brier_difference=0.,positive_Jensen_violation=0.)
    endpoints=[];resolutions=[];roots=0;verified=raw.copy();initial_nonstandard=int((~raw.support_status.isin(['interior','outside_mean_support'])).sum())
    for key,x,y,m,predictions in cases:
        detail=by_key[key];v=detail['separator_direction']
        if v is not None:
            v=np.asarray(v,float);v/=np.linalg.norm(v)
            signed=(2*y[:m]-1)*(x[:m]@v);positive=np.maximum(signed,0);negative=np.maximum(-signed,0);s=float(positive.mean());nu=float(np.sqrt(np.mean(positive**2)));t=float(negative.mean());cl=float(np.sqrt(np.linalg.eigvalsh(x[:m].T@x[:m]/m)[-1]));dv=float(np.mean(abs(x[:m]@v))+np.mean(abs(x[m:]@v)))
        for prediction,g in predictions:
            part=raw[(raw['case']==key)&(raw.prediction==prediction)].sort_values('power',ascending=False)
            assert len(part)==len(plan['powers'])
            good=part[part.fit_status=='numerical_finite_root'];threshold=float(good.power.max()) if len(good) else None
            if threshold is not None:
                endpoint=independent_support(x,y,g,m,threshold,'interior');endpoints.append(dict(case=key,prediction=prediction,power=threshold,**endpoint))
            outside=part if threshold is None else part[part.power>threshold]
            if len(outside):
                power=float(outside.power.min());endpoint=independent_support(x,y,g,m,power,'outside');endpoints.append(dict(case=key,prediction=prediction,power=power,**endpoint))
                d=np.asarray(endpoint['direction'])
                for _,row in outside.iterrows():
                    loss=objective(x,y,g,m,float(row.power));cost=float(loss.weights@np.maximum(x@d,0)-loss.moment@d);assert cost< -1e-9
                    mask=(verified['case']==key)&(verified.prediction==prediction)&(verified.power==row.power)
                    verified.loc[mask,'verified_support_status']='outside_mean_support'
                    if row.support_status not in ['interior','outside_mean_support']:
                        resolutions.append(dict(case=key,prediction=prediction,power=float(row.power),initial_status=row.support_status,verified_status='outside_mean_support',direct_recession_cost=cost,direction=d.tolist()))
            for _,row in good.iterrows():
                matching=[f for f in detail['fits'] if f['prediction']==prediction and f['power']==row.power];assert len(matching)==1;b=np.asarray(matching[0]['coefficient'])
                loss=objective(x,y,g,m,float(row.power));_,gradient=loss.value(b);res=float(np.linalg.norm(gradient));roots+=1
                maxima['gradient_difference']=max(maxima['gradient_difference'],abs(res-row.gradient_l2));maxima['replayed_gradient_over_power']=max(maxima['replayed_gradient_over_power'],res/row.power)
                assert res/row.power<1.1e-7
                if v is not None:
                    weight=positive/(m*s);score=x[:m]@b
                    bound=s/(nu*cl)*max(0.,np.log(s/(2*(row.power*dv+res+t))))
                    maxima['bound_violation']=max(maxima['bound_violation'],bound-np.linalg.norm(b));assert bound<=np.linalg.norm(b)+1e-7
                    actual=float(np.sum(positive*expit(-(2*y[:m]-1)*score))/m)
                    lower=s/2*np.exp(-nu*cl*np.linalg.norm(b)/s)
                    maxima['positive_Jensen_violation']=max(maxima['positive_Jensen_violation'],lower-actual);assert actual>=lower-1e-12
                if mode=='gaussian':
                    nll,brier=normal_risks(b)
                    assert row.test_classification_error==.5
                else:
                    u=xall[prefix:]@b;nll=float(np.mean(np.logaddexp(0,u)-yall[prefix:]*u));brier=float(np.mean((expit(u)-yall[prefix:])**2))
                maxima['test_log_loss_difference']=max(maxima['test_log_loss_difference'],abs(nll-row.test_log_loss));maxima['test_Brier_difference']=max(maxima['test_Brier_difference'],abs(brier-row.test_Brier_loss))
                assert abs(nll-row.test_log_loss)<2e-7 and abs(brier-row.test_Brier_loss)<2e-8
                mask=(verified['case']==key)&(verified.prediction==prediction)&(verified.power==row.power);verified.loc[mask,'verified_support_status']='interior'
    assert verified.verified_support_status.notna().all()
    assert len(resolutions)==initial_nonstandard
    assert sha(raw_path)==before
    out=a.out/mode;out.mkdir(parents=True,exist_ok=True);verified.to_csv(out/'path_records_verified.csv',index=False)
    save(out/'support_endpoint_certificates.json',endpoints);save(out/'support_resolutions.json',resolutions)
    status=dict(date='2026-11-12',mode=mode,retained_rows=len(raw),root_records_independently_replayed=roots,independent_support_endpoint_programs=len(endpoints),all_intermediate_powers_covered_by_same_prediction_coercivity_nesting=True,initial_nonstandard_support_states=initial_nonstandard,resolved_support_states=len(resolutions),remaining_unresolved=0,original_records_unchanged=True,original_records_sha256=before,maxima=maxima,source_sha256=sha(Path(__file__)))
    save(out/'independent_risk_verification.json',status);print(json.dumps(status),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    for mode in ['public','gaussian']:run(a,mode)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
