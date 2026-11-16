"""Positive-power support nesting, zero-endpoint boundary and coefficient paths.

Classical support LPs and vanishing-regularization ideas are attributed prior.
This script verifies the stated corrected objective,not a new tuned estimator.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.integrate import quad
from scipy.optimize import brentq,linprog
from scipy.special import expit,ndtr
from scipy.stats import norm
from finite_ratio_choice_boundary import tail,critical_ratio
from power_tuned_choice_audit import objective,support
from public_conjoint_optimizer_audit import save


def positive_limit(tau,e):
 def value(v):return float((1-e)*tail(-e*v)+e*tail(-(1-e)*v)+tau*(tail((1-e)*v/tau)+tail(e*v/tau)))
 fp=lambda a:2*(a*ndtr(-a)-norm.pdf(a))
 def gradient(v):return float(-(1-e)*e*fp(-e*v)-e*(1-e)*fp(-(1-e)*v)+(1-e)*fp((1-e)*v/tau)+e*fp(e*v/tau))
 hi=1.
 while gradient(hi)<0:hi*=2
 v=brentq(gradient,0.,hi,xtol=1e-12)
 return 1+tau-value(v),v


def rescaled_support(x,y,g,m,power=None):
 """Relative-interior LP;None is the limiting box with unbounded label uppers."""
 x=np.asarray(x,float);n,p=x.shape;N=n-m;tau=N/m
 keep=x[m:] if power==1 else x
 if np.linalg.matrix_rank(keep)<p:return {'status':'rank_deficient'}
 signs=np.r_[1-2*y[:m],np.ones(N)];A=(x*signs[:,None]).T
 r=np.abs(y[:m]-g[:m]);lo=np.r_[r,-g[m:]/tau]
 if power is None:
  hi=np.r_[np.full(m,np.inf),(1-g[m:])/tau]
  width=np.r_[np.ones(m),np.full(N,1/tau)]
 else:
  assert 0<power<=1
  hi=np.r_[r+(1/power-1),(1-g[m:])/tau];width=hi-lo
 eye=sparse.eye(n,format='csr');alpha=sparse.csr_matrix(width[:,None]);top=sparse.hstack([-eye,alpha],format='csr');rhs=[-lo]
 finite=np.isfinite(hi)
 aub=sparse.vstack([top,sparse.hstack([eye[finite],alpha[finite]],format='csr')],format='csr');rhs.append(hi[finite])
 aeq=sparse.hstack([sparse.csr_matrix(A),sparse.csr_matrix((p,1))],format='csr')
 fit=linprog(np.r_[np.zeros(n),-1.],A_ub=aub,b_ub=np.r_[*rhs],A_eq=aeq,b_eq=np.zeros(p),bounds=[(None,None)]*n+[(0.,.5)],method='highs')
 if fit.status==2:return {'status':'outside_mean_support'}
 if not fit.success:return {'status':'unresolved','message':fit.message}
 v=fit.x[:-1];margin=float(fit.x[-1]);resid=float(np.max(np.abs(A@v)))
 if resid>1e-7:return {'status':'unresolved','residual':resid}
 result={'status':'interior' if margin>1e-8 else 'boundary_or_near_boundary','margin':margin,'residual':resid}
 if power is None and result['status']=='interior':
  # Construct an explicit positive power whose finite box contains this witness.
  extra=max(1.,float(np.max(v[:m]-r)))+1.
  result['certified_positive_power']=min(.5,1/(1+2*extra))
 return result


def numeric_checks(out):
 rows=[];quaderr=0.;monotone=0.
 for tau in [.25,1.,4.]:
  for e in [.1,.25,.5]:
   limit,v=positive_limit(tau,e);assert limit>.5
   independent=lambda a:quad(lambda z:(z-a)**2*norm.pdf(z),a,max(12.,a+12.),epsabs=2e-12)[0]
   val=(1-e)*independent(-e*v)+e*independent(-(1-e)*v)+tau*(independent((1-e)*v/tau)+independent(e*v/tau))
   quaderr=max(quaderr,abs(1+tau-val-limit))
   prev=limit
   for lam in [.001,.02,.05,.2,.5,1.]:
    value,_=critical_ratio(tau,e,lam);monotone=max(monotone,value-prev);assert value<=prev+1e-11
    rows.append({'tau':tau,'epsilon':e,'power':lam,'critical_ratio':value,'positive_power_right_limit':limit,'limit_minimizer':v,'zero_endpoint':.5});prev=value
   assert abs(rows[-6]['critical_ratio']-limit)<1e-9
 assert quaderr<1e-10
 pd.DataFrame(rows).to_csv(out/'vanishing_power_boundaries.csv',index=False)
 rng=np.random.default_rng(20261124);matches=0;statuscounts={}
 for j in range(120):
  m=12+j%9;N=m;p=3+j%24;lam=[.05,.2,.5,1.][j%4];e=[.25,.5][j%2]
  x=rng.normal(size=(m+N,p));y=rng.integers(0,2,m+N).astype(float);z=np.logical_xor(y,rng.random(m+N)<e).astype(float);g=e+(1-2*e)*z
  one=support(objective(x,y,g,m,lam))['status'];two=rescaled_support(x,y,g,m,lam)['status'];assert one==two,(j,one,two)
  matches+=1;statuscounts[one]=statuscounts.get(one,0)+1
 # Three exact one-coordinate recession outcomes at the same human separation.
 one=np.ones((4,1));y=np.array([1.,1.,0.,0.]);examples=[]
 for name,gl,gu,expected in [('rescued',.75,.25,'interior'),('boundary',.75,.75,'boundary_or_near_boundary'),('exterior',.25,.75,'outside_mean_support')]:
  g=np.array([gl,gl,gu,gu]);lim=rescaled_support(one,y,g,2)
  assert lim['status']==expected,(name,lim)
  for lam in [.001,.05,.5,1.]:assert support(objective(one,y,g,2,lam))['status']==expected
  examples.append({'case':name,'label_prediction':gl,'auxiliary_prediction':gu,'positive_power_limit_status':lim['status']})
 path=[]
 # Two-dimensional strict-separation example;the limiting LP has a unique point.
 XL=np.array([[1.,0.],[0.,1.],[1.,1.]]);XU=np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
 # a(v)=.25||v||_1-(2/15)(v1+v2),with v1,v2>=1 on the margin constraints.
 lp=linprog(np.r_[-np.full(2,2/15),np.full(4,.25)],A_ub=np.block([[-XL,np.zeros((3,4))],[XU,-np.eye(4)]]),b_ub=np.r_[-np.ones(3),np.zeros(4)],bounds=[(None,None)]*2+[(0.,None)]*4,method='highs')
 assert lp.success and max(abs(lp.x[:2]-1))<1e-12
 # Independent scalar score uses expit(-b),avoiding cancellation of1-expit(b).
 for lam in [1e-1,1e-2,1e-3,1e-4,1e-6,1e-8]:
  beta1=np.log(2/lam-1);L=np.log(1/lam)
  grad=lambda b:-(1-lam)*(expit(-b)+expit(-2*b))/3+lam*(.5*expit(b)-23/60)
  beta2=brentq(grad,0.,100.,xtol=1e-12)
  rel=abs(grad(beta2))/lam;assert rel<1e-10
  path.append({'power':lam,'log_inverse_power':L,'analytic_1d_root':beta1,'analytic_1d_ratio':beta1/L,'symmetric_2d_each_coefficient':beta2,'normalized_2d_coordinate':beta2/L,'limiting_LP_coordinate':1.,'scaled_score_error':rel})
 assert path[-1]['normalized_2d_coordinate']<path[0]['normalized_2d_coordinate'] and abs(path[-1]['normalized_2d_coordinate']-1)<.06
 pd.DataFrame(path).to_csv(out/'vanishing_power_coefficient_path.csv',index=False)
 result={'normal_quadrature_limit_max_error':quaderr,'positive_power_monotonicity_numeric_tolerance_excess':monotone,'original_rescaled_LP_matches':matches,'LP_status_counts':statuscounts,'analytic_recession_cases':examples,'limiting_margin_LP_solution':lp.x[:2],'coefficient_path_scaled_score_max':max(r['scaled_score_error'] for r in path),'ordinary_zero_endpoint_separate':True,'joint_vanishing_power_sample_limit_proven':False,'independent_novelty_proven':False}
 save(out/'vanishing_power_numeric_checks.json',result);return result


def phase(out,reps):
 rng=np.random.default_rng(20261123);powers=[1.,.5,.2,.05];rows=[];patterns=[];done=0;violations=0;certificates=0
 for m in [40,80]:
  for e in [.25,.5]:
   for ratio in [.45,.6,.7]:
    p=int(round(m*ratio));counts={str(lam):{} for lam in [0.,*powers,'limit']};pattern={}
    for rep in range(reps):
     x=rng.normal(size=(2*m,p));y=rng.integers(0,2,2*m).astype(float);z=np.logical_xor(y,rng.random(2*m)<e).astype(float);g=e+(1-2*e)*z
     statuses={0.:support(objective(x,y,g,m,0.))['status']}
     for lam in powers:statuses[lam]=rescaled_support(x,y,g,m,lam)['status']
     limit=rescaled_support(x,y,g,m);statuses['limit']=limit['status']
     bits=[statuses[lam]=='interior' for lam in powers]
     violations+=sum(a and not b for a,b in zip(bits,bits[1:]))
     assert not any(bits) or limit['status']=='interior',(m,e,p,rep,statuses)
     if limit['status']=='interior':
      cert=rescaled_support(x,y,g,m,limit['certified_positive_power']);assert cert['status']=='interior',(m,e,p,rep,limit,cert);certificates+=1
     key=''.join('1' if statuses[lam]=='interior' else '0' for lam in [0.,*powers,'limit']);pattern[key]=pattern.get(key,0)+1
     for lam,status in statuses.items():counts[str(lam)][status]=counts[str(lam)].get(status,0)+1
    assert violations==0
    for lam,c in counts.items():
     unresolved=sum(v for k,v in c.items() if k in ['unresolved','rank_deficient']);boundary=c.get('boundary_or_near_boundary',0)
     theoretical=.5 if lam=='0.0' else positive_limit(1.,e)[0] if lam=='limit' else critical_ratio(1.,e,float(lam))[0]
     rows.append({'labels':m,'auxiliary_rows':m,'epsilon':e,'dimensions':p,'dimension_label_ratio':p/m,'power':lam,'replications':reps,'interior':c.get('interior',0),'outside':c.get('outside_mean_support',0),'boundary_or_near_boundary':boundary,'unresolved':unresolved,'finite_frequency':c.get('interior',0)/reps,'critical_ratio':theoretical})
    for key,n in pattern.items():patterns.append({'labels':m,'epsilon':e,'dimensions':p,'pattern_zero_unit_half_point2_point05_limit':key,'count':n})
    done+=1;pd.DataFrame(rows).to_csv(out/'vanishing_power_phase.csv',index=False);pd.DataFrame(patterns).to_csv(out/'vanishing_power_paired_patterns.csv',index=False)
    save(out/'vanishing_power_status.json',{'completed_DGP_blocks':done,'requested_DGP_blocks':12,'replications_per_block':reps,'positive_power_nesting_violations':violations,'explicit_small_power_certificates_checked':certificates,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print('Positive-power DGP blocks complete',done,'of12',flush=True)


def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--checks-only',action='store_true');p.add_argument('--replications',type=int,default=500);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 print(json.dumps(numeric_checks(a.out),default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x),flush=True)
 if not a.checks_only:phase(a.out,a.replications)

if __name__=='__main__':main()
