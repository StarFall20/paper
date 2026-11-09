"""Independent finite-state checks for the matched-risk proposition and loss tails."""
from pathlib import Path
import argparse,json
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
from scipy.stats import norm
from scipy.special import ndtr
from imperfect_predictor_choice_boundary import matched_moment_values
from power_tuned_choice_audit import objective,support
from public_conjoint_optimizer_audit import save


def states(name):
 c=.4;high=.499;ph=c*c/(high*high);K=.9*ph*high/2
 conf=[(c,1.,.5-K/c)] if name=='constant_confidence' else [(0.,1-ph,.5),(high,ph,.05)]
 rows=[]
 for C,p,e in conf:
  for y in [0.,1.]:
   for z in [0.,1.]:rows.append((C,y,z,.5+C*(2*z-1),p*.5*((1-e) if y==z else e)))
 return np.array(rows)


def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 arrays={n:states(n) for n in ['constant_confidence','mixed_confidence']};moments={};coverror=0.;brier=[]
 for name,d in arrays.items():
  C,y,z,g,w=d.T;assert abs(sum(w)-1)<1e-14
  moments[name]={'mean_prediction':float(w@g),'prediction_variance':float(w@((g-.5)**2)),
   'label_prediction_covariance':float(w@((y-.5)*(g-.5))),'Brier_risk':float(w@((y-g)**2))}
  brier.append(moments[name]['Brier_risk'])
 for tau in [.05,.1,.25,1.,4.]:
  for lam in [0.,.25,.5,1.]:
   variances=[]
   for d in arrays.values():
    C,y,z,g,w=d.T;label=(.5-y)+lam*(g-.5);aux=lam*(.5-g)
    variances.append(float(w@(label**2)+(w@(aux**2))/tau))
   coverror=max(coverror,abs(variances[0]-variances[1]))
 assert coverror<1e-13 and abs(brier[0]-brier[1])<1e-14
 # Evaluate the boundary with numerical normal integration, independently of f.
 values=matched_moment_values();qerror=0.
 def expected_distance(lo,hi):
  left=quad(lambda h:(lo-h)**2*norm.pdf(h),-12,lo,epsabs=2e-12)[0]
  right=quad(lambda h:(h-hi)**2*norm.pdf(h),hi,12,epsabs=2e-12)[0]
  return left+right
 for row in values:
  d=arrays[row['prediction']];C,y,z,g,w=d.T;tau=row['tau'];u=row['scaled_minimizer'];t=tau*u
  primary=-t*t*float(w@((y-g)**2))
  # Auxiliary sign orientation has interval[-(.5+C),.5-C] for either Z.
  auxiliary=tau*sum(weight*(1-expected_distance(-(.5+c)*u,(.5-c)*u)) for c,weight in zip(C,w))
  qerror=max(qerror,abs(primary+auxiliary-row['critical_ratio']))
 assert qerror<1e-9,qerror
 # Individual labelled losses are unbounded below; their sum can be coercive.
 x=np.ones((4,1));y=np.array([0.,1.,0.,0.]);g=np.array([.2,.8,.2,.8]);tails=[]
 for lam in [.3,.7,1.]:
  loss=objective(x,y,g,2,lam);status=support(loss)['status'];assert status=='interior'
  assert abs(loss.value(np.zeros(1))[1][0])<1e-14
  for Y,G,sign in [(0.,.2,-1.),(1.,.8,1.)]:
   u=sign*1000.;individual=(1-lam)*np.logaddexp(0,u)-(Y-lam*G)*u
   expected=-lam*.2*1000.;assert abs(individual-expected)<1e-10
   tails.append({'power':lam,'Y':Y,'prediction':G,'u':u,'label_row_loss':individual,'aggregate_loss':loss.value(np.array([u]))[0],'aggregate_moment_status':status})
 # Check strict convexity of H_u(c^2) over the displayed support without a near-zero numerical division.
 min_curvature=np.inf
 for u in [.2,1.,2.,4.]:
  for c in [.01,.1,.3,.499]:
   fp=lambda a:2*(a*ndtr(-a)-norm.pdf(a))
   hp=u*(fp((.5+c)*u)-fp((.5-c)*u))
   hpp=2*u*u*(ndtr(-(.5+c)*u)+ndtr(-(.5-c)*u))
   curvature=(c*hpp-hp)/(4*c**3);assert curvature>0
   min_curvature=min(min_curvature,curvature)
 result={'prediction_state_moments':moments,'all_fixed_power_covariance_max_difference':coverror,'normal_quadrature_boundary_max_error':qerror,'H_of_squared_confidence_min_checked_curvature':float(min_curvature),'label_row_tail_examples':tails,'general_prior_directly_applicable':False,'novelty_proven':False}
 save(a.out/'matched_moment_independent_checks.json',result);print(json.dumps(result))


if __name__=='__main__':main()
