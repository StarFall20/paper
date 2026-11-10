"""Independent proper-target augmentation comparison with the corrected loss."""
from pathlib import Path
import argparse,json
import numpy as np
from power_tuned_choice_audit import WeightedLogit,support,objective
from public_conjoint_optimizer_audit import save

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 rng=np.random.default_rng(20261125);counts={};matches=0
 for j in range(40):
  m=12;N=16;d=3+j%12;x=rng.normal(size=(m+N,d));y=rng.integers(0,2,m).astype(float)
  g=rng.integers(0,2,N).astype(float) if j%2 else rng.uniform(.05,.95,N)
  statuses=[]
  for c in [.001,.1,1.,10.]:
   w=np.r_[np.full(m,1/m),np.full(N,c/N)]/(1+c);b=x.T@(w*np.r_[y,g]);sc=support(WeightedLogit(x,w,b));statuses.append(sc['status'])
  assert len(set(statuses))==1,(j,statuses)
  assert statuses[0] in ['interior','boundary_or_near_boundary']
  counts[statuses[0]]=counts.get(statuses[0],0)+1;matches+=1
 # Fixed one-coordinate example:proper augmentation is finite at every c>0,
 # while residual correction can remain exterior despite identical curvature rank.
 x=np.ones((4,1));y=np.array([1.,1.,0.,0.]);g=np.array([.25,.25,.75,.75]);ordinary=[];corrected=[]
 for c in [.001,.1,1.,10.]:
  w=np.r_[np.full(2,.5),np.full(2,c/2)]/(1+c);b=x.T@(w*np.r_[y[:2],g[2:]]);ordinary.append(support(WeightedLogit(x,w,b))['status'])
 for lam in [.001,.1,.5,1.]:corrected.append(support(objective(x,y,g,2,lam))['status'])
 assert set(ordinary)=={'interior'} and set(corrected)=={'outside_mean_support'}
 result={'random_fixed_designs_with_invariant_proper_weight_status':matches,'proper_status_counts':counts,'proper_comparison_weights':[.001,.1,1.,10.],'same_rank_analytic_proper_statuses':ordinary,'same_rank_analytic_corrected_statuses':corrected,'new_general_separation_theorem_claimed':False,'independent_novelty_proven':False}
 save(a.out/'proper_augmentation_invariance_checks.json',result);print(json.dumps(result))

if __name__=='__main__':main()
