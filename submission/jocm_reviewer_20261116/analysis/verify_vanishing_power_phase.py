"""Reconstruct only unresolved Gaussian cells and retain their raw classifications."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
from vanishing_power_choice import rescaled_support
from power_tuned_choice_audit import support,objective
from public_conjoint_optimizer_audit import save

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();frame=pd.read_csv(a.out/'vanishing_power_phase.csv');verified=frame.copy();targets=frame[frame.unresolved.gt(0)]
 assert len(targets)==1,'This verifier recognizes the completed500draw plan,not a changed experiment'
 rng=np.random.default_rng(20261123);resolutions=[]
 for m in [40,80]:
  for e in [.25,.5]:
   for ratio in [.45,.6,.7]:
    d=int(round(m*ratio));part=targets[(targets.labels==m)&(targets.epsilon==e)&(targets.dimensions==d)]
    found=0
    for rep in range(500):
     x=rng.normal(size=(2*m,d));y=rng.integers(0,2,2*m).astype(float);z=np.logical_xor(y,rng.random(2*m)<e).astype(float);g=e+(1-2*e)*z
     if part.empty:continue
     for _,row in part.iterrows():
      lam=float(row.power);raw=rescaled_support(x,y,g,m,lam)
      if raw['status']!='unresolved':continue
      found+=1;other=support(objective(x,y,g,m,lam));resolved=other['status'] in ['interior','outside_mean_support']
      resolutions.append({'labels':m,'epsilon':e,'dimensions':d,'replication':rep,'power':lam,'raw_rescaled_check':raw,'independent_original_moment_check':other,'resolved':resolved})
      if resolved:
       key=(verified.labels==m)&(verified.epsilon==e)&(verified.dimensions==d)&(verified.power==str(row.power))
       assert key.sum()==1
       verified.loc[key,'unresolved']-=1;field='interior' if other['status']=='interior' else 'outside';verified.loc[key,field]+=1;verified.loc[key,'finite_frequency']=verified.loc[key,'interior']/500
    if not part.empty:assert found==int(part.unresolved.sum())
 assert resolutions
 verified.to_csv(a.out/'vanishing_power_phase_verified.csv',index=False)
 save(a.out/'vanishing_power_resolution.json',{'raw_phase_sha256':hashlib.sha256((a.out/'vanishing_power_phase.csv').read_bytes()).hexdigest(),'raw_unresolved':int(frame.unresolved.sum()),'remaining_unresolved':int(verified.unresolved.sum()),'cases':resolutions,'raw_outputs_overwritten':False})
 print(json.dumps({'raw_unresolved':int(frame.unresolved.sum()),'remaining_unresolved':int(verified.unresolved.sum()),'independent_statuses':[r['independent_original_moment_check']['status'] for r in resolutions]}))

if __name__=='__main__':main()
