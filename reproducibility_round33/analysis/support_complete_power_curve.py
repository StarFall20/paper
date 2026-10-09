"""Non-ceiling power planning for unrestricted-fibre score/CMH and LR."""
from __future__ import annotations
import argparse,csv,os
import numpy as np
from fibre_baseline_comparison import conditional_tests
from support_complete_fibre_benchmark import generate


def wilson(successes,n):
    z=1.959963984540054; p=successes/n; d=1+z*z/n
    mid=(p+z*z/(2*n))/d; half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return mid-half,mid+half


def run(reps=500,randomization_reps=499,out='results/support_complete_power_curve.csv'):
    rows=[]
    for n in (200,400,800):
        for eta in (0.,.05,.10,.15,.20,.30,.45):
            condition='null' if eta==0 else 'hidden_orthogonal'
            counts={'support_complete':0,'complete_fibre_lr':0}
            for rep in range(reps):
                pts,fibre,profile,y=generate(20261007+rep+1000*n,n,condition,eta)
                pv=conditional_tests(pts,fibre,profile,y,randomization_reps,20262000+rep)
                for method in counts: counts[method]+=int(pv[method]<=.05)
            row=dict(condition=condition,respondents=n,eta=eta,replications=reps,randomization_reps=randomization_reps)
            for method,c in counts.items():
                lo,hi=wilson(c,reps);row[method+'_rejection_rate']=c/reps;row[method+'_ci_low']=lo;row[method+'_ci_high']=hi
            rows.append(row)
            print(n,eta,{m:c/reps for m,c in counts.items()},flush=True)
    os.makedirs(os.path.dirname(out) or '.',exist_ok=True)
    with open(out,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reps',type=int,default=500);p.add_argument('--randomization-reps',type=int,default=499);p.add_argument('--out',default='results/support_complete_power_curve.csv');run(**vars(p.parse_args()))
