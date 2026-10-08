"""Power curves indexed by departure amplitude and respondent count."""
from __future__ import annotations
import argparse, csv, os
import numpy as np
from support_complete_fibre_benchmark import generate, score_test
from support_complete_vs_saturated_benchmark import saturated_lr_pvalue
from support_complete_fibre_design import profiles_3x3

def run(reps=100, randomization_reps=199, out='results/support_complete_power_curve.csv'):
    rows=[]
    for condition in ('null','hidden_orthogonal'):
        for respondents in (200,400,800):
            for eta in (0.10,0.20,0.30,0.45):
                score_rej=[]; lr_rej=[]
                for rep in range(reps):
                    pts,fibre,profile,y=generate(20261007+rep+1000*respondents,
                        respondents=respondents,condition=condition,eta=eta)
                    ps=score_test(pts,fibre,profile,y,'support_complete',
                        randomization_reps=randomization_reps,seed=20262000+rep)
                    pl=saturated_lr_pvalue(pts,fibre,profile,y,
                        reps=randomization_reps,seed=20263000+rep)
                    score_rej.append(int(ps<0.05)); lr_rej.append(int(pl<0.05))
                rows.append({'condition':condition,'respondents':respondents,'eta':eta,
                    'replications':reps,'randomization_reps':randomization_reps,
                    'support_complete_rejection_rate':float(np.mean(score_rej)),
                    'saturated_lr_rejection_rate':float(np.mean(lr_rej))})
                print(condition,respondents,eta,round(float(np.mean(score_rej)),3),round(float(np.mean(lr_rej)),3),flush=True)
    os.makedirs(os.path.dirname(out) or '.',exist_ok=True)
    with open(out,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('wrote',len(rows),'rows to',out)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reps',type=int,default=100);p.add_argument('--randomization-reps',type=int,default=199);p.add_argument('--out',default='results/support_complete_power_curve.csv');run(**vars(p.parse_args()))
