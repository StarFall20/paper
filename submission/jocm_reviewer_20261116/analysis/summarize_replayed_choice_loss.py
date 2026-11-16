"""Summarize known-truth expected choice loss from original-draw replay.

This is finite-support KL, not independent human predictive validation.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.special import expit,log_expit
from public_conjoint_optimizer_audit import save


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    checks=[]
    for u0 in [-10.,-1.,0.,2.,10.]:
        for u in [-1000.,-10.,0.,10.,1000.]:
            truth=expit(u0)
            stored=np.logaddexp(0,u)-np.logaddexp(0,u0)-truth*(u-u0)
            independent=truth*(log_expit(u0)-log_expit(u))+(1-truth)*(log_expit(-u0)-log_expit(-u))
            checks.append(abs(stored-independent))
    assert max(checks)<1e-10
    trials=pd.read_csv(a.out/'replayed_power_trial_diagnostics.csv')
    assert len(trials)==2500 and not trials.duplicated(['replication','power']).any()
    assert trials.excess_logloss.dropna().min()>=-1e-12
    wide=trials.pivot(index='replication',columns='power',values='excess_logloss');rows=[]
    for ref,comp in [(1.,.01),(1.,.1),(1.,.5),(0.,.01)]:
        common=wide[[ref,comp]].dropna();difference=common[comp]-common[ref]
        rows.append({'power_reference':ref,'power_comparator':comp,'shared_finite_estimates':len(common),
            'mean_KL_reference':common[ref].mean(),'mean_KL_comparator':common[comp].mean(),
            'mean_KL_difference':difference.mean(),'difference_MCSE':difference.std(ddof=1)/np.sqrt(len(common)),
            'median_KL_difference':difference.median(),'fraction_comparator_larger_KL':difference.gt(0).mean(),
            'unit':'expected Bernoulli KL,nats per choice,uniform1200-row frozen support',
            'new_independent_response_sample':False})
    f=pd.DataFrame(rows);f.to_csv(a.out/'replayed_choice_loss_pairs.csv',index=False)
    save(a.out/'replayed_choice_loss_checks.json',{'stable_logistic_Bregman_vs_independent_Bernoulli_KL_checks':len(checks),
        'maximum_KL_identity_difference':max(checks),'original_draws':500,'paired_comparisons':len(f),
        'original_model_fits_reused':True,'new_independent_human_validation':False})
    print(f[['power_reference','power_comparator','shared_finite_estimates','mean_KL_reference','mean_KL_comparator','difference_MCSE']].to_string(index=False))


if __name__=='__main__':main()
