"""Finite-sample marginal-null checks valid with arbitrary within-record dependence.

Each 2x3 table uses its conditional multivariate-hypergeometric law; LR ordering
is fixed before enumeration. Bonferroni/Holm controls the eight-task family
without assuming a joint response-vector null. No individual data are released.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from scipy.special import gammaln,logsumexp,xlogy
import dh36_observational_analysis as old
from dh36_assumption_conditioned_validation import js


def exact_table(count0,count1):
    total=count0+count1; n1=int(count1.sum()); n0=int(count0.sum()); n=n0+n1
    denominator=gammaln(n+1)-gammaln(n1+1)-gammaln(n0+1)
    vals=[]; lp=[]; e1=total*n1/n; e0=total*n0/n
    obs=2*np.sum(xlogy(count1,np.divide(count1,e1,out=np.ones(3),where=e1>0))+xlogy(count0,np.divide(count0,e0,out=np.ones(3),where=e0>0)))
    for a in range(max(0,n1-int(total[1]+total[2])),min(int(total[0]),n1)+1):
        lo=max(0,n1-a-int(total[2])); hi=min(int(total[1]),n1-a)
        b=np.arange(lo,hi+1); c=n1-a-b; c1=np.column_stack([np.full(len(b),a),b,c]); c0=total-c1
        ll=(gammaln(total+1)-gammaln(c1+1)-gammaln(c0+1)).sum(1)-denominator
        lr=2*(xlogy(c1,np.divide(c1,e1,out=np.ones_like(c1,dtype=float),where=e1>0))+xlogy(c0,np.divide(c0,e0,out=np.ones_like(c0,dtype=float),where=e0>0))).sum(1)
        vals.append(lr); lp.append(ll)
    vals=np.concatenate(vals); lp=np.concatenate(lp)
    pv=float(np.exp(logsumexp(lp[vals>=obs-1e-10])))
    # All possible tail ranks must have at most alpha conditional null mass.
    unique=np.unique(np.round(vals,10)); exact_size={}
    for alpha in [.01,.05,.1]:
        order=np.argsort(-vals); sv=vals[order]; pp=np.exp(lp[order]); cum=np.cumsum(pp)
        # Group numerical ties before assigning tail probabilities.
        starts=np.r_[0,np.where(np.abs(np.diff(sv))>1e-9)[0]+1]; ends=np.r_[starts[1:]-1,len(sv)-1]
        size=0.
        for start,end in zip(starts,ends):
            if cum[end]<=alpha+1e-10: size+=float(pp[start:end+1].sum())
        exact_size[str(alpha)]=size
        assert size<=alpha+1e-8
    return {'statistic_LR':float(obs),'exact_conditional_p':min(1.,pv),'enumerated_tables':len(vals),'probability_mass_error':abs(float(np.exp(logsumexp(lp)))-1),'conditional_size_checks':exact_size}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input-dir',type=Path,required=True); ap.add_argument('--instrument',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); args=ap.parse_args()
    spec=json.loads(args.instrument.read_text()); d,*_=old.read_exports(args.input_dir,spec); y=d[[f'q{t}' for t in range(1,9)]].to_numpy(int)-1; z=d.version2.to_numpy(); rows=[]; validation=[]
    for t in range(8):
        c0=np.bincount(y[z==0,t],minlength=3); c1=np.bincount(y[z==1,t],minlength=3); r=exact_table(c0,c1)
        validation.append({'task':t+1,**r}); rows.append({'task':t+1,'n_V1':int((z==0).sum()),'n_V2':int((z==1).sum()),'LR':r['statistic_LR'],'exact_conditional_p':r['exact_conditional_p'],'enumerated_tables':r['enumerated_tables']})
        swapped=exact_table(c1,c0); assert abs(swapped['exact_conditional_p']-r['exact_conditional_p'])<1e-9
    pv=np.array([r['exact_conditional_p'] for r in rows]); order=np.argsort(pv); holm=np.empty(8); prev=0
    for pos,j in enumerate(order): prev=max(prev,min(1.,(8-pos)*pv[j])); holm[j]=prev
    for j,r in enumerate(rows): r['holm_p_8']=holm[j]; r['bonferroni_p_8']=min(1.,8*pv[j])
    pd.DataFrame(rows).to_csv(args.out/'exact_marginal_task_tests.csv',index=False)
    js(args.out/'exact_marginal_validation.json',{'global_Bonferroni_p':float(min(1.,8*pv.min())),'global_reject_0_05':bool(8*pv.min()<=.05),'target':'equal three-category marginal probabilities for all eight tasks','assumptions':'independent records, each group multinomial sampling with common task probabilities under null; arbitrary dependence across tasks within record','finite_sample_validity':'Exact conditional tests per task; Bonferroni/Holm union bound requires no cross-task independence. No claim of exact pooled omnibus weak-null test.','validation':validation,'source_and_assignment_verified':False})
    print('EXACT MARGINAL GLOBAL BONFERRONI P',min(1.,8*pv.min()),flush=True)

if __name__=='__main__': main()
