"""Retrospective exact sign check for the first endpoint-loss allocation.

The direction and its support interpretation use classical recession geometry.
The exact calculation defines frozen predictions by their stored IEEE values.
"""
from pathlib import Path
from fractions import Fraction
from functools import reduce
from math import lcm,gcd
import argparse,json,hashlib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from public_conjoint_optimizer_audit import load_data,save
from public_conjoint_matched_audit import nuisance
from observable_low_label_path import ridge_anchor
from power_tuned_choice_audit import objective,power_weight,solve


def integer_ray(raw):
    f=[Fraction(float(v)).limit_denominator(1000000) for v in raw]
    den=lcm(*(q.denominator for q in f));v=[int(q*den) for q in f]
    divisor=reduce(gcd,(abs(q) for q in v));v=[q//divisor for q in v]
    # The source LP uses L1 normalization. Check the proposed rational ray.
    normal=sum(abs(q) for q in v)
    assert max(abs(float(a)/normal-b) for a,b in zip(v,raw))<1e-10
    return v


def exact_check(x,y,g,m,d):
    assert np.max(abs(x-np.rint(x)))==0
    xi=np.rint(x).astype(int)
    u=[sum(int(a)*b for a,b in zip(row,d)) for row in xi]
    margins=[(2*int(y[i])-1)*u[i] for i in range(m)]
    human=sum(Fraction(max(u[i],0)-int(y[i])*u[i],m) for i in range(m))
    gg=[Fraction(float(v)) for v in g]
    unit=sum((max(u[i],0)-gg[i]*u[i])/(len(x)-m) for i in range(m,len(x)))
    unit+=sum((gg[i]-int(y[i]))*u[i]/m for i in range(m))
    sensitivity=Fraction(sum(abs(v) for v in u[:m]),m)+Fraction(sum(abs(v) for v in u[m:]),len(x)-m)
    assert human==0 and min(margins)>=0 and max(margins)>0 and unit<0
    radius=-unit/sensitivity
    return dict(integer_direction=d,minimum_integer_human_margin=min(margins),
        positive_human_margins=sum(v>0 for v in margins),human_recession_exact='0',
        unit_recession_numerator=str(unit.numerator),unit_recession_denominator=str(unit.denominator),
        unit_recession_normalized_L1=float(unit)/sum(abs(v) for v in d),
        frozen_prediction_uniform_perturbation_radius=float(radius),
        perturbation_radius_numerator=str(radius.numerator),perturbation_radius_denominator=str(radius.denominator),
        every_positive_power_unbounded_below=True,zero_power_human_nonattainment=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    records=a.base/'results/records.csv'
    if not records.exists():records=records.with_suffix('.csv.gz')
    f=pd.read_csv(records);assert len(f)==36000
    first=f[(f.auxiliary_rows==50)&(f.method=='source_tuned_each_N')].set_index(['trial','label_rows','prediction'])
    last=f[(f.auxiliary_rows==1200-f.label_rows)&(f.method=='source_tuned_each_N')].set_index(['trial','label_rows','prediction'])
    lost=first.status.eq('interior')&last.status.eq('outside_mean_support')
    keys=sorted(lost[lost].index);assert keys
    selected=min(k[0] for k in keys);keys=[k for k in keys if k[0]==selected]
    save(a.out/'secondary_case_plan.json',dict(timing='After the full primary audit and its independent verification.',
        selection_rule='The smallest trial among tuned first-to-last finite-to-exterior transitions; retain every such prediction/budget condition in that trial.',
        selected_trial=int(selected),selected_conditions=len(keys),new_independent_cohorts=0,
        purpose='Make the existing rare event concrete and check saved directions in exact arithmetic; no prevalence, new tuning or new theorem claim.'))
    x,y,z,_,_=load_data(a.input_dir);rng=np.random.RandomState(20261109)
    for trial in range(selected+1):order=(rng.permutation(240)[:,None]*5+np.arange(5)[None]).reshape(-1)
    xx=x[order];zz=z[order]
    directions=json.loads((a.base/'results/exterior_directions.json').read_text());cards=[];predictions=[]
    for trial,m,pred in keys:
        yy=np.r_[y[order[:m]],np.full(1200-m,np.nan)];g,_=nuisance(xx,yy,zz,m,pred=='AI')
        pilot,_=ridge_anchor(xx[:m],yy[:m]);before,info=power_weight(xx[:m+50],yy[:m+50],g[:m+50],m,pilot)
        root,check=solve(objective(xx[:m+50],yy[:m+50],g[:m+50],m,before),pilot)
        assert root is not None and check['status']=='finite_root'
        ray=next(d for d in directions if d['trial']==trial and d['label_rows']==m and d['prediction']==pred and d['auxiliary_rows']==1200-m and d['method']=='source_tuned_each_N')
        integers=integer_ray(ray['direction']);exact=exact_check(xx,yy,g,m,integers)
        cards.append(dict(trial=int(trial),label_rows=int(m),prediction=pred,before_auxiliary_rows=50,after_auxiliary_rows=1200-int(m),
            before_power=before,before_root=root.tolist(),before_root_check=check,
            after_power=float(last.loc[(trial,m,pred),'power']),before_PSD_trace=float(first.loc[(trial,m,pred),'psd_score_trace']),
            after_PSD_trace=float(last.loc[(trial,m,pred),'psd_score_trace']),**exact))
        predictions.append(dict(trial=int(trial),label_rows=int(m),prediction=pred,g=g.tolist()))
    save(a.out/'frozen_case_predictions.json',predictions);save(a.out/'exact_case.json',dict(selection='Retrospective illustration, one existing allocation.',conditions=cards,
        exactness='Rational integer attributes, binary labelled choices and the stored dyadic frozen predictions. Before-pool interior and roots receive numerical checks; after-pool recession signs are exact.',
        raw_human_choice_vectors_shipped=False,new_model_draws=0))
    print(json.dumps(dict(selected_trial=int(selected),conditions=len(cards),before_roots_checked=len(cards),after_exact_all_power_obstructions=len(cards))))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
