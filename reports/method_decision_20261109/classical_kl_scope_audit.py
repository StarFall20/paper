"""Profile-KL/model-discrimination equivalence and fixed-weight map scope checks."""
from pathlib import Path
from fractions import Fraction as F
import argparse,json
import numpy as np
import pandas as pd
from scipy.special import softmax,xlogy
from threadpoolctl import threadpool_limits
import published_dce_decision_audit as a


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--route-data',type=Path,required=True); ap.add_argument('--vehicle-data',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); args=ap.parse_args()
    route,*_=a.read_route(args.route_data); vehicle,*_=a.read_vehicle(args.vehicle_data)
    rows=[]; maxerr=0.; ratios=[]
    for kind,menus,param in [('route',route,10),('vehicle',vehicle,60)]:
        pairs,scope=a.generate_pairs(menus,kind,param); mm=[m for pair in pairs for m in pair]; m=len(pairs); groups=[np.array([2*t,2*t+1]) for t in range(m)]
        x,p,v,h,hu=a.model_arrays(mm,kind,param); theta=np.array([1.,-.2,.3]); beta=np.array([0.,-.3,-.2,-.3] if kind=='route' else [0.,0.,-1.,.1,.05,.1,.2,.1]); u=x@beta
        for q in [.1,.25,.5,.75,.9]:
            w=np.tile([1-q,q],m)/m
            f,c=a.two_information_methods(h,v,groups,w)
            local=float(theta@f@theta); classical=float(theta@c@theta); assert abs(local-classical)<1e-12; maxerr=max(maxerr,abs(local-classical))
            for eta in [.1,.05,.01,.002]:
                true=softmax(u+eta*(hu@theta),axis=1); kl=0.
                for ix in groups:
                    mass=w[ix].sum(); bestnull=np.einsum('n,nj->j',w[ix]/mass,true[ix])
                    kl+=float(np.sum(w[ix,None]*xlogy(true[ix],true[ix]/bestnull)))
                approx=.5*eta**2*local; ratio=kl/approx
                if eta==.002: ratios.append(abs(ratio-1))
                rows.append({'dataset':kind,'allocation_q':q,'eta':eta,'local_fibre_information':local,'classical_directional_dispersion':classical,'classical_profiled_KL':kl,'half_eta2_local_information':approx,'KL_second_order_ratio':ratio,'response_family':'binary logit' if kind=='route' else 'three-category softmax','scope':'local matched-alternative check; not a field power calculation'})
    assert max(ratios)<.01
    pd.DataFrame(rows).to_csv(args.out/'profiled_KL_model_discrimination_check.csv',index=False)
    # At a noncommensurate rational weight near 10 CHF/hour there are no
    # accidental ties in this finite released menu catalogue. All arithmetic exact.
    weights=[F(5),F(10),F(20),F(40),F(10000001,1000000)]
    audits=[]
    raw=list({a.rawkey(x):x for x in route}.values())
    for hourly in weights:
        keys=[]
        for menu in raw:
            key=tuple(v for row in menu for v in [row[1]+hourly/F(60)*row[0],row[2],row[3]])
            keys.append(key)
        audits.append({'CHF_per_hour_exact':str(hourly),'CHF_per_hour_decimal':float(hourly),'raw_menus':len(raw),'candidate_fibres_absolute_summary':len(set(keys)),'menu_contrast_rank':len(raw)-len(set(keys)),'changes_the_candidate_null':hourly!=10,'interpretation':'different maps define different nulls; not evidence that the original fixed-weight test is invalid'})
    assert audits[-1]['menu_contrast_rank']==0
    pd.DataFrame(audits).to_csv(args.out/'fixed_weight_scope_sensitivity.csv',index=False)
    a.save(args.out/'KL_and_scope_verification.json',{'matched_directional_information_max_error':maxerr,'profiled_KL_second_order_max_relative_error_eta_0_002':max(ratios),'KL_cells':len(rows),'noncommensurate_weight_finite_catalogue_injective':True,'fixed_candidate_nulls_changed_not_posthoc_rejected':True,'scope':'finite released menus only; no assertion that a finite observed sample saturates a continuous-population model'})
    print('CLASSICAL_KL_AND_SCOPE_COMPLETE',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1): main()
