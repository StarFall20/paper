"""Audit the finite solution of a specified prediction-assisted binary logit loss.

The attainable-moment LP is classical. This interface preserves the selected
objective and does not silently substitute clipping, tuning or regularization.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from power_tuned_choice_audit import objective,support,solve


def audit(x,y,g,m,power=1.,fit=True):
    x=np.asarray(x,float);y=np.asarray(y,float);g=np.asarray(g,float)
    if x.ndim!=2 or x.shape[1]==0:raise ValueError('X must be a nonempty two-dimensional feature matrix')
    n,d=x.shape
    if len(y)!=n or len(g)!=n:raise ValueError('X,choice and prediction lengths differ')
    if not 0<m<n:raise ValueError('Both labelled and auxiliary rows are required')
    if not 0<=power<=1:raise ValueError('Power must lie between zero and one')
    if not np.isfinite(x).all() or not np.isfinite(g).all() or np.any((g<0)|(g>1)):raise ValueError('Features must be finite and predictions must be probabilities')
    if not np.isfinite(y[:m]).all() or not np.isin(y[:m],[0,1]).all():raise ValueError('Labelled choices must be zero or one')
    yy=np.r_[y[:m],np.zeros(n-m)]
    loss=objective(x,yy,g,m,power);check=support(loss)
    result={'objective':'L_label(Y)+power*(L_aux(prediction)-L_label(prediction))',
        'labels':m,'auxiliary_rows':n-m,'parameters':d,'power':float(power),
        'support':check,'coefficient':None,'fit_status':'not_attempted',
        'scope':'numerical finite-objective audit; no CI guarantee or distributional phase approximation',
        'automatic_target_change':False,'classical_support_method':True}
    if fit and check['status']=='interior':
        beta,fitcheck=solve(loss,np.zeros(d));result['fit_status']=fitcheck['status'];result['fit_check']=fitcheck
        if beta is not None:result['coefficient']=beta.tolist()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',type=Path);p.add_argument('--features',required=True,help='Comma-separated feature columns; include an intercept column explicitly if desired')
    p.add_argument('--power',type=float,default=1.);p.add_argument('--no-fit',action='store_true');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    frame=pd.read_csv(a.input);features=[s.strip() for s in a.features.split(',') if s.strip()]
    required=set(features+['choice','prediction','labelled']);missing=required-set(frame.columns)
    if missing:raise ValueError('Missing columns: '+','.join(sorted(missing)))
    if not frame.labelled.isin([0,1]).all():raise ValueError('labelled must contain only zero or one')
    frame=frame.sort_values('labelled',ascending=False,kind='stable');m=int(frame.labelled.sum())
    result=audit(frame[features].to_numpy(float),frame.choice.to_numpy(float),frame.prediction.to_numpy(float),m,a.power,not a.no_fit)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n');print(json.dumps({'support_status':result['support']['status'],'fit_status':result['fit_status']}))


if __name__=='__main__':main()
