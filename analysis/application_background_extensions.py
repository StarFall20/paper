"""Bounded exploratory background interactions with source-sensitive validation."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
from scipy.stats import norm
from threadpoolctl import threadpool_limits
import dh36_observational_analysis as old
from application_support_models import build,save
from application_panel_comparison import Panel,fit as panel_fit
from questionnaire_reuse_audit import fit as cluster_fit


def design(base,form,d,modifier,cell_names):
    # The same source definition used in the descriptive audit; no relabelling.
    source=d.ai_label.to_numpy().astype(float)
    adj=base[:,:,:,:2]*source[:,None,None,None]
    base=np.concatenate([base,adj],axis=-1)
    if modifier=='ever_paid':
        m=d.paid_service.isin([1,2]).to_numpy(); unknown=d.paid_service.isin([4,5]).to_numpy()
    else:
        m=(d.app_use==4).to_numpy(); unknown=(d.app_use==5).to_numpy()
    # All 300 sensitivity records have the same source indicator, so source
    # columns are omitted by design instead of fitting a singular model.
    if not source.any(): base=base[:,:,:,:-2]
    if form=='linear': idx=np.arange(5); primary=np.zeros(base.shape[-1]);primary[3:5]=[-1,1]
    else: idx=np.r_[0,1,np.arange(4,11)];primary=np.zeros(base.shape[-1]);primary[cell_names.index('G2_fee_89')]=1
    interaction=base[:,:,:,idx]*m[:,None,None,None]
    missing=base[:,:,:,:2]*unknown[:,None,None,None]
    x=np.concatenate([base,interaction,missing],axis=-1)
    c=np.zeros(x.shape[-1]); c[base.shape[-1]:base.shape[-1]+len(idx)]=primary[idx]
    fullrank=int(np.linalg.matrix_rank((x[:,:,:-1]-x[:,:,-1:]).reshape(-1,x.shape[-1])))
    assert fullrank==x.shape[-1],(form,modifier,fullrank,x.shape[-1])
    return base,x,c,idx,{'modifier_records':int(m.sum()),'reference_records':int((~m&~unknown).sum()),'unknown_records':int(unknown.sum()),'source_intercepts':bool(source.any()),'rank':fullrank}


def money_contrast(b,cov,k,idx):
    # -10*(gamma_G2-gamma_G1)/beta_fee for both groups. Delta method is
    # explicitly supplementary; weak denominators are flagged, not hidden.
    effect=b[4]-b[3]; p=b[2]; p1=p+b[k+list(idx).index(2)]; e1=effect+b[k+list(idx).index(4)]-b[k+list(idx).index(3)]
    val0=-10*effect/p;val1=-10*e1/p1
    g0=np.zeros(len(b));g0[3]=10/p;g0[4]=-10/p;g0[2]=10*effect/p**2
    g1=np.zeros(len(b));g1[3]=g1[k+list(idx).index(3)]=10/p1;g1[4]=g1[k+list(idx).index(4)]=-10/p1;g1[2]=g1[k+list(idx).index(2)]=10*e1/p1**2
    g=g1-g0;se=float(np.sqrt(max(0,g@cov@g)))
    c0=np.zeros(len(b));c0[2]=1;c1=c0.copy();c1[k+list(idx).index(2)]=1
    return {'reference_CNY':val0,'modifier_CNY':val1,'difference_CNY':val1-val0,'delta_ci_low':val1-val0-1.96*se,'delta_ci_high':val1-val0+1.96*se,'reference_fee_coefficient':p,'modifier_fee_coefficient':p1,'both_negative_and_separated_from_zero':bool(p+1.96*np.sqrt(c0@cov@c0)<0 and p1+1.96*np.sqrt(c1@cov@c1)<0),'scope':'exploratory linear-fee ratios; model-specific utility scales; no causal or population WTP claim'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input-dir',type=Path,required=True);ap.add_argument('--instrument',type=Path,required=True);ap.add_argument('--folds',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    spec=json.loads(a.instrument.read_text());d,*_=old.read_exports(a.input_dir,spec);forms,names,*_=build(spec,d)
    folds=d[['participant_id']].merge(pd.read_csv(a.folds),on='participant_id',how='left',validate='one_to_one').fold.to_numpy();assert not pd.isna(folds).any()
    yy=d[[f'q{i}' for i in range(1,9)]].to_numpy(int)-1
    rows=[];money=[];fits={};foldchecks=[];base_predictions={}
    for population,use in [('all_360',np.ones(len(d),bool)),('300_other_source_labels',~d.ai_label.to_numpy())]:
        dd=d.loc[use].reset_index(drop=True);y=yy[use];ff=folds[use]
        for modifier in ['ever_paid','weekly_use']:
            for form in ['linear','cells']:
                base,x,c,idx,counts=design(forms[form][use],form,dd,modifier,names)
                kinds=['MNL','RI'] if modifier=='ever_paid' and form=='cells' else ['MNL']
                for kind in kinds:
                    key='_'.join([population,modifier,form,kind])
                    if kind=='MNL':b,cov,checks=cluster_fit(x,y)
                    else:
                        r=panel_fit(x,y,kind,True);b=np.array(r['beta']);cov=np.array(r['covariance']);checks=r['checks'];c=np.r_[c,0.]
                    effect=float(c@b);se=float(np.sqrt(c@cov@c));pv=float(2*norm.sf(abs(effect/se)))
                    fits[key]={'beta':b,'covariance':cov,'checks':checks,'group_counts':counts,'contrast_vector':c}
                    row={'population':population,'modifier':modifier,'form':form,'kind':kind,'difference_in_G2_G1_utility':effect,'cluster_se':se,'ci_low':effect-1.95996398454*se,'ci_high':effect+1.95996398454*se,'p_exploratory':pv,**counts}
                    cache=(population,form,kind)
                    if cache not in base_predictions:
                        bl=np.zeros(len(y))
                        for fold in np.unique(ff):
                            tr=ff!=fold;te=~tr; rr=panel_fit(base[tr],y[tr],kind);bl[te]=Panel(base[te],y[te],kind).eval(rr['beta'],True)[3]/8
                            foldchecks.append({'population':population,'model':'baseline_'+form+'_'+kind,'fold':fold,**rr['checks']})
                        base_predictions[cache]=bl
                    el=np.zeros(len(y))
                    for fold in np.unique(ff):
                        tr=ff!=fold;te=~tr; rr=panel_fit(x[tr],y[tr],kind);el[te]=Panel(x[te],y[te],kind).eval(rr['beta'],True)[3]/8
                        foldchecks.append({'population':population,'model':modifier+'_'+form+'_'+kind,'fold':fold,**rr['checks']})
                    delta=el-base_predictions[cache];sdelta=delta.std(ddof=1)/np.sqrt(len(y))
                    row.update({'cv_baseline_joint_nll':-base_predictions[cache].mean(),'cv_extended_joint_nll':-el.mean(),'cv_mean_logscore_gain':delta.mean(),'cv_descriptive_ci_low':delta.mean()-1.96*sdelta,'cv_descriptive_ci_high':delta.mean()+1.96*sdelta})
                    rows.append(row)
                    if form=='linear':money.append({'population':population,'modifier':modifier,**money_contrast(b,cov,base.shape[-1],idx)})
                    print(key,{k:row[k] for k in ['difference_in_G2_G1_utility','ci_low','ci_high','p_exploratory','cv_mean_logscore_gain']},flush=True)
                    save(a.out/'background_model_fits.json',fits);pd.DataFrame(rows).to_csv(a.out/'background_contrasts.csv',index=False);save(a.out/'background_cv_fit_checks.json',foldchecks)
    # Family is fixed in the plan, including both populations and RI follow-ups.
    assert len(rows)==10
    order=np.argsort([r['p_exploratory'] for r in rows]);last=0.
    for j,i in enumerate(order):last=max(last,min(1.,(len(rows)-j)*rows[i]['p_exploratory']));rows[i]['p_Holm_10_exploratory']=last
    pd.DataFrame(rows).to_csv(a.out/'background_contrasts.csv',index=False);pd.DataFrame(money).to_csv(a.out/'background_linear_fee_ratios.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1): main()
