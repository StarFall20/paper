"""Boundary and source sensitivity follow-up; not a new behavioral mechanism."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits
from application_panel_comparison import Panel,fit
from application_support_models import build,save,cell_contrast,fit_ab
import dh36_observational_analysis as old

def main():
    p=argparse.ArgumentParser(); p.add_argument('--input-dir',type=Path,required=True); p.add_argument('--instrument',type=Path,required=True); p.add_argument('--folds',type=Path,required=True); p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    spec=json.loads(a.instrument.read_text()); d,*_=old.read_exports(a.input_dir,spec); models,names,*_=build(spec,d); y=d[[f'q{i}' for i in range(1,9)]].to_numpy(int)-1
    folds=d[['participant_id']].merge(pd.read_csv(a.folds),on='participant_id',how='left',validate='one_to_one').fold.to_numpy(); source=d.ai_label.to_numpy(); oo=(y==2).all(1)
    df=pd.DataFrame({'version':d.version,'export_source_label_group':np.where(source,'AI样本','other_source_labels'),'all_outside':oo})
    counts=df.groupby(['version','export_source_label_group']).agg(records=('all_outside','size'),all_eight_outside=('all_outside','sum')).reset_index(); counts.to_csv(a.out/'all_outside_by_export_source.csv',index=False)
    rows=[]; fits={}; contrasts=[]; diagnostics=[]
    for population,select,defs in [('all_360',np.ones(len(y),bool),[('spike_linear','linear','spike'),('spike_cells','cells','spike')]),('300_other_source_labels',~source,[('MNL_linear','linear','MNL'),('RI_linear','linear','RI'),('participation_2class','linear','participation'),('taste_2class','linear','taste')])]:
        yy=y[select]; ff=folds[select]; nn=len(yy)
        for label,form,kind in defs:
            xx=models[form][select]; result=fit(xx,yy,kind,True); fits[population+'_'+label]=result
            joint=np.zeros(nn); marg=np.zeros(nn); predref=np.zeros((nn,9))
            for fold in np.sort(np.unique(ff)):
                train=ff!=fold; test=~train; f=fit(xx[train],yy[train],kind); obj=Panel(xx[test],yy[test],kind)
                _,_,_,ll,pr,rf=obj.eval(f['beta'],True); joint[test]=ll; marg[test]=np.log(pr[np.arange(test.sum())[:,None],np.arange(8)[None],yy[test]]).sum(1); predref[test]=rf
                diagnostics.append({'population':population,'model':label,'fold':fold,**f['checks']})
            rows.append({'population':population,'model':label,'records':nn,'parameters':len(result['beta']),'fivefold_joint_nll_per_choice':-joint.mean()/8,'fivefold_marginal_nll_per_choice':-marg.mean()/8,'observed_all_outside':int((yy==2).all(1).sum()),'fivefold_predicted_all_outside':predref[:,8].sum(),'at_bound':str(result['checks']['at_bound'])})
            if kind=='spike': result['outside_point_mass_estimate']=expit(result['beta'][-1])
            if form=='cells': contrasts.append({'population':population,'model':label,**cell_contrast(result['beta'],result['covariance'],names,('G2',89))})
            print(population,label,result['checks'],flush=True)
            save(a.out/'source_boundary_model_fits.json',fits); pd.DataFrame(rows).to_csv(a.out/'source_boundary_comparison.csv',index=False)
        b,cov,cols,check=fit_ab(models['cells'][select],yy); cn=[names[j] for j in cols]
        contrasts.append({'population':population,'model':'conditional_A_B_cells',**cell_contrast(b,cov,cn,('G2',89))}); fits[population+'_conditional_A_B_cells']={'beta':b,'covariance':cov,'checks':check,'names':cn}
    save(a.out/'source_boundary_model_fits.json',fits); save(a.out/'source_boundary_fold_checks.json',diagnostics); pd.DataFrame(contrasts).to_csv(a.out/'source_boundary_same_price_contrasts.csv',index=False)
    source_path=a.out/'source_statement.json'
    # Re-estimation must not erase later independently inspected provenance.
    if not source_path.exists():
        save(source_path,{'date':'2026-11-09','researcher_report':'The researcher reports that the platform described the service as recruitment of human respondents.','evidence_level':'researcher-transmitted platform description; original service documentation not inspected','analysis_population':'all 360 records retained in pooled analyses','export_field':'60 records have original 来源 label AI样本; retained unchanged in private exports and audit metadata','sensitivity':'source-group analyses reflect observed source-pattern differences, not a declaration that the 60 records were generated or invalid','all_outside_pattern':'42 all-outside records, all in the 60-record group; other 300 contain none','version_allocation':'different recruitment channels each administered only one version; no allocation record; no causal version effect claim'})
    print(pd.DataFrame(rows).to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1): main()
