"""Human primary estimates and labelled AI-pretest pairing controls.

Original exports remain unchanged. Re-pairings are computational controls,
never observed respondents. Only original held-out human records are scored.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits
import dh36_observational_analysis as old
from application_panel_comparison import Panel, fit
from application_support_models import build, save, cell_contrast, fit_ab


def main():
    p=argparse.ArgumentParser()
    for name in ['input-dir','instrument','folds','classification','primary-out','pairing-out']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.primary_out.mkdir(parents=True,exist_ok=True);a.pairing_out.mkdir(parents=True,exist_ok=True)
    classification=json.loads(a.classification.read_text())
    assert classification['synthetic_records']['total']==60 and classification['primary_records']['total']==300
    d,provenance,*_=old.read_exports(a.input_dir,json.loads(a.instrument.read_text()))
    expected={'V1':'dbe667ecc7511b69fcb5c36147f986499dbbbf23ec1d7f638eb137cfbb080e13','V2':'40b3cc7daee37b5ad0e1060b77c12cdccc9c5dff9cff1a57a75e32b8d5b7ce01'}
    assert {v['version']:v['sha256'] for v in provenance}==expected
    forms,names,*_=build(json.loads(a.instrument.read_text()),d)
    y=d[[f'q{i}' for i in range(1,9)]].to_numpy(int)-1;ai=d.ai_label.to_numpy();human=~ai;version=d.version2.to_numpy()
    folds=d[['participant_id']].merge(pd.read_csv(a.folds),on='participant_id',how='left',validate='one_to_one').fold.to_numpy()
    assert not pd.isna(folds).any() and len(np.unique(folds))==5 and human.sum()==300 and ai.sum()==60
    for v in [0,1]: assert (human&(version==v)).sum()==150 and (ai&(version==v)).sum()==30
    fits={};rows=[];contrasts=[];fold_fits={};diagnostics=[]
    definitions=[('MNL_linear','linear','MNL'),('MNL_cells','cells','MNL'),('RI_linear','linear','RI'),('RI_cells','cells','RI')]
    for label,form,kind in definitions:
        xx=forms[form][human];yy=y[human];ff=folds[human]
        result=fit(xx,yy,kind,True);fits[label]=result
        if form=='cells': contrasts.append({'population':'human_primary_300','model':label,**cell_contrast(np.array(result['beta']),np.array(result['covariance']),names,('G2',89))})
        joint=np.zeros(len(yy));marginal=np.zeros(len(yy));oo=np.zeros(len(yy))
        for fold in np.unique(ff):
            tr=ff!=fold;te=~tr;rr=fit(xx[tr],yy[tr],kind);fold_fits[(label,int(fold))]=rr
            _,_,_,ll,pr,rf=Panel(xx[te],yy[te],kind).eval(rr['beta'],True)
            joint[te]=ll;marginal[te]=np.log(pr[np.arange(te.sum())[:,None],np.arange(8)[None],yy[te]]).sum(1);oo[te]=rf[:,8]
            diagnostics.append({'model':label,'fold':int(fold),**rr['checks']})
        rows.append({'model':label,'human_records':300,'choice_rows':2400,'parameters':len(result['beta']),'joint_nll_per_choice':-joint.mean()/8,'marginal_nll_per_choice':-marginal.mean()/8,'observed_all_outside':0,'predicted_all_outside':oo.sum()})
        save(a.primary_out/'model_fits.json',fits);pd.DataFrame(rows).to_csv(a.primary_out/'primary_model_comparison.csv',index=False)
        print('Human primary',label,rows[-1],flush=True)
    b,cov,cols,check=fit_ab(forms['cells'][human],y[human]);cn=[names[j] for j in cols]
    fits['conditional_A_B_cells']={'beta':b,'covariance':cov,'checks':check,'names':cn}
    contrasts.append({'population':'human_primary_300','model':'conditional_A_B_cells',**cell_contrast(b,cov,cn,('G2',89))})
    early=human&d.before_bulk_collection_date.to_numpy();assert early.sum()==1
    keep=human&~early;rr=fit(forms['cells'][keep],y[keep],'MNL',True)
    fits['MNL_cells_without_early_record']=rr
    contrasts.append({'population':'human_299_early_record_sensitivity','model':'MNL_cells',**cell_contrast(np.array(rr['beta']),np.array(rr['covariance']),names,('G2',89))})
    save(a.primary_out/'model_fits.json',fits);save(a.primary_out/'fold_fit_checks.json',diagnostics)
    pd.DataFrame(contrasts).to_csv(a.primary_out/'same_price_rule_contrasts.csv',index=False)
    save(a.primary_out/'population_audit.json',{'human_primary':300,'AI_pretest_excluded_from_human_estimation':60,'human_choice_rows':2400,'early_record_flag_not_automatic_invalidity':1,'raw_hashes':expected,'classification_sha256':hashlib.sha256(a.classification.read_bytes()).hexdigest(),'source_definition_resolved':True,'version_randomization_verified':False,'new_AI_samples_requested':False,'raw_records_changed':False})

    x=forms['linear'];records=[];parameters=[];fitchecks=[];invariance=[];original={}
    def evaluate(label,condition,seed,fold,train,yt,result):
        test=human&(folds==fold);kind={'MNL_linear':'MNL','RI_linear':'RI','spike_linear':'spike'}[label]
        _,_,_,ll,pr,rf=Panel(x[test],y[test],kind).eval(result['beta'],True)
        ml=np.log(pr[np.arange(test.sum())[:,None],np.arange(8)[None],y[test]]).sum()
        records.append({'condition':condition,'seed':int(seed),'model':label,'fold':int(fold),'human_heldout_records':int(test.sum()),'joint_logscore_sum':ll.sum(),'marginal_logscore_sum':ml,'predicted_human_all_outside':rf[:,8].sum()})
        parameters.append({'condition':condition,'seed':int(seed),'model':label,'fold':int(fold),'training_human_records':int((human&train).sum()),'training_AI_records':int((ai&train).sum()),'training_AI_constant_vectors':int(np.all(yt[ai[train]]==yt[ai[train],:1],axis=1).sum()) if (ai&train).any() else 0,'training_AI_all_outside':int(np.all(yt[ai[train]]==2,axis=1).sum()) if (ai&train).any() else 0,'service_intercept_sd':float(np.exp(result['beta'][-1])) if kind=='RI' else None,'outside_component_weight':float(expit(result['beta'][-1])) if kind=='spike' else None})
        fitchecks.append({'condition':condition,'seed':int(seed),'model':label,'fold':int(fold),**result['checks']})
    for fold in np.unique(folds):
        htrain=human&(folds!=fold);mixed=folds!=fold
        for label,kind in [('MNL_linear','MNL'),('RI_linear','RI'),('spike_linear','spike')]:
            rr=fold_fits[(label,int(fold))] if kind!='spike' else fit(x[htrain],y[htrain],kind)
            evaluate(label,'human_only',-1,fold,htrain,y[htrain],rr)
            rr=fit(x[mixed],y[mixed],kind);original[(label,int(fold))]=rr
            evaluate(label,'mixed_original_AI',-1,fold,mixed,y[mixed],rr)
    pd.DataFrame(records).to_csv(a.pairing_out/'fold_prediction_scores.csv',index=False)
    for seed in range(202611090,202611100):
        for fold in np.unique(folds):
            train=folds!=fold;yt=y[train].copy();st=ai[train];vt=version[train];rng=np.random.default_rng(seed+int(fold)*100000)
            for v in [0,1]:
                use=np.where(st&(vt==v))[0]
                for task in range(8): yt[use,task]=rng.permutation(yt[use,task])
                for task in range(8): assert np.array_equal(np.bincount(yt[use,task],minlength=3),np.bincount(y[train][use,task],minlength=3))
            beta=original[('MNL_linear',int(fold))]['beta']
            oldval,oldgrad=Panel(x[train],y[train],'MNL').eval(beta);newval,newgrad=Panel(x[train],yt,'MNL').eval(beta)
            err=abs(oldval-newval);gerr=float(np.max(abs(oldgrad-newgrad)));assert err<1e-8 and gerr<1e-8
            invariance.append({'seed':seed,'fold':int(fold),'task_version_counts_preserved':True,'MNL_nll_difference':err,'MNL_gradient_max_difference':gerr})
            evaluate('MNL_linear','mixed_repaired_AI',seed,fold,train,yt,original[('MNL_linear',int(fold))])
            for label,kind in [('RI_linear','RI'),('spike_linear','spike')]:
                rr=fit(x[train],yt,kind);evaluate(label,'mixed_repaired_AI',seed,fold,train,yt,rr)
        pd.DataFrame(records).to_csv(a.pairing_out/'fold_prediction_scores.csv',index=False)
        pd.DataFrame(parameters).to_csv(a.pairing_out/'fold_model_parameters.csv',index=False)
        save(a.pairing_out/'fold_fit_checks.json',fitchecks)
        print('Pairing sensitivity seed complete',seed,flush=True)
    summary=pd.DataFrame(records).groupby(['condition','seed','model'],as_index=False).agg(human_heldout_records=('human_heldout_records','sum'),joint_logscore_sum=('joint_logscore_sum','sum'),marginal_logscore_sum=('marginal_logscore_sum','sum'),predicted_human_all_outside=('predicted_human_all_outside','sum'))
    assert summary.human_heldout_records.eq(300).all()
    summary['human_joint_nll_per_choice']=-summary.joint_logscore_sum/2400;summary['human_marginal_nll_per_choice']=-summary.marginal_logscore_sum/2400
    summary.to_csv(a.pairing_out/'human_target_prediction.csv',index=False)
    pd.DataFrame(invariance).to_csv(a.pairing_out/'exact_marginal_controls.csv',index=False)
    save(a.pairing_out/'numerical_checks.json',{'raw_hashes':expected,'classification_sha256':hashlib.sha256(a.classification.read_bytes()).hexdigest(),'fixed_repairings':10,'human_records_scored_once_per_condition':300,'all_50_training_marginal_controls_passed':len(invariance)==50,'maximum_MNL_objective_difference':max(v['MNL_nll_difference'] for v in invariance),'maximum_MNL_gradient_difference':max(v['MNL_gradient_max_difference'] for v in invariance),'original_files_changed':False,'generated_control_records_presented_as_observations':False,'formal_permutation_p_value_or_power_claim':False})
    print(summary[['condition','seed','model','human_joint_nll_per_choice']].to_string(index=False),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
