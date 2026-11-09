"""Human-only re-pairing controls for the AI-pretest originality audit."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits
import dh36_observational_analysis as old
from application_support_models import build,save
from application_panel_comparison import Panel,fit


def main():
    p=argparse.ArgumentParser()
    for k in ['input-dir','instrument','folds','out']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    spec=json.loads(a.instrument.read_text());d,provenance,*_=old.read_exports(a.input_dir,spec);human=~d.ai_label.to_numpy();d=d.loc[human].reset_index(drop=True)
    assert len(d)==300 and d.version2.value_counts().to_dict()=={0:150,1:150}
    forms,*_=build(spec,d);x=forms['linear'];y=d[[f'q{i}' for i in range(1,9)]].to_numpy(int)-1;v=d.version2.to_numpy()
    ff=d[['participant_id']].merge(pd.read_csv(a.folds),on='participant_id',how='left',validate='one_to_one').fold.to_numpy();assert not pd.isna(ff).any()
    rows=[];diagnostics=[];params=[];controls=[];original={};higher=[]
    def evaluate(condition,seed,fold,yt,result,kind):
        test=ff==fold
        _,_,_,ll,pr,rf=Panel(x[test],y[test],kind).eval(result['beta'],True)
        marginal=np.log(pr[np.arange(test.sum())[:,None],np.arange(8)[None],y[test]]).sum()
        rows.append({'condition':condition,'seed':seed,'fold':int(fold),'model':kind,'human_heldout_records':int(test.sum()),'joint_logscore_sum':float(ll.sum()),'marginal_logscore_sum':float(marginal),'predicted_all_outside':float(rf[:,8].sum())})
        params.append({'condition':condition,'seed':seed,'fold':int(fold),'model':kind,'training_constant_vectors':int(np.all(yt==yt[:,:1],axis=1).sum()),'training_all_outside':int(np.all(yt==2,axis=1).sum()),'service_intercept_sd':float(np.exp(result['beta'][-1])) if kind=='RI' else None,'outside_component_weight':float(expit(result['beta'][-1])) if kind=='spike' else None,'beta':result['beta']})
        diagnostics.append({'condition':condition,'seed':seed,'fold':int(fold),'model':kind,**result['checks']})
    for fold in np.unique(ff):
        train=ff!=fold
        for kind in ['MNL','RI','spike']:
            rr=fit(x[train],y[train],kind);original[(kind,int(fold))]=rr;evaluate('human_original',-1,fold,y[train],rr,kind)
    settings=[('human_rank_aligned',-1)]+[('human_random_repaired',s) for s in range(202611090,202611100)]
    for condition,seed in settings:
        for fold in np.unique(ff):
            train=ff!=fold;yt=y[train].copy();vt=v[train];rng=np.random.default_rng(max(seed,0)+int(fold)*100000)
            for version in [0,1]:
                use=np.where(vt==version)[0]
                for t in range(8):
                    before=yt[use,t].copy()
                    yt[use,t]=np.sort(before) if condition=='human_rank_aligned' else rng.permutation(before)
                    assert np.array_equal(np.bincount(before,minlength=3),np.bincount(yt[use,t],minlength=3))
            rr=original[('MNL',int(fold))];ov,og=Panel(x[train],y[train],'MNL').eval(rr['beta']);nv,ng=Panel(x[train],yt,'MNL').eval(rr['beta'])
            error=abs(ov-nv);ge=float(max(abs(og-ng)));assert error<1e-8 and ge<1e-8
            controls.append({'condition':condition,'seed':seed,'fold':int(fold),'fixed_margins':True,'MNL_objective_error':error,'MNL_gradient_error':ge})
            evaluate(condition,seed,fold,yt,rr,'MNL')
            for kind in ['RI','spike']:
                rr=fit(x[train],yt,kind);evaluate(condition,seed,fold,yt,rr,kind)
                if kind=='RI' and (condition=='human_rank_aligned' or seed==202611090):
                    delta=Panel(x[train],yt,'RI',161).eval(rr['beta'])[0]-Panel(x[train],yt,'RI',61).eval(rr['beta'])[0]
                    higher.append({'condition':condition,'seed':seed,'fold':int(fold),'loglik_difference_61_vs_161':float(delta)})
        pd.DataFrame(rows).to_csv(a.out/'human_negative_control_fold_scores.csv',index=False)
        save(a.out/'human_negative_control_parameters.json',params);save(a.out/'human_negative_control_fit_checks.json',diagnostics)
        print(condition,seed,'complete',flush=True)
    out=pd.DataFrame(rows).groupby(['condition','seed','model'],as_index=False).agg(records=('human_heldout_records','sum'),joint_logscore=('joint_logscore_sum','sum'),marginal_logscore=('marginal_logscore_sum','sum'),predicted_all_outside=('predicted_all_outside','sum'))
    assert out.records.eq(300).all();out['joint_nll_per_choice']=-out.joint_logscore/2400;out['marginal_nll_per_choice']=-out.marginal_logscore/2400
    out.to_csv(a.out/'human_negative_control_summary.csv',index=False);pd.DataFrame(controls).to_csv(a.out/'human_negative_control_invariance.csv',index=False)
    save(a.out/'human_negative_control_checks.json',{'human_records':300,'AI_records_used':0,'original_raw_hashes':{z['version']:z['sha256'] for z in provenance},'fixed_marginal_controls':len(controls),'maximum_MNL_objective_error':max(z['MNL_objective_error'] for z in controls),'maximum_MNL_gradient_error':max(z['MNL_gradient_error'] for z in controls),'higher_quadrature_checks':higher,'raw_records_changed':False,'controls_presented_as_observations':False,'formal_permutation_test':False})
    print(out.groupby(['condition','model']).joint_nll_per_choice.agg(['mean','min','max']).to_string(),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
