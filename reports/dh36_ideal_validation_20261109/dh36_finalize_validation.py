"""High-precision checks, panel boundary repair and DGP-calibrated power.
Run after dh36_assumption_conditioned_validation.py; never edits raw exports.
"""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import chi2
from threadpoolctl import threadpool_limits
import dh36_observational_analysis as old
from dh36_assumption_conditioned_validation import (SEED,js,interval,perm_stats,random_assignments,permutation_suite,copula_y)
from dh36_panel_model import fit_fixed_price,fixed_price_eval


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input-dir',type=Path,required=True); ap.add_argument('--instrument',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--permutations',type=int,default=999999); ap.add_argument('--null-repetitions',type=int,default=5000); ap.add_argument('--null-permutations',type=int,default=999); ap.add_argument('--power-repetitions',type=int,default=1000); args=ap.parse_args(); out=args.out
    spec=json.loads(args.instrument.read_text()); d,*_=old.read_exports(args.input_dir,spec)
    y=d[[f'q{t}' for t in range(1,9)]].to_numpy(int)-1; z=d.version2.to_numpy(int); base=old.menu_arrays(spec)
    tests,_=permutation_suite(y,z,args.permutations,SEED); tests.to_csv(out/'randomization_tests_high_precision.csv',index=False)
    print('million permutations',tests[['test','permutation_p_plus_one']].to_dict('records'),flush=True)
    fits={}; rows=[]; checks={}
    for interaction in [False,True]:
        nm='random_intercept_version_ASC' if interaction else 'random_intercept_common'
        low=fit_fixed_price(y,z,base,nodes=31,interaction=interaction); fit=fit_fixed_price(y,z,base,nodes=101,interaction=interaction)
        high=fit_fixed_price(y,z,base,nodes=161,interaction=interaction)
        fits[nm]=fit
        for row in fit['parameters']: rows.append({'model':nm,**row})
        checks[nm]={'nodes31':low['validation'],'nodes101':fit['validation'],'nodes161':high['validation'],'loglik31':low['loglik'],'loglik101':fit['loglik'],'loglik161':high['loglik'],'max_coefficient_change101_to161':float(max(abs(fit['beta']-high['beta'])))}
        assert max(abs(fit['beta']-high['beta']))<1e-4 and fit['validation']['gradient_max']<1e-3 and fit['validation']['hessian_min_eigenvalue']>0
    f=fits['random_intercept_version_ASC']; delta=f['beta'][6:8]; cov=f['cov'][6:8,6:8]; w=float(delta@np.linalg.inv(cov)@delta)
    js(out/'random_intercept_model_validation.json',{'models':checks,'version_ASC_model_conditional_Wald':{'W':w,'df':2,'p':float(chi2.sf(w,2)),'scope':'restricted random-intercept logit; not the unrestricted sufficiency null'},'variance_boundary_repair':'Lognormal price SD hit lower bound in expanded mixed logit; final model fixes its variance to zero and estimates only service-intercept heterogeneity.'})
    pd.DataFrame(rows).to_csv(out/'random_intercept_model_coefficients.csv',index=False)
    _,folds=old.predictive_validation(d,base,SEED)
    vals={nm:np.zeros(len(y)) for nm in ['MNL_product',*fits]}; cvchecks=[]
    for fold in range(5):
        tr=folds!=fold; te=~tr
        f=old.fit_mnl(d.loc[tr],base,'mnl_common'); x,_=old.design(d.loc[te],base,'mnl_common'); u=x@f['beta']; lp=(u-logsumexp(u,axis=1,keepdims=True)).reshape(te.sum(),8,3)
        vals['MNL_product'][te]=np.take_along_axis(lp,y[te,:,None],axis=2)[:,:,0].sum(1)
        for interaction in [False,True]:
            nm='random_intercept_version_ASC' if interaction else 'random_intercept_common'
            f=fit_fixed_price(y[tr],z[tr],base,nodes=61,interaction=interaction)
            assert f['validation']['gradient_max']<1e-3 and f['validation']['solver_success']
            vals[nm][te]=fixed_price_eval(f['beta'],y[te],z[te],base,161,interaction,True)[3]
            cvchecks.append({'fold':fold,'model':nm,**f['validation'],'training_n':int(tr.sum()),'validation_n':int(te.sum()),'no_heldout_outcomes_in_fit_or_initialization':True})
        print('independent-start CV',fold,flush=True)
    rows=[]
    for nm,val in vals.items():
        diff=(val-vals['MNL_product'])/8; se=diff.std(ddof=1)/np.sqrt(len(y))
        gain2=(val-vals['random_intercept_common'])/8; se2=gain2.std(ddof=1)/np.sqrt(len(y))
        rows.append({'model':nm,'joint_NLL_per_record':-val.mean(),'joint_NLL_divided_by_8':-val.mean()/8,'gain_vs_MNL_per_task':diff.mean(),'descriptive_gain_ci_low':diff.mean()-1.96*se,'descriptive_gain_ci_high':diff.mean()+1.96*se,'gain_vs_random_intercept_common':gain2.mean(),'descriptive_gain_vs_RI_ci_low':gain2.mean()-1.96*se2,'descriptive_gain_vs_RI_ci_high':gain2.mean()+1.96*se2,'folds':5,'target':'joint 8-response vector for new record; fixed multistart from training data only'})
    pd.DataFrame(rows).to_csv(out/'independent_start_panel_joint_CV.csv',index=False); js(out/'panel_CV_convergence_checks.json',cvchecks)
    # Extend every predeclared scenario, preserving the initial 500-run output.
    p=(np.eye(3)[y].sum(0)+.5)/(len(y)+1.5); rng=np.random.default_rng(SEED+100); rows=[]
    for scenario,r0,r1 in [('joint_null_independent',0.,0.),('joint_null_correlated',.5,.5),('weak_null_unequal_dependence',.05,.8)]:
        rej=np.zeros(4,int); B=args.null_permutations; R=args.null_repetitions
        for rep in range(R):
            yy=np.vstack([copula_y(rng,p,180,r0),copula_y(rng,p,180,r1)]); zz=np.r_[np.zeros(180),np.ones(180)]
            obs=perm_stats(yy,zz,zz)[0]; pp=perm_stats(yy,zz,random_assignments(rng,B,360)); pv=(1+(pp>=obs-1e-10).sum(0))/(B+1)
            rej[:3]+=pv<=.05; rej[3]+=chi2.sf(obs[0],16)<=.05
            if (rep+1)%1000==0: print('extended null',scenario,rep+1,flush=True)
        for j,nm in enumerate(['studentized_Wald_permutation','pooled_score_permutation','saturated_composite_LR_permutation','studentized_Wald_chi16']):
            ci=interval(rej[j],R); rows.append({'scenario':scenario,'test':nm,'n':360,'MC_repetitions':R,'permutations_per_rep':B if j<3 else 0,'rejection_rate':rej[j]/R,'mc_low':ci[0],'mc_high':ci[1],'scope':'joint null' if r0==r1 else ('weak marginal null; studentized or balanced-equivalent score' if j!=2 else 'joint null false; no weak-null validity claim')})
        pd.DataFrame(rows).to_csv(out/'null_calibration_extended.csv',index=False)
    # Independent simulated-null cutoffs avoid inflated chi-square power.
    # This is DGP-calibrated power, not a conditional permutation probability.
    rng=np.random.default_rng(SEED+101); cutoffs={}; critrows=[]
    for n in [360,600,800,1200]:
        zz=np.r_[np.zeros(n//2),np.ones(n//2)]; ww=[]
        for rep in range(5000):
            yy=copula_y(rng,p,n,.5); ww.append(perm_stats(yy,zz,zz)[0,0])
        cutoffs[n]=float(np.quantile(ww,.95,method='higher')); critrows.append({'n':n,'null_MC':5000,'rho':.5,'critical_Wald_95':cutoffs[n]})
    pd.DataFrame(critrows).to_csv(out/'power_null_critical_values.csv',index=False)
    rows=[]
    for direction in ['all_tasks_A_vs_B','one_task_A_vs_B','all_tasks_service_vs_optout']:
        for n in [360,600,800,1200]:
            zz=np.r_[np.zeros(n//2),np.ones(n//2)]
            for delta in [0.,.03,.05,.08,.10]:
                change=np.zeros_like(p)
                if direction=='all_tasks_A_vs_B': change[:,0]=1; change[:,1]=-1
                elif direction=='one_task_A_vs_B': change[1,0]=1; change[1,1]=-1
                else: change[:,0]=.5; change[:,1]=.5; change[:,2]=-1
                p0=p-delta/2*change; p1=p+delta/2*change; k=0
                for rep in range(args.power_repetitions):
                    yy=np.vstack([copula_y(rng,p0,n//2,.5),copula_y(rng,p1,n//2,.5)])
                    k+=perm_stats(yy,zz,zz)[0,0]>cutoffs[n]
                ci=interval(k,args.power_repetitions); rows.append({'direction':direction,'n':n,'max_probability_difference':delta,'rho_latent_copula':.5,'repetitions':args.power_repetitions,'power_or_size':k/args.power_repetitions,'mc_low':ci[0],'mc_high':ci[1],'calibration':'independent 5000-null-MC 95% cutoff; hypothetical balanced complete assignment'})
        pd.DataFrame(rows).to_csv(out/'power_DGP_calibrated.csv',index=False); print('calibrated power',direction,flush=True)
    js(out/'finalization_record.json',{'high_precision_permutations':args.permutations,'null_MC_per_scenario':args.null_repetitions,'null_permutations_per_rep':args.null_permutations,'power_MC_per_cell':args.power_repetitions,'power_null_cutoff_MC_per_n':5000,'raw_hashes_unchanged':all(hashlib.sha256((args.input_dir/f'{v}_raw.xlsx').read_bytes()).hexdigest()==json.loads((out/'technical_verification.json').read_text())['raw_sha256'][v] for v in ['V1','V2']),'full_record_population':360,'fixed_price_panel_model':'six or eight estimated parameters; random price variance set exactly to zero','initial_results_preserved':True,'complete':True})
    print('FINAL NUMERICAL VALIDATION COMPLETE',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1): main()
