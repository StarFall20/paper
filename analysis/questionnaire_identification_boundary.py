"""Exhaustive direct menu contrasts and classical context-sensitivity bounds."""
from pathlib import Path
import argparse,json,itertools
import numpy as np,pandas as pd
from scipy.stats import binomtest,binom,norm
from scipy.optimize import linprog
from threadpoolctl import threadpool_limits
import dh36_observational_analysis as old
from application_support_models import build,fit_ab,save


def ratio_interval(N,D,cov):
    z=norm.ppf(.975);vn,vd,nd=cov[0,0],cov[1,1],cov[0,1]
    aa=D*D-z*z*vd;bb=-2*N*D+2*z*z*nd;cc=N*N-z*z*vn;disc=bb*bb-4*aa*cc
    bounded=aa>0 and disc>=0
    roots=sorted([(-bb-np.sqrt(disc))/(2*aa),(-bb+np.sqrt(disc))/(2*aa)]) if bounded else [None,None]
    return {'net_outside_fraction':float(N/D) if D else None,'fieller_bounded':bool(bounded),'fieller_low':roots[0],'fieller_high':roots[1]}


def enumerate_menus(spec):
    rows=[]
    for t,s in itertools.combinations(range(8),2):
        a,b=spec['tasks'][t],spec['tasks'][s]
        changes=[(j,k) for j in range(2) for k in ['sharing','prices'] if a[k][j]!=b[k][j]]
        unchanged=[j for j in range(2) if a['sharing'][j]==b['sharing'][j] and a['prices'][j]==b['prices'][j]]
        rows.append({'first_task':t+1,'second_task':s+1,'changed_fields':','.join(f'{"AB"[j]}_{k}' for j,k in changes),'unchanged_service':','.join('AB'[j] for j in unchanged),'single_price_change':len(changes)==1 and changes[0][1]=='prices','single_rule_change':len(changes)==1 and changes[0][1]=='sharing','unchanged_service_index':unchanged[0] if len(unchanged)==1 else -1})
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input-dir',type=Path,required=True);ap.add_argument('--instrument',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out;out.mkdir(parents=True,exist_ok=True)
    spec=json.loads(a.instrument.read_text());d,provenance,*_=old.read_exports(a.input_dir,spec);y=d[[f'q{i}' for i in range(1,9)]].to_numpy(int)-1;models,names,*_=build(spec,d)
    menus=enumerate_menus(spec);menus.to_csv(out/'all_28_menu_pairs.csv',index=False)
    eligible=menus[menus.unchanged_service_index>=0];prices=menus[menus.single_price_change]
    assert len(eligible)==2 and len(prices)==1 and not menus.single_rule_change.any()
    populations=[('all_360',np.ones(len(d),bool)),('300_other_source_labels',~d.ai_label.to_numpy()),('V1_180',d.version.eq('V1').to_numpy()),('V2_180',d.version.eq('V2').to_numpy())]
    sw=[];flows=[];shares=[];ratios=[];context=[];contrast=[];power=[];direct=[];audit={}
    for population,use in populations:
        yy=y[use];n=len(yy)
        for r in eligible.itertuples():
            t,s=r.first_task-1,r.second_task-1;j=r.unchanged_service_index
            forward=int(np.sum((yy[:,t]==j)&(yy[:,s]==2)));backward=int(np.sum((yy[:,t]==2)&(yy[:,s]==j)));m=forward+backward
            result=binomtest(forward,m,.5);ci=result.proportion_ci(method='exact')
            sw.append({'population':population,'first_task':t+1,'second_task':s+1,'unchanged_service':'AB'[j],'service_to_outside':forward,'outside_to_service':backward,'discordant_path_records':m,'records_retained':n,'conditional_forward_fraction':forward/m,'exact_ci_low':ci.low,'exact_ci_high':ci.high,'exact_p':result.pvalue,'pooled_Bonferroni_2':min(1,2*result.pvalue) if population=='all_360' else None})
            if population=='all_360':
                reject=np.array([binomtest(k,m,.5).pvalue<=.025 for k in range(m+1)])
                for theta in [.55,.60,.65,.70]:power.append({'first_task':t+1,'second_task':s+1,'conditional_discordant_records':m,'fixed_alternative_forward_probability':theta,'two_sided_test_alpha':.025,'exact_conditional_power':float(binom.pmf(np.arange(m+1),m,theta)[reject].sum()),'scope':'illustrative fixed alternatives at observed discordant count; not observed-effect power or a full-study sample-size recommendation'})
        for r in prices.itertuples():
            t,s=r.first_task-1,r.second_task-1
            which=next(j for j in range(2) if spec['tasks'][t]['prices'][j]!=spec['tasks'][s]['prices'][j]);assert which==0
            delta=np.eye(3)[yy[:,s]]-np.eye(3)[yy[:,t]];mean=delta.mean(0);cov=np.cov(delta,rowvar=False,ddof=1)/n
            for j,label in enumerate(['A','B','outside']):
                se=np.sqrt(cov[j,j]);shares.append({'population':population,'first_task':t+1,'second_task':s+1,'alternative':label,'first_share':float(np.mean(yy[:,t]==j)),'second_share':float(np.mean(yy[:,s]==j)),'difference':mean[j],'paired_se':se,'ci_low':mean[j]-1.96*se,'ci_high':mean[j]+1.96*se})
                for k,label2 in enumerate(['A','B','outside']):flows.append({'population':population,'first_choice':label,'second_choice':label2,'count':int(np.sum((yy[:,t]==j)&(yy[:,s]==k)))})
            vals=np.column_stack([delta[:,2],-delta[:,0]]);cv=np.cov(vals,rowvar=False,ddof=1)/n;N,D=vals.mean(0)
            margin=delta[:,2]-delta[:,1];mu=margin.mean();se=margin.std(ddof=1)/np.sqrt(n)
            ratios.append({'population':population,'net_outside_gain':N,'net_A_loss':D,**ratio_interval(N,D,cv),'outside_minus_B_gain':mu,'margin_ci_low':mu-1.96*se,'margin_ci_high':mu+1.96*se,'point_drift_bound_to_erase_margin':abs(mu)/2,'drift_bound_to_erase_positive_lower_interval':max(0,mu-1.96*se)/2})
            for eta in [0.,.005,.01,.025,.05,.075,.1]:
                # Unknown probability drift h sums to zero; |h_j| <= eta.
                # c=(0,-1,1), max c'h = 2 eta. LP independently verifies it.
                c=np.array([0.,-1.,1.]);lp=linprog(-c,A_eq=np.ones((1,3)),b_eq=[0.],bounds=[(-eta,eta)]*3,method='highs');assert lp.success and abs(-lp.fun-2*eta)<1e-9
                context.append({'population':population,'max_absolute_probability_drift_per_option':eta,'worst_case_outside_minus_B_gain':mu-2*eta,'worst_case_lower_95_interval':mu-1.96*se-2*eta,'interpretation':'hypothetical fixed bound, not measured fatigue; no causal task-order identification'})
    # Exact contrast weights for conditional A/B observed-cell utilities.
    idx=[0]+list(range(4,11));x=models['cells'][0];DD=(x[:,0]-x[:,1])[:,idx];cc=np.zeros(len(idx));cc[idx.index(names.index('G2_fee_89'))]=1;w=np.linalg.solve(DD.T,cc)
    assert np.max(abs(DD.T@w-cc))<1e-12 and np.linalg.matrix_rank(DD)==8
    z=norm.ppf(.975)
    for population,use in populations+[('V1_other_source_150',d.version.eq('V1').to_numpy()&~d.ai_label.to_numpy()),('V2_other_source_150',d.version.eq('V2').to_numpy()&~d.ai_label.to_numpy())]:
        yy=y[use];n=len(yy);pa=np.mean(yy==0,axis=0);pb=np.mean(yy==1,axis=0);assert min(pa)>0 and min(pb)>0
        e=np.log(pa/pb);influence=(yy==0)/pa-(yy==1)/pb;person=influence@w;se=person.std(ddof=1)/np.sqrt(n);val=float(w@e)
        direct.append({'population':population,'records':n,'observed_conditional_logodds_contrast':val,'paired_delta_se':se,'ci_low':val-z*se,'ci_high':val+z*se,'Linf_task_logodds_offset_to_erase_point':abs(val)/sum(abs(w)),'scope':'observable summary; utility-package interpretation requires the stated cross-task index assumptions; not a model-free causal contrast'})
    for population,use in populations[:2]:
        b,cov,columns,check=fit_ab(models['cells'][use],y[use]);cn=[names[j] for j in columns];c=np.zeros(len(b));c[cn.index('G2_fee_89')]=1;effect=float(c@b);se=float(np.sqrt(c@cov@c));l1=float(abs(w).sum())
        for eps in [0.,.01,.025,.05,.075,.1]:
            lp=linprog(w,bounds=[(-eps,eps)]*8,method='highs');assert lp.success and abs(lp.fun+l1*eps)<1e-9
            contrast.append({'population':population,'max_absolute_task_logodds_offset':eps,'conditional_model_effect':effect,'structural_lower':effect-l1*eps,'structural_upper':effect+l1*eps,'sampling_and_offset_lower':effect-z*se-l1*eps,'sampling_and_offset_upper':effect+z*se+l1*eps,'offset_bound_to_erase_positive_point':abs(effect)/l1,'scope':'classical sensitivity under an assumed bound; not an estimate of task bias or a causal rule effect'})
        audit[population+'_conditional_fit']=check
    # Algebra check under arbitrary stable heterogeneous utilities and arbitrary
    # utilities of the changed third alternative; no fabricated respondent data.
    rng=np.random.default_rng(20261109);u=rng.normal(size=(2000,2));changed=rng.normal(size=(2000,2));den0=np.exp(u).sum(1)+np.exp(changed[:,0]);den1=np.exp(u).sum(1)+np.exp(changed[:,1]);p01=np.exp(u[:,0]+u[:,1])/(den0*den1);p10=np.exp(u[:,1]+u[:,0])/(den0*den1)
    assert np.max(abs(p01-p10))<1e-14
    audit.update({'menu_pair_enumeration_total':len(menus),'eligible_unchanged_service_pairs':len(eligible),'single_price_change_pairs':int(menus.single_price_change.sum()),'single_rule_change_pairs':int(menus.single_rule_change.sum()),'conditional_cell_within_version_design_rows':8,'conditional_cell_within_version_design_rank':8,'within_version_conditional_cell_lack_of_fit_df':0,'G2_G1_at_89_logodds_weights':w,'weights_L1_norm':float(abs(w).sum()),'classical_box_bound_LP_checks_passed':True,'panel_logit_path_identity_numerical_max_error':float(max(abs(p01-p10))),'source_hashes':[{'version':p['version'],'sha256':p['sha256']} for p in provenance],'raw_records_or_identifiers_exported':False,'no_new_theory_claim':True,'no_preregistration_or_exact_randomization_claim':True})
    save(out/'boundary_numerical_audit.json',audit)
    for name,rows in [('conditional_path_swap_tests',sw),('paired_choice_shares',shares),('paired_transition_counts',flows),('net_share_decomposition',ratios),('hypothetical_order_drift_sensitivity',context),('conditional_rule_offset_sensitivity',contrast),('illustrative_exact_conditional_power',power),('direct_task_logodds_contrasts',direct)]:pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
    print(pd.DataFrame(ratios).to_string(index=False));print(pd.DataFrame(sw).to_string(index=False));print(json.dumps({'contrast_weights':w.tolist(),'conditional_within_version_rank':8,'lack_of_fit_df':0},ensure_ascii=False))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
