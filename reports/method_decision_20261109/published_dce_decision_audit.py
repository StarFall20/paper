"""Published-support and classical-baseline audit for a research go/no-go decision.

Inputs are public SP archives. Only displayed attributes enter this audit; source
choices and respondent covariates are not used for a behavioral rejection claim.
Candidate reductions and generated probes are analyst-defined sensitivity cases,
not claims about the original authors' hypotheses or experimental assignments.
"""
from __future__ import annotations
import argparse,hashlib,json,math
from collections import defaultdict
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax,xlogy
from scipy.stats import norm
from threadpoolctl import threadpool_limits


def save(path,obj):
    def convert(x):
        if isinstance(x,np.ndarray): return x.tolist()
        if isinstance(x,np.generic): return x.item()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=convert,allow_nan=False)+'\n')


def read_route(path):
    df=pd.read_csv(path,dtype=str); cols=['tt1','tc1','hw1','ch1','tt2','tc2','hw2','ch2']
    menus=[np.array([F(v) for v in row],object).reshape(2,4) for row in df[cols].itertuples(index=False,name=None)]
    assert len(menus)==3492 and df.ID.nunique()==388
    return menus,len(df),int(df.ID.nunique())


def read_vehicle(path):
    df=pd.read_csv(path,sep=r'\s+',header=None,dtype=str); menus=[]
    for key,g in df.groupby(1,sort=False):
        assert len(g)==3
        menus.append(np.array([[F(v) for v in row] for row in g.iloc[:,3:].itertuples(index=False,name=None)],object))
    assert len(menus)==1484 and df[0].nunique()==100
    return menus,len(menus),int(df[0].nunique())


def rawkey(m): return tuple(m.ravel())


def candidate(m,kind,parameter):
    if kind=='route': return np.array([[r[1]+F(parameter,60)*r[0],r[2],r[3]] for r in m],object)
    return np.array([[r[0]+parameter*r[1],*r[2:]] for r in m],object)


def key_phi(m,kind,parameter,representation):
    b=candidate(m,kind,parameter)
    if representation=='utility_difference_vector': b=b[:-1]-b[-1]
    return tuple(b.ravel())


def model_arrays(menus,kind,parameter):
    b=np.array([candidate(m,kind,parameter) for m in menus],float); raw=np.array(menus,float); n,J,_=raw.shape
    if kind=='route':
        attrs=np.stack([b[:,:,0]/20,b[:,:,1]/30,b[:,:,2]],axis=2)
        asc=np.zeros((n,J,1)); asc[:,0,0]=1; x=np.concatenate([asc,attrs],axis=2)
        beta=np.array([0.,-.3,-.2,-.3]); t=raw[:,:,0]/60; c=raw[:,:,1]/20
        h=np.stack([t,t*t,t*c],axis=2)
    else:
        attrs=np.stack([b[:,:,0]/50000,raw[:,:,2],raw[:,:,3],raw[:,:,5],raw[:,:,6],raw[:,:,7]],axis=2)
        asc=np.zeros((n,J,2)); asc[:,0,0]=1; asc[:,1,1]=1; x=np.concatenate([asc,attrs],axis=2)
        beta=np.array([0.,0.,-1.,.1,.05,.1,.2,.1]); p=raw[:,:,0]/50000; o=raw[:,:,1]/50
        h=np.stack([p,p*p,p*o],axis=2)
    pr=softmax(x@beta,axis=1); v=np.array([np.diag(p[:-1])-np.outer(p[:-1],p[:-1]) for p in pr])
    hd=h[:,:-1]-h[:,-1:, :]
    return x,pr,v,hd,h


def two_information_methods(h,v,groups,w):
    """Independent within-fibre centering vs classical nuisance Schur complement."""
    k=h.shape[-1]; centered=np.zeros((k,k)); full=np.einsum('n,ndk,nde,nel->kl',w,h,v,h,optimize=True)
    for ids in groups:
        mass=w[ids].sum()
        if mass==0: continue
        # p0 and its categorical metric are functions of the candidate map.
        assert np.max(abs(v[ids]-v[ids[0]]))<1e-12
        mean=np.einsum('n,ndk->dk',w[ids]/mass,h[ids])
        hc=h[ids]-mean
        centered+=np.einsum('n,ndk,nde,nel->kl',w[ids],hc,v[ids],hc,optimize=True)
        A=mass*v[ids[0]]
        B=np.einsum('n,nde,nek->dk',w[ids],v[ids],h[ids],optimize=True)
        full-=B.T@np.linalg.solve(A,B)
    return (centered+centered.T)/2,(full+full.T)/2


def audit(menus,kind,parameter,representation):
    unique={}; frequency=defaultdict(int)
    for m in menus: unique.setdefault(rawkey(m),m); frequency[rawkey(m)]+=1
    mm=list(unique.values()); w=np.array([frequency[rawkey(m)] for m in mm],float); w/=w.sum()
    lookup=defaultdict(list)
    for i,m in enumerate(mm): lookup[key_phi(m,kind,parameter,representation)].append(i)
    groups=[np.array(ix,int) for ix in lookup.values()]
    x,pr,v,h,_=model_arrays(mm,kind,parameter)
    a,b=two_information_methods(h,v,groups,w)
    err=float(np.max(abs(a-b))); rel=err/max(1.,float(np.max(abs(a)))); assert rel<1e-9
    # Saturated profile contrasts: numerical ranks per fibre, independent of H.
    blockrank=0; blocks=0
    for ids in groups:
        if len(ids)<2: continue
        q=w[ids]/w[ids].sum(); C=np.diag(q)-np.outer(q,q)
        er=np.linalg.eigvalsh(C); rank=int((er>1e-10).sum())
        assert rank==len(ids)-1
        blockrank+=rank; blocks+=1
    formula=len(mm)-len(groups); assert blockrank==formula
    J=len(mm[0]); eig=np.linalg.eigvalsh(a)
    return {'dataset':kind,'parameter':parameter,'representation':representation,'observed_choice_sets':len(menus),'unique_raw_menus':len(mm),'candidate_fibres':len(groups),'non_singleton_fibres':blocks,'menu_contrast_rank':formula,'categorical_saturated_contrast_df':formula*(J-1),'numeric_block_rank':blockrank,'declared_three_direction_rank':int((eig>1e-10).sum()),'declared_information_min_eigenvalue':float(eig.min()),'classical_Schur_max_difference':err,'metric_min_eigenvalue_over_menus':float(np.min(np.linalg.eigvalsh(v))),'choice_outcomes_used':False},a,b


def generate_pairs(menus,kind,parameter,count=16):
    raw=np.array(menus,float); lower=raw.reshape(-1,raw.shape[-1]).min(0); upper=raw.reshape(-1,raw.shape[-1]).max(0)
    selected=[]; seen=set(); original=set(map(rawkey,menus))
    # One declared A-arm change: +6 minutes / -1 CHF at 10 CHF/hour;
    # +1200 dollars upfront / -20 dollars per month at a 60-month horizon.
    for m in menus:
        key=key_phi(m,kind,parameter,'absolute_menu_summary')
        if key in seen: continue
        probe=m.copy()
        if kind=='route':
            probe[0,0]+=6; probe[0,1]-=F(parameter,10)
        else:
            probe[0,0]+=1200; probe[0,1]-=F(1200,parameter)
        f=np.array(probe,float)
        if np.any(f<lower-1e-10) or np.any(f>upper+1e-10): continue
        if rawkey(probe) in original: continue
        assert candidate(m,kind,parameter).tolist()==candidate(probe,kind,parameter).tolist()
        selected.append((m,probe)); seen.add(key)
        if len(selected)==count: break
    assert len(selected)==count
    return selected,{'support_model':'attribute ticks and global marginal min/max from the archive; other attributes fixed','original_joint_design_constraints_recovered':False,'respondent_plausibility_checked':False,'type':'generated diagnostic probes, never original field tasks','pairs':len(selected)}


def compact_info(x,pr,w):
    V=np.array([np.diag(p)-np.outer(p,p) for p in pr])
    return np.einsum('n,njk,njl,nlm->km',w,x,V,x,optimize=True)


def repair_comparison(menus,kind,parameter):
    pairs,scope=generate_pairs(menus,kind,parameter); mm=[m for pair in pairs for m in pair]; groups=[np.array([2*i,2*i+1]) for i in range(len(pairs))]
    x,p,v,h,hh=model_arrays(mm,kind,parameter); m=len(pairs); rows=[]
    reference=None; best=None
    for q in [0.,.1,.25,.5,.75,.9,1.]:
        w=np.tile([1-q,q],m)/m
        f,c=two_information_methods(h,v,groups,w)
        ni=compact_info(x,p,w)
        if reference is None: reference=ni
        err=float(max(abs(ni-reference).ravel())); assert err<1e-12
        delta=np.max(abs(f-c)); assert delta<1e-9
        eig=np.linalg.eigvalsh(f); sign,logdet=np.linalg.slogdet(ni)
        row={'dataset':kind,'parameter':parameter,'generated_pairs':m,'raw_probe_assignment_probability':q,'fixed_choice_opportunity_budget':320,'choice_opportunities_per_fibre':20,'new_questionnaire_sample_recommendation':False,'within_fibre_metric_rank':int(np.linalg.matrix_rank(f,tol=1e-10)),'fibre_E_information':float(eig.min()),'classical_residual_E_information':float(np.linalg.eigvalsh(c).min()),'fibre_information_trace':float(np.trace(f)),'null_parameter_information_rank':int(np.linalg.matrix_rank(ni)),'null_information_logdet':float(logdet) if sign>0 else None,'null_information_max_change':err,'matrix_equivalence_max_error':float(delta)}
        rows.append(row)
        if q==.5: best=f
    # At fixed two-point fibres, q(1-q) is simultaneously maximized at 1/2.
    for row in rows:
        q=row['raw_probe_assignment_probability']; w=np.tile([1-q,q],m)/m
        f,_=two_information_methods(h,v,groups,w)
        assert np.linalg.eigvalsh(best-f).min()>-1e-10
    probe_rows=[]
    for t,(base,probe) in enumerate(pairs):
        for orientation,menu in enumerate([base,probe]):
            for j,alt in enumerate(menu):
                probe_rows.append({'dataset':kind,'pair':t+1,'generated_orientation':orientation,'alternative_position':j+1,'raw_attributes':';'.join(str(f) for f in alt),'candidate_attributes':';'.join(str(f) for f in candidate(menu,kind,parameter)[j]),'scope':'generated diagnostic; not human validated'})
    return pd.DataFrame(rows),pd.DataFrame(probe_rows),scope,pairs


def test_statistics(counts0,counts1):
    """Full profile score and CMH computed independently with fixed margins."""
    counts0=np.atleast_3d(counts0) if counts0.ndim<2 else counts0
    c0=np.atleast_3d(counts0) if counts0.ndim<2 else counts0
    # Inputs either m x J or B x m x J. Normalize to batch.
    if c0.ndim==2: c0=c0[None,:,:]
    c1=counts1[None,:,:] if counts1.ndim==2 else counts1
    B,m,J=c0.shape; fs=np.zeros(B); cmh=np.zeros(B); lr=np.zeros(B)
    for t in range(m):
        a=c0[:,t,:]; b=c1[:,t,:]; total=a+b; n0=a.sum(1); n1=b.sum(1); n=n0+n1; p=total/n[:,None]
        V=np.array([np.diag(pp[:-1])-np.outer(pp[:-1],pp[:-1]) for pp in p])
        inv=np.linalg.pinv(V,hermitian=True)
        dp=b[:,:-1]/n1[:,None]-a[:,:-1]/n0[:,None]
        factor=n/(n-1)*(1/n0+1/n1)
        fs+=np.einsum('bi,bij,bj->b',dp,inv,dp)/factor
        centered=b[:,:-1]-n1[:,None]*p[:,:-1]
        cmh+=np.einsum('bi,bij,bj->b',centered,inv,centered)/(n0*n1/(n-1))
        e0=p*n0[:,None]; e1=p*n1[:,None]
        lr+=2*(xlogy(a,np.divide(a,e0,out=np.ones_like(p),where=e0>0))+xlogy(b,np.divide(b,e1,out=np.ones_like(p),where=e1>0))).sum(1)
    return np.column_stack([fs,cmh,lr])


def comparator_simulation(pairs,kind,parameter,reps=500,B=499,seed=20261109):
    # A controlled design benchmark: one choice per simulated record.
    # No inference about repeated human records or new field sample size.
    pairs=pairs[:8]; mm=[x for pair in pairs for x in pair]; x,p,v,h,hu=model_arrays(mm,kind,parameter)
    m=len(pairs); theta=np.array([1.,-.2,.3]); U=x@np.array([0.,-.3,-.2,-.3] if kind=='route' else [0.,0.,-1.,.1,.05,.1,.2,.1])
    signal=hu@theta; signal=signal.reshape(m,2,-1); signal-=signal.mean(1,keepdims=True)
    variance=np.mean(np.sum(p.reshape(m,2,-1)*signal**2,axis=2)-np.sum(p.reshape(m,2,-1)*signal,axis=2)**2)
    assert variance>0; signal/=np.sqrt(variance); U=U.reshape(m,2,-1)
    rng=np.random.default_rng(seed); records=[]; maxerr=0.; maxpdiff=0.
    for n in [200,400,800]:
        nper=n//m; n0=nper//2; n1=nper-n0
        for eta in [0.,.1,.2]:
            probs=softmax(U+eta*signal,axis=2); reject=np.zeros(3,int); mismatches=0
            for rep in range(reps):
                c0=np.array([rng.multinomial(n0,probs[t,0]) for t in range(m)])
                c1=np.array([rng.multinomial(n1,probs[t,1]) for t in range(m)])
                obs=test_statistics(c0,c1)[0]; total=c0+c1
                draws=np.stack([rng.multivariate_hypergeometric(total[t],n1,size=B) for t in range(m)],axis=1)
                perm=test_statistics(total[None,:,:]-draws,draws)
                pv=(1+(perm>=obs-1e-10).sum(0))/(B+1)
                reject+=pv<=.05; mismatches+=int(abs(pv[0]-pv[1])>1e-15)
                maxerr=max(maxerr,float(np.max(abs(perm[:,0]-perm[:,1])))); maxpdiff=max(maxpdiff,float(abs(pv[0]-pv[1])))
            for j,name in enumerate(['fibre_full_profile_score','classical_full_profile_CMH','saturated_profile_LR']):
                phat=reject[j]/reps; z=norm.ppf(.975); den=1+z*z/reps; center=(phat+z*z/(2*reps))/den; half=z*np.sqrt(phat*(1-phat)/reps+z*z/(4*reps*reps))/den
                records.append({'dataset_source':kind,'n_independent_single_choice_records':n,'eta_RMS_centered_utility_departure':eta,'replications':reps,'conditional_margin_draws':B,'method':name,'rejection_rate':phat,'mc_low':center-half,'mc_high':center+half,'framework_CMH_p_mismatches':mismatches,'scope':'simulation on generated diagnostic support; not human data or field sample planning'})
            print('benchmark',kind,n,eta,flush=True)
    assert maxerr<1e-8 and maxpdiff==0
    return pd.DataFrame(records),{'max_score_CMH_statistic_difference':maxerr,'max_score_CMH_p_difference':maxpdiff,'response_metric_and_alternative_space_matched':True,'repeated_human_records_modeled':False}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--route-data',type=Path,required=True); ap.add_argument('--vehicle-data',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--replications',type=int,default=500); ap.add_argument('--draws',type=int,default=499); args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    route,rt,rp=read_route(args.route_data); vehicle,vt,vp=read_vehicle(args.vehicle_data)
    sources=[{'case':'Swiss public transport route SC','observed_tasks':rt,'persons':rp,'sha256':hashlib.sha256(args.route_data.read_bytes()).hexdigest(),'data_url':'https://apollochoicemodelling.com/files/examples/data/apollo_swissRouteChoiceData.csv','dictionary':'https://apollochoicemodelling.com/files/examples/data/apollo_swissRouteChoiceData_dictionary.pdf','study_reference':'Axhausen et al. 2008, Transport Policy, 10.1016/j.tranpol.2008.02.001','candidate_map':'analyst-defined generalized cost at 5/10/20/40 CHF per hour; not the original authors claim'},
    {'case':'Train vehicle stated choice public subset','observed_tasks':vt,'persons':vp,'sha256':hashlib.sha256(args.vehicle_data.read_bytes()).hexdigest(),'archive_url':'https://eml.berkeley.edu/Software/source_code/train_mxlmsl_06.zip','source_readme':'Train and Sonnier 2005 chapter, public software archive, version September 2006','candidate_map':'analyst-defined undiscounted ownership cost at 48/60/84 months; not the original authors claim'}]
    save(args.out/'published_data_sources.json',{'sources':sources,'choices_and_person_covariates_used_for_rejections':False,'scope':'audit of released displayed task support; not full original design generation or causal sufficiency validation','synthetic_examples_counted_as_published_field_designs':False})
    records=[]
    for kind,menus,parameters in [('route',route,[5,10,20,40]),('vehicle',vehicle,[48,60,84])]:
        for parameter in parameters:
            for representation in ['absolute_menu_summary','utility_difference_vector']:
                row,*_=audit(menus,kind,parameter,representation); records.append(row)
    pd.DataFrame(records).to_csv(args.out/'published_support_and_classical_equivalence.csv',index=False)
    repairs=[]; probes=[]; scopes={}; simulations=[]; verification={}
    for kind,menus,parameter in [('route',route,10),('vehicle',vehicle,60)]:
        repair,probe,scope,pairs=repair_comparison(menus,kind,parameter); repairs.append(repair); probes.append(probe); scopes[kind]=scope
        sim,check=comparator_simulation(pairs,kind,parameter,args.replications,args.draws); simulations.append(sim); verification[kind]=check
    pd.concat(repairs).to_csv(args.out/'fixed_budget_repair_vs_classical.csv',index=False); pd.concat(probes).to_csv(args.out/'generated_diagnostic_menus.csv',index=False)
    pd.concat(simulations).to_csv(args.out/'matched_classical_test_benchmark.csv',index=False)
    save(args.out/'generated_probe_scope.json',scopes)
    save(args.out/'audit_technical_verification.json',{'real_archive_cases':2,'published_map_cases':14,'all_fibre_Schur_checks_passed':True,'all_saturated_rank_checks_passed':True,'fixed_budget_information_checks_passed':True,'equal_pair_allocation_Loewner_dominance_checked':True,'simulations':verification,'no_independent_new_method_claim_established':True,'raw_source_hashes_preserved':True})
    print('PUBLISHED_DCE_AUDIT_COMPLETE',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1): main()
