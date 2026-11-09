"""One external DCE design audit; classical rank diagnostics, no novelty claim.

Input: BRAC University's original public CHORUS CSV. No respondent records or
answer vectors are written. Only aggregate design checks are exported.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np,pandas as pd

EXPECTED='0804176ec8f3114fd220da90966d8f26'
ATTR=['Med_trt_cost','wait_time','Service_mch','referral','provider','consul_fee']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    data=a.input.read_bytes();assert hashlib.md5(data).hexdigest()==EXPECTED,'Original source checksum mismatch'
    aliases={'MedicineTreatmentCost':'Med_trt_cost','WaitingTime':'wait_time','ServicesBeyondMCH':'Service_mch','ReferralServices':'referral','ProviderType':'provider','ConsultationFee':'consul_fee'}
    d=pd.read_csv(a.input,usecols=['case','ChoiceSet','Block_Set','Alternative_slno',*aliases]).rename(columns=aliases)
    decoding={'Alternative_slno':{'Facility_1':1,'Facility_2':2,'Opt-out':0},'Med_trt_cost':{'Free of cost':1,'Half payment':2,'Full payment':3},'wait_time':{'30 minutes':30,'60 minutes':60,'90 minutes':90},'Service_mch':{'Yes':1,'No':2},'referral':{'Yes':1,'No':2},'provider':{'MBBS Doctor':1,'Paramedic':2,'Drug seller':3},'consul_fee':{'BDT 100':100,'BDT 200':200,'BDT 300':300,'Free':0}}
    for col,mapping in decoding.items():
        values=d[col].astype(str).str.strip();unmapped=set(values)-set(mapping)-{'','nan'};assert not unmapped,(col,unmapped)
        d[col]=values.map(mapping)
    assert d.Alternative_slno.notna().all()
    required=['ChoiceSet','Block_Set','Alternative_slno',*ATTR]
    # Blank fields and "Free" both occur on the opt-out row. These are
    # inactive attribute placeholders, not facility attribute variations.
    d.loc[d.Alternative_slno==0,ATTR]=0.
    menu=d.groupby(['Block_Set','ChoiceSet','Alternative_slno'],dropna=False)[ATTR].nunique(dropna=False)
    varying_slots=int((menu>1).any(axis=1).sum())
    # The file contains two attribute variants in every declared block/task
    # slot. Keep complete observed menus, not an invented canonical design.
    selected=[]; seen=set();variants=[]
    for (_,block,task),g in d.sort_values('Alternative_slno').groupby(['case','Block_Set','ChoiceSet']):
        assert len(g)==3 and g.Alternative_slno.tolist()==[0,1,2]
        # Missing values have one explicit signature; NaN is not self-equal.
        flat=[None if pd.isna(v) else float(v) for v in g[ATTR].to_numpy().ravel()]
        signature=(block,task,*flat)
        if signature not in seen:
            seen.add(signature);selected.append(g[required]);variants.append((block,task))
    menus=pd.concat(selected,ignore_index=True);nmenus=len(selected)
    declared_slots=len(d[['Block_Set','ChoiceSet']].drop_duplicates())
    variant_distribution=pd.DataFrame(variants,columns=['block','task']).groupby(['block','task']).size().value_counts().sort_index().to_dict()
    raw_active=d.Alternative_slno!=0
    missing=d.loc[raw_active,ATTR].isna().any(axis=1)
    if missing.any():
        result={'status':'preflight_requires_source_clarification','source':'BRAC University, CHORUS BANGLADESH DCE DATASET','source_url':'https://zenodo.org/records/19478777','source_md5':EXPECTED,'source_sha256':hashlib.sha256(data).hexdigest(),'data_rows':len(d),'choice_occasions':len(d)//3,'declared_block_task_slots':declared_slots,'observed_complete_row_triplet_variants':nmenus,'ordered_variants_per_declared_slot_distribution':{str(k):int(v) for k,v in variant_distribution.items()},'active_facility_rows':int(raw_active.sum()),'active_facility_rows_with_missing_attributes':int(missing.sum()),'missing_active_by_attribute':{c:int(v) for c,v in d.loc[raw_active,ATTR].isna().sum().items()},'active_declared_profile_slots_with_varying_attributes':int((menu.loc[menu.index.get_level_values('Alternative_slno')!=0]>1).any(axis=1).sum()),'decoding_from_csv_labels':decoding,'outside_placeholder_rule':'Blank/Free attributes on opt-out are inactive placeholders normalized to zero; active-facility missing values are not filled.','design_ranks_computed':False,'outcome_models_fitted':False,'respondent_data_exported':False,'reason':'Declared block/task slots do not specify a single stable observed menu and some active facility rows have missing attributes. The original planned design cannot be certified from this file without clarification. No canonicalization, modal fill or complete-case replacement is used to produce a passing audit.'}
        (a.out/'public_healthcare_design_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2));return
    active=menus.Alternative_slno!=0
    assert not menus.loc[active,ATTR].isna().any().any(),'Missing active-facility attributes'
    # Confirm the observed active levels against the archive codebook.
    levels={c:sorted(menus.loc[active,c].dropna().unique().tolist()) for c in ATTR}
    expected={'Med_trt_cost':[1,2,3],'wait_time':[30,60,90],'Service_mch':[1,2],'referral':[1,2],'provider':[1,2,3],'consul_fee':[100,200,300]}
    assert levels==expected,(levels,expected)
    arr=[(menus.Alternative_slno==i).astype(float).to_numpy() for i in [1,2]];names=['ASC_Facility1','ASC_Facility2']
    for col in ATTR[:-1]:
        for lev in levels[col][1:]:arr.append(((menus[col]==lev)&active).to_numpy(float));names.append(f'{col}_{lev}')
    nuisance=np.column_stack(arr)
    fee=np.where(active.to_numpy(),(menus.consul_fee.to_numpy(float)-100)/100,0.)
    cat=np.column_stack([((menus.consul_fee==lev)&active).to_numpy(float) for lev in [200,300]])
    xx=np.column_stack([nuisance,fee]);xc=np.column_stack([nuisance,cat]);curvature=cat[:,0]
    def diffs(m):
        m=m.reshape(nmenus,3,-1);return (m[:,:-1]-m[:,-1:]).reshape(-1,m.shape[-1])
    D,Dc,z=diffs(xx),diffs(xc),diffs(curvature[:,None])[:,0]
    r=np.linalg.matrix_rank(D);rc=np.linalg.matrix_rank(Dc);res=z-D@np.linalg.lstsq(D,z,rcond=None)[0]
    da=np.column_stack([D,z]);assert np.linalg.matrix_rank(da)==rc
    information=np.linalg.svd(Dc,compute_uv=False)
    # Optional block/alternative intercept adjustment is reported separately.
    blocks=sorted(menus.Block_Set.unique());extra=[]
    for block in blocks[1:]:
        for alt in [1,2]:extra.append(((menus.Block_Set==block)&(menus.Alternative_slno==alt)).to_numpy(float))
    db=diffs(np.column_stack([xc,*extra]));rankb=np.linalg.matrix_rank(db)
    pair=menus[active].groupby(['Med_trt_cost','consul_fee']).size().reset_index(name='profile_appearances_across_menus')
    pair.to_csv(a.out/'public_medicine_category_by_consultation_fee_support.csv',index=False)
    result={'source':'BRAC University, CHORUS BANGLADESH DCE DATASET','source_url':'https://zenodo.org/records/19478777','source_md5':EXPECTED,'source_sha256':hashlib.sha256(data).hexdigest(),'data_rows':len(d),'blocks':len(blocks),'distinct_menus':nmenus,'profiles_in_menus':len(menus),'active_attribute_levels':levels,'linear_fee_columns':D.shape[1],'linear_fee_rank':int(r),'categorical_fee_columns':Dc.shape[1],'categorical_fee_rank':int(rc),'fee_curvature_additional_rank':int(rc-r),'fee_curvature_residual_sum_squares':float(res@res),'categorical_fee_design_smallest_singular_value':float(min(information)),'categorical_fee_design_condition_number':float(max(information)/min(information)),'categorical_fee_with_block_alternative_intercepts_columns':db.shape[1],'categorical_fee_with_block_alternative_intercepts_rank':int(rankb),'medicine_cost_codes_are_not_currency':True,'outcome_models_fitted':False,'respondent_data_exported':False,'inference':'An independently released healthcare DCE design audit; classical estimability diagnostics only. Full rank does not prove power, truth of additivity, random recruitment, external validity or novelty.'}
    (a.out/'public_healthcare_design_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
