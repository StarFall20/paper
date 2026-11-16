"""Retain every condition and every pair in the verified one-step audit."""
from pathlib import Path
import argparse,itertools,json
import numpy as np
import pandas as pd

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--verification',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    v=json.loads((a.verification/'verification.json').read_text());assert v['state']=='passed' and v['records_checked']==60000
    f=pd.read_csv(a.results/'trial_records.csv.gz');s=pd.read_csv(a.results/'summary.csv')
    keys=['study','prediction','label_rows','auxiliary_rows','epsilon'];methods=['pooled_clipped','pooled_unclipped','labelled_clipped','labelled_unclipped','human_ridge']
    f['raw_outside']=(f.raw_weight.lt(0)|f.raw_weight.gt(1))&f.raw_weight.notna()
    raw=f.groupby(keys+['method'],as_index=False).agg(raw_outside=('raw_outside','sum'),raw_negative=('raw_weight',lambda x:int(x.lt(0).sum())),raw_above_one=('raw_weight',lambda x:int(x.gt(1).sum())),max_curvature_condition=('curvature_condition','max'),max_MSE=('MSE','max'))
    s=s.merge(raw,on=keys+['method'],validate='one_to_one');s.to_csv(a.out/'method_summary.csv',index=False)
    pairs=[]
    for condition,part in f.groupby(keys):
        risk=part.pivot(index='replication',columns='method',values='excess_logloss');mse=part.pivot(index='replication',columns='method',values='MSE')
        assert risk.shape==(500,5)
        for left,right in itertools.combinations(methods,2):
            valid=risk[left].notna()&risk[right].notna();d=(risk.loc[valid,left]-risk.loc[valid,right]);dm=mse.loc[valid,left]-mse.loc[valid,right]
            pairs.append(dict(zip(keys,condition))|dict(left=left,right=right,shared_finite=int(valid.sum()),mean_loss_left_minus_right=float(d.mean()),loss_MCSE=float(d.std(ddof=1)/np.sqrt(len(d))),median_paired_loss_difference=float(d.median()),left_lower_loss=int(d.lt(0).sum()),equal_loss=int(d.eq(0).sum()),right_lower_loss=int(d.gt(0).sum()),mean_MSE_left_minus_right=float(dm.mean()),MSE_MCSE=float(dm.std(ddof=1)/np.sqrt(len(dm)))))
    pair=pd.DataFrame(pairs);assert len(pair)==240;pair.to_csv(a.out/'all_pairwise.csv',index=False)
    clipping=pair[((pair.left.eq('pooled_clipped')&pair.right.eq('pooled_unclipped'))|(pair.left.eq('labelled_clipped')&pair.right.eq('labelled_unclipped')))].copy()
    assert len(clipping)==48;clipping.to_csv(a.out/'clipping_effects.csv',index=False)
    raw_records=f[f.method.isin(['pooled_unclipped','labelled_unclipped'])]
    outside=raw_records[raw_records.outside_unit_interval];negative=raw_records[raw_records.negative_curvature]
    classifications={}
    for info in ['pooled','labelled']:
        t=clipping[clipping.left.eq(info+'_clipped')]
        # Differences are clipped minus unrestricted; positive values favor unrestricted updates.
        classifications[info]={'conditions':len(t),'unrestricted_lower_mean_loss':int(t.mean_loss_left_minus_right.gt(1e-12).sum()),'unrestricted_higher_mean_loss':int(t.mean_loss_left_minus_right.lt(-1e-12).sum()),'same_mean_loss':int(t.mean_loss_left_minus_right.abs().le(1e-12).sum()),'maximum_absolute_mean_loss_change':float(t.mean_loss_left_minus_right.abs().max())}
    overview=dict(state='complete',conditions=24,replications_per_condition=500,method_records=60000,original_cohort_families=1500,new_independent_cohorts=0,new_human_or_AI_answers=0,all_legacy_cells_reconciled=24,all_updates_available=bool(f.available.all()),raw_domain_method_records=len(raw_records),outside_domain_method_records=len(outside),outside_domain_condition_repetition_cases=len(outside.drop_duplicates(keys+['replication'])),outside_domain_original_cohort_families=len(outside.drop_duplicates(['study','replication'])),negative_curvature_method_records=len(negative),negative_curvature_condition_repetition_cases=len(negative.drop_duplicates(keys+['replication'])),clipping_comparison=classifications,all_pairwise_rows=len(pair),finite_global_roots_or_interval_validity_claimed=False,formal_PPI_A_to_E_compared=False)
    (a.out/'overview.json').write_text(json.dumps(overview,indent=2,allow_nan=False)+'\n');print(json.dumps(overview))
    display=s.pivot(index=keys,columns='method',values='mean_excess_logloss').reset_index()
    columns=keys+methods;lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for _,r in display.iterrows():lines.append('| '+' | '.join([str(r[k]) for k in keys]+[f'{r[k]:.6f}' for k in methods])+' |')
    (a.out/'all_conditions_mean_loss.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
