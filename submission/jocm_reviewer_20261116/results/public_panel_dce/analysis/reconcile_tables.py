"""Reconcile both new manuscript tables directly to complete method records.

No refitting, random draws or source workbook is needed. The prior independently
verified file hashes bind all results. Older table checks remain in quick_check.
"""
from pathlib import Path
import argparse,hashlib,json,re
import pandas as pd
def main(root,out):
    p=root/'results/public_panel_dce';v=json.loads((p/'verification/verification.json').read_text());assert v['state']=='passed'
    for n,h in v['result_hashes'].items():assert hashlib.sha256((p/'results'/n).read_bytes()).hexdigest()==h,n
    a=pd.read_csv(p/'results/method_records.csv');assert len(a)==16000
    assert not a.duplicated(['allocation','labelled_people','method']).any() and a.groupby(['labelled_people','method']).size().eq(500).all()
    t4=pd.read_csv(root/'manuscript/tableC4.csv',dtype=str);t5=pd.read_csv(root/'manuscript/tableC5.csv',dtype=str)
    counts={'Ordinary MNL':'labelled_MLE','Unit power':'corrected_unit','Half power':'corrected_half','Cluster power':'corrected_cluster_power'}
    for r in t4.to_dict('records'):
        m=int(r['Labelled people'])
        for heading,method in counts.items():
            g=a[(a.labelled_people==m)&(a.method==method)]
            assert r[heading]==f"{int(g.fit.eq('passed_score').sum())}/500"
    labels={'Labelled MNL':'labelled_MLE','Labelled ridge':'labelled_ridge','All-human MNL':'all_human_MLE','All-human ridge':'all_human_ridge','Unit correction':'corrected_unit','Half correction':'corrected_half','Cluster power':'corrected_cluster_power','Soft pooling':'pooling'}
    for r in t5.to_dict('records'):
        for m in [5,10,20,40]:
            g=a[(a.labelled_people==m)&(a.method==labels[r['Estimator']])&a.fit.eq('passed_score')]
            assert r[f'{m} people']==f'{g.heldout_logloss.mean():.4f}'
    paper=(root/'manuscript/paper.md').read_text()
    for name,frame in [('C4',t4),('C5',t5)]:
        block=re.search(r'^Table '+name+r'  [^\n]+\n\n((?:\|[^\n]*\n)+)',paper,re.M).group(1)
        parsed=[[c.strip() for c in line.strip('|').split('|')] for line in block.strip().splitlines()]
        assert parsed[0]==frame.columns.tolist() and parsed[2:]==frame.values.tolist()
    assert len(re.findall(r'(?:^\|[^\n]*\n)+',paper,re.M))==20
    assert len(t4)==4 and len(t5)==8
    # Statements in the new empirical text are checked on paired allocations.
    paired=[]
    for m in [5,10,20,40]:
        g=a[a.labelled_people==m];s=g.pivot(index='allocation',columns='method',values='support')
        paired.append(int((s.labelled_MLE.eq('interior')&s.corrected_unit.eq('outside')).sum()))
        cp=g[g.method.eq('corrected_cluster_power')]
        assert cp.loc[cp.support.eq('boundary_or_near_boundary'),'power'].eq(0).all()
    assert paired==[40,18,0,0]
    assert v['available_coefficients_checked']==15418 and v['all_exterior_rays_checked']==253 and v['independent_original_row_residual_LPs']==162
    result=dict(state='passed',new_tables_reconciled=['C4','C5'],table_C4_rows=4,table_C5_rows=8,complete_method_records=16000,paired_unit_outside_with_ordinary_interior=paired,all_20_table_blocks_present=True,old_18_table_values_checked_by_existing_entry=True,new_primary_fits_or_MC_run=False,prior_original_row_verification_hash=hashlib.sha256((p/'verification/verification.json').read_bytes()).hexdigest())
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package-root',type=Path,default=Path(__file__).resolve().parents[3]);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.package_root,a.out)
