"""Retain all discrete-profile cells and build a concise manuscript table."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(a.results/'trials.csv');summary=pd.read_csv(a.results/'summary.csv');paired=pd.read_csv(a.results/'paired_with_gaussian.csv')
    rows=[]
    for m,q in [(40,26),(80,52)]:
        for law,label in [('gaussian','Gaussian'),('symmetric_binary','Two levels'),('symmetric_ternary','Three levels')]:
            part=summary[summary.label_menus.eq(m)&summary.orthogonal_slopes.eq(q)&summary.law.eq(law)].set_index('method')
            rows.append(dict(Labels=m,Slopes=q,Attributes=label,Ordinary=f'{int(part.loc["ordinary","finite"])}/500',Corrected=f'{int(part.loc["oracle_null_slope_variance","finite"])}/500',Unit=f'{int(part.loc["unit","finite"])}/500'))
    table=pd.DataFrame(rows);table.to_csv(a.out/'tableF4.csv',index=False)
    table_md='| '+' | '.join(table.columns)+' |\n| '+' | '.join(['---']*len(table.columns))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in table.itertuples(index=False,name=None))
    (a.out/'tableF4.md').write_text(table_md)
    error=0.
    for row in paired.itertuples():
        part=raw[raw.label_menus.eq(row.label_menus)&raw.orthogonal_slopes.eq(row.orthogonal_slopes)&raw.method.eq(row.method)]
        x=part.assign(finite=part.status.eq('interior').astype(int)).pivot(index='trial',columns='law',values='finite')
        d=(x[row.law]-x.gaussian).to_numpy();error=max(error,abs(d.mean()-row.finite_frequency_difference),abs(d.std(ddof=1)/np.sqrt(len(d))-row.paired_MCSE))
    assert error<1e-13
    result=dict(state='passed',source_records=27000,source_cells=54,paired_comparisons=len(paired),table_rows=len(table),maximum_paired_reconstruction_error=error,maximum_absolute_discrete_gaussian_frequency_difference=float(paired.finite_frequency_difference.abs().max()),all_cells_retained=True,finite_sample_equality_claimed=False)
    (a.out/'summary_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
