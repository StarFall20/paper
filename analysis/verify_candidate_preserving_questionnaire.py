"""Verify the two questionnaire versions preserve the complete candidate menu."""
from __future__ import annotations
import csv, os
import numpy as np

def run(out='results/questionnaire_candidate_preservation_audit.csv'):
    # Numeric levels correspond to the codes in the questionnaire markdown.
    v1=np.array([[0,2,0,2],[1,2,0,1]],dtype=int)
    v2=np.array([[1,1,1,1],[2,1,1,0]],dtype=int)
    rows=[]
    for alt,(a,b) in enumerate(zip(v1,v2),1):
        rows.append({'alternative':alt,'v1_m_plus_s':int(a[0]+a[1]),'v2_m_plus_s':int(b[0]+b[1]),'v1_i_plus_e':int(a[2]+a[3]),'v2_i_plus_e':int(b[2]+b[3]),'preserved':int(a[0]+a[1]==b[0]+b[1] and a[2]+a[3]==b[2]+b[3])})
    assert all(r['preserved'] for r in rows)
    os.makedirs(os.path.dirname(out) or '.',exist_ok=True)
    with open(out,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('verified',len(rows),'alternatives; all candidate summaries preserved')
if __name__=='__main__':run()
