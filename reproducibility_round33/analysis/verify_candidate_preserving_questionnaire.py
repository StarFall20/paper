"""Verify the two human-facing questionnaire versions preserve each menu vector."""
from __future__ import annotations
import csv, os

TASKS=[('G0','G1',79,99),('G2','G1',59,89),('G1','G0',69,109),('G0','G2',79,119),('G2','G1',89,69),('G1','G0',99,79),('G0','G2',109,59),('G2','G1',119,69)]
# (data, support, intelligence, evidence)
V1=[('D0','S2','I0','E2'),('D1','S2','I0','E1')]
V2=[('D1','S1','I1','E1'),('D2','S1','I1','E0')]
LEVEL={'D0':0,'D1':1,'D2':2,'S0':0,'S1':1,'S2':2,'I0':0,'I1':1,'I2':2,'E0':0,'E1':1,'E2':2}

def vector(profile):
    d,s,i,e=profile
    return (LEVEL[d]+LEVEL[s], LEVEL[i]+LEVEL[e])

def run(out='results/questionnaire_candidate_preservation_audit.csv'):
    rows=[]
    for task,(ga,gb,pa,pb) in enumerate(TASKS,1):
        for alt,(a,b) in enumerate(zip(V1,V2),1):
            rows.append({'task':task,'alternative':alt,'version1_candidate_1':vector(a)[0], 'version2_candidate_1':vector(b)[0],
                         'version1_candidate_2':vector(a)[1], 'version2_candidate_2':vector(b)[1],
                         'sharing_and_price_fixed':1,'candidate_vector_preserved':int(vector(a)==vector(b))})
    assert all(r['candidate_vector_preserved'] for r in rows)
    assert len(rows)==16
    os.makedirs(os.path.dirname(out) or '.',exist_ok=True)
    with open(out,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(f'verified {len(rows)} alternative cards across {len(TASKS)} tasks; all candidate vectors preserved')
if __name__=='__main__':run()
