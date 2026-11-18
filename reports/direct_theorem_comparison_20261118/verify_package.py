"""Verify inventory, source bindings and saved comparison results; no refitting."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,re
R=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
expected={}
for line in (R/'MANIFEST.sha256').read_text().splitlines():
    h,n=line.split('  ',1);assert n not in expected;assert sha(R/n)==h,n;expected[n]=h
actual={str(f.relative_to(R)) for f in R.rglob('*') if f.is_file() and f.name not in ['MANIFEST.sha256','.DS_Store'] and '__pycache__' not in f.parts}
assert set(expected)==actual
v=json.loads((R/'results/verification.json').read_text());assert v['status']=='passed'
for n,h in v['inputs'].items():assert expected[n]==h,n
assert expected['verify_independently.py']==v['verifier_sha256']
rows=json.loads((R/'results/population_pair.json').read_text());assert len(rows)==2
for row in rows:
    lo,hi=map(F,row['boundary_interval_fraction']);k=F(row['witness_dimension_ratio'])
    assert lo>k if row['power']=='1/4' else hi<k
    assert row['all_rows_strictly_convex'] and row['predictor_and_data_law_unchanged']
assert [x['positive_recession'] for x in v['exact_finite_checks']]==['1/80','-1/80']
replay=json.loads((R/'replay_verification.json').read_text());assert replay['status']=='passed'
for n,h in replay['output_sha256'].items():assert expected[n]==h,n
for f in R.rglob('*.md'):
    for target in re.findall(r'\]\(([^)]+)\)',f.read_text()):
        if target.startswith(('http:','https:','#')):continue
        assert (f.parent/target.split('#',1)[0]).exists(),(str(f),target)
print(json.dumps(dict(status='passed',files=len(expected)+1,population_conditions=2,exact_finite_conditions=2,inventory_check_only=True)))
