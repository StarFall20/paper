"""Check delivered inventory and saved evidence without repeating calculations."""
from pathlib import Path
import csv, hashlib, json, re
R=Path(__file__).resolve().parent
expected={}
for line in (R/'MANIFEST.sha256').read_text().splitlines():
    h,n=line.split('  ',1);assert n not in expected
    assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n
    expected[n]=h
actual={str(f.relative_to(R)) for f in R.rglob('*') if f.is_file() and f.name not in ['MANIFEST.sha256','.DS_Store'] and '__pycache__' not in f.parts}
assert set(expected)==actual
verification=json.loads((R/'results/independent_verification.json').read_text())
assert verification['status']=='passed' and verification['conditions_checked']==196
assert verification['endpoint_cases_checked']==28
for n,h in verification['inputs'].items():assert expected[n]==h,n
assert expected['verify_independently.py']==verification['verifier_sha256']
assert len(json.loads((R/'results/boundaries.json').read_text()))==196
assert len(list(csv.DictReader((R/'results/boundaries.csv').open())))==196
assert len(json.loads((R/'results/endpoints.json').read_text()))==28
replay=json.loads((R/'replay_verification.json').read_text())
assert replay['status']=='passed' and replay['all_scientific_outputs_byte_identical']
for n,h in replay['output_sha256'].items():assert expected[n]==h,n
for f in R.rglob('*.md'):
    for target in re.findall(r'\]\(([^)]+)\)',f.read_text()):
        if target.startswith(('http:','https:','#')):continue
        assert (f.parent/target.split('#',1)[0]).exists(),(str(f),target)
print(json.dumps(dict(status='passed',files=len(expected)+1,conditions=196,endpoint_cases=28,
                     inventory_check_only=True)))
