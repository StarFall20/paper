from pathlib import Path
import hashlib,json,ast
ROOT=Path(__file__).resolve().parent
rows=(ROOT/'MANIFEST.sha256').read_text().splitlines();listed=set()
for row in rows:
 h,n=row.split('  ',1);assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n;listed.add(n)
actual={str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and p.name!='.DS_Store' and p!=ROOT/'MANIFEST.sha256' and not {'local_inputs','regenerated','regenerated_full','__pycache__'}.intersection(p.relative_to(ROOT).parts)}
assert actual==listed,(actual-listed,listed-actual)
sources=json.loads((ROOT/'source_record.json').read_text())['scientific_sources']
for n,h in sources.items():assert hashlib.sha256((ROOT/'analysis'/n).read_bytes()).hexdigest()==h
for p in (ROOT/'analysis').glob('*.py'):ast.parse(p.read_text())
assert json.loads((ROOT/'verification/verification.json').read_text())['source_sha256']==sources['verify_public_auxiliary_budget.py']
for name in ['verification.json','exact_case_verification.json']:assert json.loads((ROOT/'verification'/name).read_text())['state']=='passed'
assert json.loads((ROOT/'results/status.json').read_text())['method_records']==36000
assert not any(n.endswith(('.pkl','.xlsx')) for n in listed)
print(json.dumps(dict(manifest_entries=len(listed),scientific_sources=len(sources),raw_numeric_inputs_shipped=False)))
