from pathlib import Path
import hashlib,json,ast
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=[x.split('  ',1) for x in (ROOT/'MANIFEST.sha256').read_text().splitlines()]
for h,rel in rows:
    assert not Path(rel).is_absolute() and '..' not in Path(rel).parts
    assert sha(ROOT/rel)==h,rel
inventory={str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and p!=ROOT/'MANIFEST.sha256' and p.name!='.DS_Store' and not {'__pycache__','regenerated','local_inputs'}.intersection(p.relative_to(ROOT).parts)}
assert inventory=={rel for _,rel in rows},'Manifest does not cover the complete retained inventory'
closure=json.loads((ROOT/'evidence/current_source_hashes.json').read_text())['sha256']
for rel,h in closure.items():assert sha(ROOT/rel)==h,rel
for p in (ROOT/'analysis').glob('*.py'):
    for n in ast.walk(ast.parse(p.read_text())):
        if isinstance(n,ast.ImportFrom) and n.module:
            name=n.module.split('.')[0]
            if name in [Path(x).stem for x in closure]:assert (p.parent/(name+'.py')).exists(),name
v=[json.loads((ROOT/'evidence/verification_final'/m/'independent_risk_verification.json').read_text()) for m in ['public','gaussian']]
assert sum(x['retained_rows'] for x in v)==1920
assert sum(x['root_records_independently_replayed'] for x in v)==1401
assert sum(x['independent_support_endpoint_programs'] for x in v)==389
assert all(x['remaining_unresolved']==0 for x in v)
for m,x in zip(['public','gaussian'],v):
    assert sha(ROOT/'evidence/results'/m/'path_records.csv')==x['original_records_sha256']
    assert sha(ROOT/'analysis/verify_vanishing_power_risk.py')==x['source_sha256']
status=json.loads((ROOT/'evidence/verified_summary/summary_checks.json').read_text())
for rel,h in status['verified_inputs_sha256'].items():assert sha(ROOT/'evidence'/rel)==h
assert status['planned_records']==1920 and status['finite_root_records']==1401 and status['unresolved']==0
assert not any(p.suffix in ['.pkl','.xlsx'] for p in ROOT.rglob('*') if p.is_file() and 'local_inputs' not in p.parts)
print(json.dumps(dict(manifest_entries_verified=len(rows),source_files_verified=len(closure),retained_records=1920,finite_fits=1401,independent_support_endpoint_programs=389,unresolved=0)))
