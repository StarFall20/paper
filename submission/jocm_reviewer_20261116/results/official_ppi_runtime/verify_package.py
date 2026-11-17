from pathlib import Path
import hashlib,json,re,csv
root=Path(__file__).resolve().parent;listed=set()
for line in (root/'MANIFEST.sha256').read_text().splitlines():
    h,n=line.split('  ',1);assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n;listed.add(n)
actual={str(f.relative_to(root)) for f in root.rglob('*') if f.is_file() and f!=root/'MANIFEST.sha256' and f.name!='.DS_Store' and not {'__pycache__','local_inputs','local_frozen','local_upstream','regenerated'}.intersection(f.relative_to(root).parts)}
assert listed==actual,(listed-actual,actual-listed)
for f in root.glob('*.md'):
    for target in re.findall(r'\]\(([^)]+)\)',f.read_text()):
        if target.startswith(('https:','http:','#')):continue
        assert (f.parent/target.split('#',1)[0]).exists(),target
v=json.loads((root/'verification/verification.json').read_text());assert v['state']=='passed'
for n,h in v['source_hashes'].items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
assert not any(n.endswith(('.npz','.pkl','.pdf')) for n in listed)
table=re.findall(r'(?:^\|[^\n]*\n)+',(root/'README.md').read_text(),re.M)[0]
cells=[[s.strip() for s in line.strip('|').split('|')] for line in table.strip().splitlines()][2:]
summary=list(csv.DictReader((root/'results/public_case_summary.csv').open()))
assert len(cells)==len(summary)==8
for row,record in zip(cells,summary):
    assert row[2:]==[record[k] for k in ['finite_numeric_arrays','last_optimizer_success','API_exceptions']]
    assert row[1]==('自动' if record['requested_power']=='automatic' else record['requested_power'])
print(json.dumps(dict(state='passed',files=len(listed),original_source_checks_present=True,raw_response_arrays_shipped=False,manuscript_or_global_priority_certified=False)))
