from pathlib import Path
import csv,hashlib,json,re
root=Path(__file__).resolve().parent;listed=set()
for line in (root/'MANIFEST.sha256').read_text().splitlines():
    h,n=line.split('  ',1);assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n;listed.add(n)
actual={str(f.relative_to(root)) for f in root.rglob('*') if f.is_file() and f!=root/'MANIFEST.sha256' and f.name!='.DS_Store' and not {'__pycache__','regenerated'}.intersection(f.relative_to(root).parts)}
assert listed==actual
for f in root.glob('*.md'):
    for target in re.findall(r'\]\(([^)]+)\)',f.read_text()):
        if target.startswith(('http:','https:','#')):continue
        assert (f.parent/target.split('#',1)[0]).exists(),target
v=json.loads((root/'verification/verification.json').read_text());assert v['state']=='passed'
for n,h in v['source_hashes'].items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
plan=json.loads((root/'plan.json').read_text())
for n,key in [('method_records.csv','input_records_sha256'),('coefficients.json','input_coefficients_sha256')]:assert hashlib.sha256((root/'reference'/n).read_bytes()).hexdigest()==plan[key]
table=re.findall(r'(?:^\|[^\n]*\n)+',(root/'README.md').read_text(),re.M)[0]
rows=[[x.strip() for x in line.strip('|').split('|')] for line in table.strip().splitlines()][2:];assert len(rows)==32
avail={(r['labelled_people'],r['method']):r for r in csv.DictReader((root/'results/availability.csv').open())}
bere={(r['labelled_people'],r['method']):r for r in csv.DictReader((root/'results/negative_price_attribute_ratios.csv').open()) if r['attribute']=='bere'}
for row in rows:
    key=tuple(row[:2]);a,b=avail[key],bere[key];assert row[2:4]==[a['available_fits'],a['negative_price']]
    assert row[4]==f"{float(b['q50']):.4f}" and row[5]==f"[{float(b['q05']):.4f}, {float(b['q95']):.4f}]"
assert not any(n.endswith(('.pdf','.xlsx','.npz','.pkl')) for n in listed)
print(json.dumps(dict(state='passed',manifest_entries=len(listed),all_32_readme_rows_reconciled=True,raw_respondent_data_included=False,global_priority_or_submission_threshold_certified=False)))
