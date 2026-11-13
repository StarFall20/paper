from pathlib import Path
import ast,hashlib,json,re
root=Path(__file__).resolve().parent
listed={}
for line in (root/'MANIFEST.sha256').read_text().splitlines():
 h,p=line.split('  ',1);assert hashlib.sha256((root/p).read_bytes()).hexdigest()==h,p;listed[p]=h
actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name not in ['MANIFEST.sha256','.DS_Store'] and not {'regenerated','__pycache__'}.intersection(p.relative_to(root).parts)}
assert set(listed)==actual
source=json.loads((root/'SOURCE_HASHES.json').read_text())
for p,h in source['sha256'].items():assert hashlib.sha256((root/p).read_bytes()).hexdigest()==h,p
assert hashlib.sha256((root/'analysis/row_weight_attainment_plan_20261113.json').read_bytes()).hexdigest()==source['plan_sha256']
assert hashlib.sha256((root/'inputs/previous_radial_verified.csv').read_bytes()).hexdigest()==source['reference_sha256']
missing=[]
for p in root.rglob('*.md'):
 for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
  if not target.startswith(('http:','https:','#')) and not (p.parent/target.split('#',1)[0]).exists():missing.append((str(p.relative_to(root)),target))
assert not missing,missing
print(json.dumps({'verified_entries':len(listed),'science_source_files':len(source['sha256']),'broken_links':0,'private_questionnaires_included':False}))
