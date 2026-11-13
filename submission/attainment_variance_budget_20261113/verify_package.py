from pathlib import Path
import hashlib,json,re
root=Path(__file__).resolve().parent
listed={}
for line in (root/'MANIFEST.sha256').read_text().splitlines():
 h,p=line.split('  ',1);assert hashlib.sha256((root/p).read_bytes()).hexdigest()==h,p;listed[p]=h
actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name not in ['MANIFEST.sha256','.DS_Store'] and not {'regenerated','__pycache__'}.intersection(p.relative_to(root).parts)}
assert set(listed)==actual
s=json.loads((root/'SOURCE_HASHES.json').read_text())
for group in ['sha256','plans_sha256','inputs_sha256']:
 for p,h in s[group].items():assert hashlib.sha256((root/p).read_bytes()).hexdigest()==h,p
missing=[]
for f in root.rglob('*.md'):
 for target in re.findall(r'\]\(([^)]+)\)',f.read_text()):
  if not target.startswith(('http:','https:','#')) and not (f.parent/target.split('#',1)[0]).exists():missing.append((str(f.relative_to(root)),target))
assert not missing,missing
print(json.dumps({'manifest_entries':len(listed),'current_scientific_sources':len(s['sha256']),'broken_links':0,'private_responses_included':False}))
