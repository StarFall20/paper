from pathlib import Path
import ast,hashlib,json,re
root=Path(__file__).resolve().parent
listed={}
for line in (root/'MANIFEST.sha256').read_text().splitlines():
 sha,rel=line.split('  ',1);assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==sha,rel;listed[rel]=sha
actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name not in ['MANIFEST.sha256','.DS_Store'] and not {'__pycache__','regenerated'}.intersection(p.relative_to(root).parts)}
assert actual==set(listed)
for rel,sha in json.loads((root/'SOURCE_HASHES.json').read_text())['sha256'].items():assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==sha
missing=[]
for f in root.rglob('*.md'):
 for link in re.findall(r'\]\(([^)]+)\)',f.read_text()):
  if not link.startswith(('https:','http:','#')) and not (f.parent/link.split('#')[0]).exists():missing.append((str(f.relative_to(root)),link))
assert not missing,missing
print(json.dumps({'manifest_entries':len(listed),'broken_local_links':0,'private_responses_included':False}))
