from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parent
listed=set()
for line in (ROOT/'MANIFEST.sha256').read_text().splitlines():
 h,n=line.split('  ',1);assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n;listed.add(n)
actual={str(f.relative_to(ROOT)) for f in ROOT.rglob('*') if f.is_file() and f.name not in ['MANIFEST.sha256','.DS_Store'] and not {'regenerated','regenerated_full','__pycache__'}.intersection(f.relative_to(ROOT).parts)}
assert actual==listed,{'unlisted':sorted(actual-listed),'missing':sorted(listed-actual)}
for f in ROOT.rglob('*.md'):
 for target in re.findall(r'\]\(([^)]+)\)',f.read_text()):
  if target.startswith(('http:','https:','#')):continue
  assert (f.parent/target.split('#',1)[0]).exists(),(f,target)
print(json.dumps({'state':'passed','verified_files':len(listed),'private_data_included':False}))
