"""Check the isolated archive inventory, checksums, links and local imports."""
from pathlib import Path
import ast,hashlib,re,json
ROOT=Path(__file__).resolve().parent
entries=(ROOT/'MANIFEST.sha256').read_text().splitlines();listed=set()
for line in entries:
 sha,path=line.split('  ',1);f=ROOT/path;assert f.is_file(),path;assert hashlib.sha256(f.read_bytes()).hexdigest()==sha,path;listed.add(path)
missing=[]
for f in ROOT.rglob('*.md'):
 for target in re.findall(r'\]\(([^)]+)\)',f.read_text()):
  if target.startswith(('http:','https:','#')):continue
  dest=(f.parent/target.split('#',1)[0]).resolve()
  if not dest.exists():missing.append((str(f.relative_to(ROOT)),target))
assert not missing,missing
for f in (ROOT/'analysis').glob('*.py'):
 tree=ast.parse(f.read_text())
 for node in ast.walk(tree):
  if isinstance(node,ast.ImportFrom) and node.module and node.module.startswith(('public_conjoint','finite_ratio','power_tuned','choice_moment','imperfect_predictor','finite_pool','audit_binary')):assert (ROOT/'analysis'/f'{node.module}.py').exists(),(f,node.module)
assert not any('trial_diagnostics' in x or 'unavailable_cases' in x or x.endswith('.pkl') for x in listed)
print(json.dumps({'manifest_entries_verified':len(listed),'broken_local_markdown_links':len(missing),'science_import_closure_present':True,'raw_pickle_inputs_shipped':False}))
