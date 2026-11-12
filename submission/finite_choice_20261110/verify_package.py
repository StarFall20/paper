"""Check the isolated archive inventory, checksums, links and local imports."""
from pathlib import Path
import ast,hashlib,re,json
ROOT=Path(__file__).resolve().parent
entries=(ROOT/'MANIFEST.sha256').read_text().splitlines();listed=set()
for line in entries:
 sha,path=line.split('  ',1);f=ROOT/path;assert f.is_file(),path;assert hashlib.sha256(f.read_bytes()).hexdigest()==sha,path;listed.add(path)
actual={str(f.relative_to(ROOT)) for f in ROOT.rglob('*') if f.is_file() and f.name not in ['MANIFEST.sha256','.DS_Store'] and not {'regenerated','__pycache__'}.intersection(f.relative_to(ROOT).parts)}
assert actual==listed,{'unlisted':sorted(actual-listed),'missing':sorted(listed-actual)}
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
  if isinstance(node,ast.ImportFrom) and node.module and node.module.startswith(('public_conjoint','finite_ratio','power_tuned','power_domain','attribute_aligned','calibrated_aligned','choice_moment','imperfect_predictor','finite_pool','audit_binary','vanishing_power','joint_power','general_joint','verify_general')):assert (ROOT/'analysis'/f'{node.module}.py').exists(),(f,node.module)
assert not any('trial_diagnostics' in x or 'unavailable_cases' in x or x.endswith('.pkl') for x in listed)
menu=json.loads((ROOT/'results/menu/source_hashes.json').read_text())
for path,sha in menu['sha256'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,path
for script in menu['current_entry_points']:assert (ROOT/'analysis'/f'{script}.py').is_file(),script
score=json.loads((ROOT/'results/score/source_hashes.json').read_text())
for path,sha in score['sha256'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,path
for script in score['current_entry_points']:assert (ROOT/'analysis'/f'{script}.py').is_file(),script
for key in ['population','phase']:assert hashlib.sha256((ROOT/score[key+'_plan']).read_bytes()).hexdigest()==score[key+'_plan_sha256']
print(json.dumps({'manifest_entries_verified':len(listed),'broken_local_markdown_links':len(missing),'science_import_closure_present':True,'menu_source_closure_files':len(menu['sha256']),'score_source_closure_files':len(score['sha256']),'raw_pickle_inputs_shipped':False}))
