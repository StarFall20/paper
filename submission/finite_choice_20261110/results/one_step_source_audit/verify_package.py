from pathlib import Path
import ast,hashlib,json,re
import pandas as pd

ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
entries={n:h for h,n in (l.split('  ',1) for l in (ROOT/'MANIFEST.sha256').read_text().splitlines())}
actual={str(f.relative_to(ROOT)) for f in ROOT.rglob('*') if f.is_file() and f!=ROOT/'MANIFEST.sha256' and f.name!='.DS_Store' and not {'__pycache__','regenerated','local_inputs'}.intersection(f.relative_to(ROOT).parts)}
assert actual==set(entries),{'unlisted':sorted(actual-set(entries)),'missing':sorted(set(entries)-actual)}
assert all(sha(ROOT/n)==h for n,h in entries.items())
record=json.loads((ROOT/'source_record.json').read_text())
for key in ['new_analysis_sources','inherited_helper_sources']:
 for n,h in record[key].items():assert sha(ROOT/'analysis'/n)==h
for n,h in record['reference_files'].items():assert sha(ROOT/'reference'/n)==h
for n,h in record['results'].items():assert sha(ROOT/n)==h
for f in ROOT.rglob('*.md'):
 for t in re.findall(r'\]\(([^)]+)\)',f.read_text()):
  if not t.startswith(('http:','https:','#')):assert (f.parent/t).exists(),(f,t)
helpers=set(record['new_analysis_sources'])|set(record['inherited_helper_sources'])
for f in (ROOT/'analysis').glob('*.py'):
 for node in ast.walk(ast.parse(f.read_text())):
  if isinstance(node,ast.ImportFrom) and node.module and node.module.startswith(('public_conjoint','power_tuned','finite_pool','choice_moment')):assert node.module+'.py' in helpers,(f,node.module)
v=json.loads((ROOT/'verification/verification.json').read_text());assert v['state']=='passed' and v['records_checked']==60000 and v['legacy_cells_reconciled']==24
f=pd.read_csv(ROOT/'results/trial_records.csv.gz');keys=['study','prediction','label_rows','auxiliary_rows','epsilon','method']
assert len(f)==60000 and f.groupby(keys).size().eq(500).all() and f.method.nunique()==5
assert not any(n.endswith(('.pkl','.pdf')) for n in entries)
print(json.dumps({'state':'passed','manifest_entries':len(entries),'records_present':len(f),'all_120_method_conditions_have_500_records':True,'source_import_closure_verified':True,'private_questionnaires_or_article_PDFs_present':False}))
