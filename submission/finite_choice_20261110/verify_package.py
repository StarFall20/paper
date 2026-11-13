"""Check the isolated archive inventory, checksums, links and local imports."""
from pathlib import Path
import ast,hashlib,re,json
ROOT=Path(__file__).resolve().parent
entries=(ROOT/'MANIFEST.sha256').read_text().splitlines();listed=set()
for line in entries:
 sha,path=line.split('  ',1);f=ROOT/path;assert f.is_file(),path;assert hashlib.sha256(f.read_bytes()).hexdigest()==sha,path;listed.add(path)
actual={str(f.relative_to(ROOT)) for f in ROOT.rglob('*') if f.is_file() and f!=ROOT/'MANIFEST.sha256' and f.name!='.DS_Store' and not {'regenerated','__pycache__'}.intersection(f.relative_to(ROOT).parts)}
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
  if isinstance(node,ast.ImportFrom) and node.module and node.module.startswith(('public_conjoint','finite_ratio','power_tuned','power_domain','attribute_aligned','calibrated_aligned','calibrated_choice','choice_moment','imperfect_predictor','finite_pool','audit_binary','vanishing_power','joint_power','general_joint','verify_general')):assert (ROOT/'analysis'/f'{node.module}.py').exists(),(f,node.module)
assert not any('trial_diagnostics' in x or 'unavailable_cases' in x or x.endswith('.pkl') for x in listed)
menu=json.loads((ROOT/'results/menu/source_hashes.json').read_text())
for path,sha in menu['sha256'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,path
for script in menu['current_entry_points']:assert (ROOT/'analysis'/f'{script}.py').is_file(),script
score=json.loads((ROOT/'results/score/source_hashes.json').read_text())
for path,sha in score['sha256'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,path
for script in score['current_entry_points']:assert (ROOT/'analysis'/f'{script}.py').is_file(),script
for key in ['population','phase']:assert hashlib.sha256((ROOT/score[key+'_plan']).read_bytes()).hexdigest()==score[key+'_plan_sha256']
observed=json.loads((ROOT/'results/observable/source_hashes.json').read_text())
for path,sha in observed['sha256'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,path
for script in observed['current_entry_points']:assert (ROOT/'analysis'/f'{script}.py').is_file(),script
for plan,key in [('analysis/observable_score_plan_20261112.json','plan_sha256'),('analysis/observable_low_label_plan_20261112.json','low_label_plan_sha256')]:assert hashlib.sha256((ROOT/plan).read_bytes()).hexdigest()==observed[key]
assert hashlib.sha256((ROOT/'results/observable/reference_power_weights.csv').read_bytes()).hexdigest()==observed['reference_weights_sha256']
complete=json.loads((ROOT/'results/complete_gmm/source_hashes.json').read_text())
for path,sha in complete['sha256'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,path
risk=json.loads((ROOT/'results/probability_risk/current_source_hashes.json').read_text())
for path,sha in risk['sha256'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,path
paper=(ROOT/'manuscript/paper.md').read_text()
assert (ROOT/'manuscript/main_text.md').read_text()==paper[:paper.index('## Appendix A')]+paper[paper.index('## Data availability'):]
radial=json.loads((ROOT/'results/radial_budget/source_hashes.json').read_text())
for group in ['sha256','plans_sha256','inputs_sha256']:
 for path,sha in radial[group].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,path
assert hashlib.sha256((ROOT/'analysis/build_radial_publication_assets.py').read_bytes()).hexdigest()==radial['publication_builder_sha256']
integration=json.loads((ROOT/'reports/core_revision_integration_20261113.json').read_text())
qa=json.loads((ROOT/'reports/core_revision_visual_qa_20261113.json').read_text())
for name,word_key,source_key,qa_key in [('paper','full_word_sha256','new_manuscript_sha256','full'),('main_text','main_word_sha256','main_text_sha256','main')]:
 assert hashlib.sha256((ROOT/f'manuscript/{name}.md').read_bytes()).hexdigest()==integration[source_key]==qa[qa_key]['source_sha256']
 assert hashlib.sha256((ROOT/f'manuscript/{name}.docx').read_bytes()).hexdigest()==integration[word_key]==qa[qa_key]['docx_sha256']
assert qa['status']=='complete'
assert len(list((ROOT/'analysis').glob('*.py')))==integration['current_science_source_count']==79
assert hashlib.sha256((ROOT/'submission_materials/Highlights.docx').read_bytes()).hexdigest()==qa['highlights']['docx_sha256']
assert hashlib.sha256((ROOT/'submission_materials/highlights.md').read_bytes()).hexdigest()==qa['highlights']['source_sha256']
pair=ROOT/'results/calibrated_pair'
pair_clean=json.loads((pair/'clean_validation.json').read_text())
assert pair_clean['status']=='complete' and pair_clean['scientific_outputs_byte_identical']
for row in pair_clean['sources']:
 assert hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
for row in pair_clean['records']:
 assert hashlib.sha256((pair/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
categorical=ROOT/'results/categorical_face'
for name,sha in json.loads((categorical/'current_source_hashes.json').read_text()).items():
 assert hashlib.sha256((ROOT/'analysis'/name).read_bytes()).hexdigest()==sha,name
face_integration=json.loads((ROOT/'reports/categorical_face_integration_20261113.json').read_text())
assert hashlib.sha256((ROOT/'inputs/categorical_face/existing_public_coefficients.json').read_bytes()).hexdigest()==face_integration['existing_public_coefficients_sha256']
assert integration['all_previous_table_blocks_unchanged'] and integration['previous_display_bodies_preserved']
print(json.dumps({'manifest_entries_verified':len(listed),'broken_local_markdown_links':len(missing),'science_import_closure_present':True,'menu_source_closure_files':len(menu['sha256']),'score_source_closure_files':len(score['sha256']),'observed_reference_source_closure_files':len(observed['sha256']),'radial_source_closure_files':len(radial['sha256']),'current_manuscript_sources_and_reviewed_Word_match':True,'raw_pickle_inputs_shipped':False}))
