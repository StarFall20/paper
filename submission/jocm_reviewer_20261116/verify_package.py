"""Check the isolated archive inventory, checksums, links and local imports."""
from pathlib import Path
import ast,hashlib,re,json
ROOT=Path(__file__).resolve().parent
entries=(ROOT/'MANIFEST.sha256').read_text().splitlines();listed=set()
for line in entries:
 sha,path=line.split('  ',1);f=ROOT/path;assert f.is_file(),path;assert hashlib.sha256(f.read_bytes()).hexdigest()==sha,path;listed.add(path)
actual={str(f.relative_to(ROOT)) for f in ROOT.rglob('*') if f.is_file() and f!=ROOT/'MANIFEST.sha256' and f.name!='.DS_Store' and not {'regenerated','regenerated_full','__pycache__'}.intersection(f.relative_to(ROOT).parts)}
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
equation_labels=re.findall(r'\\qquad\s+\(([A-G]\.\d+[a-z]?)\)',paper)
assert len(equation_labels)==len(set(equation_labels)), 'Duplicate appendix equation labels'
assert (ROOT/'manuscript/main_text.md').read_text()==paper[:paper.index('## Appendix A')]+paper[paper.index('## Data availability'):]
radial=json.loads((ROOT/'results/radial_budget/source_hashes.json').read_text())
for group in ['sha256','plans_sha256','inputs_sha256']:
 for path,sha in radial[group].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,path
assert hashlib.sha256((ROOT/'analysis/build_radial_publication_assets.py').read_bytes()).hexdigest()==radial['publication_builder_sha256']
current=json.loads((ROOT/'reports/current_manuscript.json').read_text())
integration=json.loads((ROOT/current['integration']).read_text())
qa=json.loads((ROOT/current['visual_qa']).read_text())
for name,word_key,source_key,qa_key in [('paper','full_word_sha256','new_manuscript_sha256','full'),('main_text','main_word_sha256','main_text_sha256','main')]:
 assert hashlib.sha256((ROOT/f'manuscript/{name}.md').read_bytes()).hexdigest()==integration[source_key]==qa[qa_key]['source_sha256']
 assert hashlib.sha256((ROOT/f'manuscript/{name}.docx').read_bytes()).hexdigest()==integration[word_key]==qa[qa_key]['docx_sha256']
assert qa['status']=='complete'
assert len(list((ROOT/'analysis').glob('*.py')))==integration['current_science_source_count']==83
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
firth_integration=json.loads((ROOT/'reports/firth_manuscript_integration_20261113.json').read_text())
assert firth_integration['unchanged_previous_table_blocks']==11 and firth_integration['previous_table2_rows_preserved']==18 and firth_integration['current_table2_rows']==20
assert integration['old_tables_preserved']==18 and integration['current_table_count']==20
assert integration['all_previous_display_bodies_unchanged']
firth=ROOT/'results/firth'
firth_clean=json.loads((firth/'clean_validation.json').read_text())
assert firth_clean['status']=='complete' and firth_clean['isolated_full_workflow_exit_code']==0
for rel,sha in firth_clean['current_sources'].items(): assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==sha
for row in firth_clean['all_primary_CSVs_byte_identical']: assert hashlib.sha256((firth/row['path']).read_bytes()).hexdigest()==row['sha256']
print(json.dumps({'manifest_entries_verified':len(listed),'broken_local_markdown_links':len(missing),'science_import_closure_present':True,'menu_source_closure_files':len(menu['sha256']),'score_source_closure_files':len(score['sha256']),'observed_reference_source_closure_files':len(observed['sha256']),'radial_source_closure_files':len(radial['sha256']),'current_manuscript_sources_and_reviewed_Word_match':True,'raw_pickle_inputs_shipped':False}))

heldout=ROOT/'results/holdout'
for line in (heldout/'MANIFEST.sha256').read_text().splitlines():
 h,n=line.split('  ',1);assert hashlib.sha256((heldout/n).read_bytes()).hexdigest()==h,n
for n in ['public_holdout_diagnostic.py','verify_public_holdout.py']:
 assert (ROOT/'analysis'/n).read_bytes()==(heldout/'analysis'/n).read_bytes(),n
assert len(re.findall(r'(?:^\|[^\n]*\n)+',paper,re.M))==20

learned=ROOT/'results/learned_training'
for line in (learned/'MANIFEST.sha256').read_text().splitlines():
 h,n=line.split('  ',1); assert hashlib.sha256((learned/n).read_bytes()).hexdigest()==h,n

auxiliary=ROOT/'results/public_auxiliary_budget'
for line in (auxiliary/'MANIFEST.sha256').read_text().splitlines():
 h,n=line.split('  ',1);assert hashlib.sha256((auxiliary/n).read_bytes()).hexdigest()==h,n

menu_quality=ROOT/'results/menu_quality'
for line in (menu_quality/'MANIFEST.sha256').read_text().splitlines():
 h,n=line.split('  ',1); assert hashlib.sha256((menu_quality/n).read_bytes()).hexdigest()==h,n

discrete=ROOT/'results/discrete_menu_boundary'
for line in (discrete/'MANIFEST.sha256').read_text().splitlines():
 h,n=line.split('  ',1);assert hashlib.sha256((discrete/n).read_bytes()).hexdigest()==h,n

one_step=ROOT/'results/one_step_source_audit'
for line in (one_step/'MANIFEST.sha256').read_text().splitlines():
 h,n=line.split('  ',1);assert hashlib.sha256((one_step/n).read_bytes()).hexdigest()==h,n

epi=ROOT/'results/epi_scope'
for line in (epi/'MANIFEST.sha256').read_text().splitlines():
 h,n=line.split('  ',1);assert hashlib.sha256((epi/n).read_bytes()).hexdigest()==h,n
assert json.loads((epi/'verification/verification.json').read_text())['state']=='passed'

matched=ROOT/'results/matched_local_information'
for line in (matched/'MANIFEST.sha256').read_text().splitlines():
 h,n=line.split('  ',1);assert hashlib.sha256((matched/n).read_bytes()).hexdigest()==h,n
matched_record=json.loads((matched/'results/exact_information.json').read_text())
assert matched_record['state']=='passed' and matched_record['exact_critical_power']=='11/20'
assert json.loads((matched/'verification/verification.json').read_text())['state']=='passed'

panel=ROOT/'results/public_panel_dce'
for line in (panel/'MANIFEST.sha256').read_text().splitlines():
 h,n=line.split('  ',1);assert hashlib.sha256((panel/n).read_bytes()).hexdigest()==h,n
panel_v=json.loads((panel/'verification/verification.json').read_text())
assert panel_v['state']=='passed' and panel_v['method_records_reconstructed']==16000
for n,h in panel_v['result_hashes'].items():assert hashlib.sha256((panel/'results'/n).read_bytes()).hexdigest()==h,n

# Bind the current integration to the completed standalone diagnostics.
for name in ['official_ppi_runtime','public_panel_price_ratios']:
 nested=ROOT/'results'/name
 for line in (nested/'MANIFEST.sha256').read_text().splitlines():
  h,n=line.split('  ',1);assert hashlib.sha256((nested/n).read_bytes()).hexdigest()==h,(name,n)
 assert json.loads((nested/'verification/verification.json').read_text())['state']=='passed'
for n,h in integration['scientific_files_unchanged'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
assert integration['preceding_tables_preserved']==20 and integration['current_display_count']==151

# New theory-comparison and bounded-scale evidence is verified without refitting.
for name in ['bounded_scale_scope','direct_theorem_comparison']:
 nested=ROOT/'results'/name
 for line in (nested/'MANIFEST.sha256').read_text().splitlines():
  h,n=line.split('  ',1);assert hashlib.sha256((nested/n).read_bytes()).hexdigest()==h,(name,n)
 if name=='bounded_scale_scope':
  assert json.loads((nested/'results/independent_verification.json').read_text())['status']=='passed'
 else:
  assert json.loads((nested/'results/verification.json').read_text())['status']=='passed'

# The complete independent released-workflow evidence is a dated source audit.
external=ROOT/'results/independent_released_workflow'
for line in (external/'MANIFEST.sha256').read_text().splitlines():
 h,n=line.split('  ',1);assert hashlib.sha256((external/n).read_bytes()).hexdigest()==h,n
e=json.loads((external/'results/summary.json').read_text())
assert e['support_counts']=={'released_power':{'interior':24},'unit_power_sensitivity':{'interior':23,'outside':1}}
assert e['unreconstructed_EST_cells']==24
ev=json.loads((external/'results/independent_verification.json').read_text())
assert ev['verified_finite_coefficients']==47 and ev['verified_exterior_directions']==1 and ev['ordinary_interior']==8

import subprocess, sys

# Verify the complete dated frontier audit and its integration into this archive.
frontier = ROOT / 'results/power_frontier'
completed = subprocess.run([sys.executable, str(frontier / 'verify_package.py')], check=True, capture_output=True, text=True)
assert json.loads(completed.stdout)['independent_coefficient_checks'] == 5732
for n,h in integration['new_evidence_files'].items(): assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest() == h,n
replay=json.loads((ROOT/integration['replay']).read_text())
assert replay['state']=='passed'
assert replay['records']['primary']['original_row_verification']['frontiers_checked']==2000
assert replay['records']['cap']['original_row_verification']['finite_coefficients']==1848


# Bind the endpoint paths and revised author-review material to this release.
endpoint = ROOT / 'results/dce_frontier_economic_path'
checked = subprocess.run([sys.executable, str(endpoint / 'verify_package.py')], check=True, capture_output=True, text=True)
receipt = json.loads(checked.stdout)
assert (receipt['paths'], receipt['positive_endpoints'], receipt['controls'], receipt['scalar_checks']) == (7536, 107, 835, 856)
endpoint_replay = json.loads((ROOT / integration['positive_endpoint_replay']).read_text())
assert endpoint_replay['state'] == 'passed' and endpoint_replay['all_numerical_outputs_byte_identical']
assert len(endpoint_replay['sha256']) == 9
for n, expected in endpoint_replay['sha256'].items():
    assert hashlib.sha256((endpoint / n).read_bytes()).hexdigest() == expected, n
for n, field in [('cover_letter_draft.md', 'source_sha256'), ('Cover_letter.docx', 'docx_sha256')]:
    assert hashlib.sha256((ROOT / 'submission_materials' / n).read_bytes()).hexdigest() == qa['cover_letter'][field], n


# Bind the independent Swissmetro and entropy audits to the current manuscript.
for name in ('swissmetro_corrected_path', 'entropy_endpoint_audit'):
    nested = ROOT/'results'/name
    entries = (nested/'MANIFEST.sha256').read_text().splitlines()
    for line in entries:
        expected, n = line.split('  ',1)
        assert hashlib.sha256((nested/n).read_bytes()).hexdigest()==expected,n
swiss = ROOT/'results/swissmetro_corrected_path/results'
sv = json.loads((swiss/'verification.json').read_text())
assert sv['state']=='passed' and sv['verified']['scores']==1960 and sv['verified']['face_primal_checks']==93
assert hashlib.sha256((swiss/'summary.json').read_bytes()).hexdigest()==sv['summary_sha256']
cost = json.loads((swiss/'label_cost/summary.json').read_text())
assert cost['path_comparisons']==1152 and sum(r['fallback_lower_loss_count'] for r in cost['results'])==0
derivative = ROOT/'results/entropy_endpoint_audit/results'
ev = json.loads((derivative/'summary.json').read_text())
assert ev['records']==856 and ev['statistics'][1]['opposing_derivatives']==107


# Bind the arbitrary local derivative construction to the current manuscript.
local_design = ROOT/'results/local_derivative_attainment'
for line in (local_design/'MANIFEST.sha256').read_text().splitlines():
    expected, n = line.split('  ', 1)
    assert hashlib.sha256((local_design/n).read_bytes()).hexdigest() == expected,n
ld = json.loads((local_design/'results/verification.json').read_text())
assert ld['state']=='passed' and ld['finite_designs_checked']==15
assert ld['exact_orders'][-1]['degree']==33 and ld['near_extremal_orders'][-1]['degree']==25
assert hashlib.sha256((local_design/'results/designs.json').read_bytes()).hexdigest()==ld['designs_sha256']


# Bind the fixed-power score and derivative construction to the current manuscript.
fp = ROOT/'results/fixed_power_identical_scores'
for line in (fp/'MANIFEST.sha256').read_text().splitlines():
    expected, n = line.split('  ', 1)
    assert hashlib.sha256((fp/n).read_bytes()).hexdigest() == expected,n
evaluated = json.loads((fp/'verification.json').read_text())
assert evaluated['status']=='passed' and evaluated['design_cases']==30 and evaluated['max_order']==33
assert hashlib.sha256((fp/'designs.json').read_bytes()).hexdigest()==evaluated['designs_sha256']
prior_design=ROOT/'results/local_derivative_attainment/results/designs.json'
assert hashlib.sha256(prior_design.read_bytes()).hexdigest()==json.loads((fp/'plan.json').read_text())['source_sha256']


# Bind prelabel design and labelled power results to the current manuscript.
sd = ROOT/'results/design_and_safe_power'
for line in (sd/'MANIFEST.sha256').read_text().splitlines():
    expected, n = line.split('  ', 1)
    assert hashlib.sha256((sd/n).read_bytes()).hexdigest()==expected,n
record = json.loads((sd/'independent_replay.json').read_text())
assert record['status']=='passed' and record['design_pairs']==768 and record['positive_witnesses']==656
for path,h in record['four_identical_results'].items():
    assert hashlib.sha256((sd/path).read_bytes()).hexdigest()==h


# Bind the paired finite-fit forecasting experiment to the current manuscript.
forecast_results = ROOT/'results/safe_power_forecast'
for line in (forecast_results/'MANIFEST.sha256').read_text().splitlines():
    expected, n = line.split('  ', 1)
    assert hashlib.sha256((forecast_results/n).read_bytes()).hexdigest()==expected,n
forecast_replay = json.loads((forecast_results/'independent_replay.json').read_text())
assert forecast_replay['status']=='passed' and forecast_replay['shared_paired_designs']==768
for path,h in forecast_replay['byte_identical_outputs'].items():
    assert hashlib.sha256((forecast_results/'results'/path).read_bytes()).hexdigest()==h

# Bind the paired finite-fit forecasting experiment to the current manuscript.
forecast_results = ROOT/'results/safe_power_forecast'
for line in (forecast_results/'MANIFEST.sha256').read_text().splitlines():
    expected, n = line.split('  ', 1)
    assert hashlib.sha256((forecast_results/n).read_bytes()).hexdigest()==expected,n
forecast_replay = json.loads((forecast_results/'independent_replay.json').read_text())
assert forecast_replay['status']=='passed' and forecast_replay['shared_paired_designs']==768
for path,h in forecast_replay['byte_identical_outputs'].items():
    assert hashlib.sha256((forecast_results/'results'/path).read_bytes()).hexdigest()==h
