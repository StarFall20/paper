"""Verify saved public evidence without fitting models or network access."""
from pathlib import Path
import hashlib
import json
import csv

ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for line in (ROOT/'MANIFEST.sha256').read_text().splitlines():
    h,n=line.split('  ',1)
    assert sha(ROOT/n)==h,n
s=json.loads((ROOT/'results/summary.json').read_text())
assert s['support_counts']=={'released_power':{'interior':24},'unit_power_sensitivity':{'interior':23,'outside':1}}
assert s['new_LLM_calls']==s['new_MC_runs']==s['new_responses']==s['new_predictors_trained']==0
assert s['maximum_alignment_coefficient_error']<2e-12
assert s['maximum_alignment_power_error']<1e-15
assert s['script_sha256']==sha(ROOT/'audit_released_workflow.py')
v=json.loads((ROOT/'results/independent_verification.json').read_text())
assert v['verified_finite_coefficients']==47 and v['verified_exterior_directions']==1
assert v['ordinary_interior']==8 and v['ordinary_controls_added_after_primary_results']
assert v['maximum_original_three_term_score']<1e-7
assert v['verifier_sha256']==sha(ROOT/'verify_released_workflow.py')
assert len(list(csv.DictReader((ROOT/'results/finite_support.csv').open())))==48
assert len(json.loads((ROOT/'results/support_details.json').read_text()))==48
w=json.loads((ROOT/'wos_search_log.json').read_text())
assert [x['hits'] for x in w['queries']]==[0,37,1,28,7,11,114,49]
assert not w['exhaustive_systematic_review_completed']
assert w['Scopus_queries_this_turn']==w['Google_Scholar_queries_this_turn']==0
print(json.dumps(dict(status='passed',saved_support_cells=48,original_rule_interior=24,unit_sensitivity_exterior=1,ordinary_controls=8,wos_queries=8)))
