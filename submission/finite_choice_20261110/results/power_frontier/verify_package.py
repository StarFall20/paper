"""Check packaged saved evidence; no model fitting, inputs or network required."""
from pathlib import Path
import csv,gzip,hashlib,json
ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for line in (ROOT/'MANIFEST.sha256').read_text().splitlines():
    h,n=line.split('  ',1);assert sha(ROOT/n)==h,n
for n,r in json.loads((ROOT/'input_provenance.json').read_text()).items():assert sha(ROOT/n)==r['sha256'],n
s=json.loads((ROOT/'results/status.json').read_text());assert s['state']=='complete'
assert s['frontiers']==2000 and s['policy_records']==4000 and s['available_coefficients']==3884
assert s['analysis_sha256']==sha(ROOT/'analysis/run_frontier.py') and s['plan_sha256']==sha(ROOT/'plan.json')
v=json.loads((ROOT/'verification/verification.json').read_text());assert v['state']=='passed'
assert v['frontiers_checked']==2000 and v['policy_records_checked']==4000 and v['finite_coefficients_checked']==3884
assert v['verifier_sha256']==sha(ROOT/'analysis/verify_frontier.py')
for n,h in v['result_hashes'].items():assert sha(ROOT/'results'/n)==h,n
assert len(json.loads(gzip.decompress((ROOT/'results/certificates.json.gz').read_bytes())))==2000
fr=list(csv.DictReader((ROOT/'results/frontiers.csv').open()));assert len(fr)==2000
for budget,count in [(5,443),(10,499),(20,500),(40,500)]:
    part=[r for r in fr if int(r['labelled_people'])==budget];assert len(part)==500
    assert sum(r['midpoint_state']=='interior' for r in part)==count
    assert all(0<=float(r['frontier'])<=1 for r in part)
ss=json.loads((ROOT/'results_cap_only/status.json').read_text());vv=json.loads((ROOT/'verification_cap_only/verification.json').read_text())
assert ss['state']=='complete' and vv['state']=='passed'
assert ss['new_model_fits']==32 and ss['reused_verified_coefficients']==1816 and ss['available_coefficients']==1848
assert ss['analysis_sha256']==sha(ROOT/'analysis/run_cap_only.py') and ss['amendment_sha256']==sha(ROOT/'cap_only_amendment.json')
assert vv['verifier_sha256']==sha(ROOT/'analysis/verify_cap_only.py') and vv['zero_powers_preserved']
for n,h in vv['result_hashes'].items():assert sha(ROOT/'results_cap_only'/n)==h,n
for r,count in zip(json.loads((ROOT/'results_cap_only/summary.json').read_text()),[353,495,500,500]):assert r['available']==count
e=json.loads((ROOT/'results/framework_embedding.json').read_text());assert e['state']=='passed' and len(e['cases'])==12
l=json.loads((ROOT/'literature_followup.json').read_text());assert [r['displayed_hits'] for r in l['queries']]==[82,17,454,108,1,4]
assert l['Scopus']['actual_document_queries']==0
print(json.dumps(dict(state='passed',frontiers=2000,planned_policy_records=4000,posthoc_control_records=2000,
    new_model_fits=3916,reused_coefficients_checked=1816,independent_coefficient_checks=5732,
    scholar_queries=6,Scopus_queries=0,manuscript_changed=False,goal_complete=False)))
