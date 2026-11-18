"""Verify the saved economic-path report and all linked result receipts."""
import csv
import gzip
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load(name):
    return json.loads((ROOT/name).read_text())

listed=set()
for line in (ROOT/'MANIFEST.sha256').read_text().splitlines():
    h,n=line.split('  ',1)
    assert sha(ROOT/n)==h,n
    listed.add(n)
expected={str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and p!=ROOT/'MANIFEST.sha256'
          and not {'local_inputs','regenerated','__pycache__'}.intersection(p.relative_to(ROOT).parts)
          and p.name!='.DS_Store'}
assert listed==expected
s=load('results/status.json')
v=load('verification/verification.json')
g=load('results_gap_spectrum/summary.json')
a=load('path_summary.json')
assert s['state']=='complete' and s['verified_path_points']==s['fitted_path_points']==v['raw_scores_forecasts_and_economics_checked']==7536
assert s['cases']=={'uncapped_positive':107,'capped_low_budget':835,'capped_higher_budget':1000,'zero_or_near_zero':58}
assert sha(ROOT/'plan.json')==s['plan_sha256'] and sha(ROOT/'analysis/audit_path.py')==s['analysis_sha256']
assert sha(ROOT/'analysis/verify_path.py')==v['verifier_sha256'] and v['state']=='passed'
assert all(sha(ROOT/'results'/n)==h for n,h in v['results_sha256'].items())
assert len(load('verification/original_row_endpoints.json'))==v['original_row_endpoints']==107
assert g['state']=='passed' and g['posthoc_path_points']==856 and g['no_retuned_weights']
assert sha(ROOT/'results_gap_spectrum/spectrum.csv')==g['csv_sha256']
assert sha(ROOT/'analysis/check_gap_spectrum.py')==g['analysis_sha256']
assert sha(ROOT/'diagnostics/gap_spectrum_plan.json')==g['source_plan_sha256']
with (ROOT/'results/paths.csv').open(newline='') as stream:
    rows=list(csv.DictReader(stream))
ids=[x[0] for x in json.loads(gzip.decompress((ROOT/'results/coefficients.json.gz').read_bytes()))]
assert [int(r['record']) for r in rows]==ids==list(range(7536))
assert all(r['fit_state']=='verified_numerical_fit' and r['price_state']=='negative' for r in rows)
assert len(a['summaries'])==32 and a['geometry']['strictly_positive_forecast_slope']==107
x=[c for c in a['summaries'] if c['group']=='uncapped_positive' and c['labelled_people']==5]
assert x[0]['holdout_loss_5_50_95'][1]>1.30 and x[1]['holdout_loss_5_50_95'][1]>2.22
assert x[1]['premium']['2.5']['paired']['half_frontier_or_half_unit']['changed']==6
replay=ROOT/'isolated_replay.json'
if replay.exists():
    z=load('isolated_replay.json')
    assert z['state']=='passed' and z['all_numerical_outputs_byte_identical']
print(json.dumps(dict(state='passed',paths=7536,positive_endpoints=107,controls=835,scalar_checks=856,manifest_entries=len(listed))))
