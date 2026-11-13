from pathlib import Path
import argparse, hashlib, json, subprocess, sys

ROOT = Path(__file__).resolve().parent
p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, default=Path('regenerated')); a = p.parse_args()
out = a.out.resolve(); out.mkdir(parents=True, exist_ok=True)
for line in (ROOT/'MANIFEST.sha256').read_text().splitlines():
    expected, rel = line.split('  ', 1)
    assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == expected, rel
base = ROOT/'evidence'; results = out/'results'; verify = out/'verification'
subprocess.run([sys.executable, str(ROOT/'analysis/firth_choice_comparator.py'), '--inputs', str(base/'inputs'), '--plan', str(base/'plan.json'), '--out', str(results)], check=True)
subprocess.run([sys.executable, str(ROOT/'analysis/verify_firth_choice_comparator.py'), '--inputs', str(base/'inputs'), '--plan', str(base/'plan.json'), '--results', str(results), '--out', str(verify)], check=True)
csvs = sorted((base/'results').glob('*.csv'))
for source in csvs:
    assert source.read_bytes() == (results/source.name).read_bytes(), source.name
assert json.loads((base/'results/run.json').read_text()) == json.loads((results/'run.json').read_text())
for name in ['numeric_checks.json', 'checks.json']:
    assert json.loads((base/'verification_complete'/name).read_text()) == json.loads((verify/name).read_text()), name
assert (base/'verification_complete/direct_objective_crosschecks.csv').read_bytes() == (verify/'direct_objective_crosschecks.csv').read_bytes()
print(json.dumps({'all_primary_CSVs_byte_identical': len(csvs), 'run_JSON_semantically_identical': True, 'independent_verification_outputs_identical': 3, 'goal_complete': False}))
