from pathlib import Path
import argparse,hashlib,json,subprocess,sys

ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--full',action='store_true');a=p.parse_args()
 out=a.out.resolve();inputs=a.input_dir.resolve()
 if out.exists() and any(out.iterdir()):raise ValueError('Use a fresh output directory')
 out.mkdir(parents=True,exist_ok=True)
 def run(args,name):
  with (out/(name+'.log')).open('w') as log:
   result=subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  if result.returncode:raise RuntimeError(name+' failed: '+(out/(name+'.log')).read_text()[-3500:])
 run([ROOT/'verify_package.py'],'inventory')
 common=['--input-dir',inputs,'--canonical',ROOT,'--reference',ROOT/'reference','--plan',ROOT/'analysis/one_step_power_comparison_plan_20261117.json']
 primary=out/'primary' if a.full else ROOT/'results'
 if a.full:run([ROOT/'analysis/one_step_power_comparison.py',*common,'--out',primary],'primary')
 run([ROOT/'analysis/verify_one_step_power_comparison.py',*common,'--results',primary,'--out',out/'verification'],'verification')
 run([ROOT/'analysis/summarize_one_step_power_comparison.py','--results',primary,'--verification',out/'verification','--out',out/'summary'],'summary')
 verified=json.loads((out/'verification/verification.json').read_text());assert verified['state']=='passed'
 files=['trial_records.csv.gz','pilot_records.csv.gz','summary.csv','status_counts.csv','analytic_checks.json']
 identical={n:sha(primary/n)==sha(ROOT/'results'/n) for n in files} if a.full else {}
 if a.full:assert all(identical.values()),identical
 aggregates={n.name:sha(n)==sha(ROOT/'summary'/n.name) for n in (out/'summary').iterdir()}
 assert all(aggregates.values()),aggregates
 receipt=dict(state='passed',full_primary_replay=a.full,records_verified=verified['records_checked'],legacy_cells_reconciled=verified['legacy_cells_reconciled'],primary_scientific_outputs_byte_identical=identical,summary_outputs_byte_identical=aggregates,new_independent_cohorts=0,new_human_or_AI_answers=0)
 (out/'reproduction_check.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

if __name__=='__main__':main()
