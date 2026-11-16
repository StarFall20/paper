from pathlib import Path
import argparse,gzip,hashlib,json,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--full',action='store_true');a=p.parse_args()
assert not a.out.exists(),'Choose a fresh output directory.'
a.out.mkdir(parents=True)
def run(n,*args):subprocess.run([sys.executable,str(ROOT/'analysis'/n),*map(str,args)],check=True)
plan=ROOT/'analysis/discrete_menu_boundary_plan_20261116.json';source=json.loads((ROOT/'source_record.json').read_text())
if a.full:
 result=a.out/'primary';run('discrete_menu_boundary.py','--repo',ROOT,'--plan',plan,'--out',result)
else:
 result=a.out/'saved_results';result.mkdir()
 for n in ['status.json','summary.csv','paired_with_gaussian.csv']:shutil.copy2(ROOT/'results'/n,result/n)
 chunks=[]
 for part in source['checkpoint_parts']:
  b=(ROOT/'results'/part['file']).read_bytes();assert hashlib.sha256(b).hexdigest()==part['sha256'];chunks.append(b)
 b=b''.join(chunks);assert len(b)==source['checkpoint_bytes'] and hashlib.sha256(b).hexdigest()==source['checkpoint_sha256'];(result/'checkpoint.json.gz').write_bytes(b)
 b=gzip.decompress((ROOT/'results/trials.csv.gz').read_bytes());assert hashlib.sha256(b).hexdigest()==source['trials_csv_sha256'];(result/'trials.csv').write_bytes(b)
run('verify_discrete_menu_boundary.py','--repo',ROOT,'--plan',plan,'--out',result,'--verification-out',a.out/'verification')
run('summarize_discrete_menu_boundary.py','--results',result,'--out',a.out/'summary')
checks={}
for name in ['summary.csv','paired_with_gaussian.csv']:checks[name]=(result/name).read_bytes()==(ROOT/'results'/name).read_bytes()
checks['trials.csv']=hashlib.sha256((result/'trials.csv').read_bytes()).hexdigest()==source['trials_csv_sha256']
checks['checkpoint.json.gz']=hashlib.sha256((result/'checkpoint.json.gz').read_bytes()).hexdigest()==source['checkpoint_sha256']
for folder,names in [('verification',['verification.json','independent_interior_LPs.csv']),('summary',['tableF4.csv','tableF4.md','summary_verification.json'])]:
 for name in names:checks[folder+'/'+name]=(a.out/folder/name).read_bytes()==(ROOT/folder/name).read_bytes()
assert all(checks.values()),checks
(a.out/'reproduction_check.json').write_text(json.dumps(dict(state='passed',full_primary_replay=a.full,scientific_outputs_byte_identical=checks),indent=2)+'\n')
print(json.dumps(dict(state='passed',full_primary_replay=a.full,identical_outputs=len(checks))))
