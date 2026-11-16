"""Check saved menu results, or reconstruct the complete original-cohort follow-up."""
from pathlib import Path
import argparse,gzip,hashlib,json,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--full',action='store_true');a=p.parse_args()
assert not a.out.exists(),'Choose a fresh output directory.'
a.out.mkdir(parents=True)
def run(script,*args):subprocess.run([sys.executable,str(ROOT/'analysis'/script),*map(str,args)],check=True)
plan=ROOT/'analysis/menu_quality_plan_20261115.json'
source=json.loads((ROOT/'source_record.json').read_text())
if a.full:
 result=a.out/'primary'
 run('menu_probability_quality.py','--plan',plan,'--out',result)
else:
 result=a.out/'saved_results';result.mkdir()
 for f in (ROOT/'results').iterdir():
  if f.name.startswith('checkpoint.part') or f.name=='quality_records.csv.gz':continue
  shutil.copy2(f,result/f.name)
 chunks=[]
 for item in source['checkpoint_parts']:
  b=(ROOT/'results'/item['file']).read_bytes();assert hashlib.sha256(b).hexdigest()==item['sha256'];chunks.append(b)
 b=b''.join(chunks);assert len(b)==source['checkpoint_bytes'] and hashlib.sha256(b).hexdigest()==source['checkpoint_sha256']
 (result/'checkpoint.json.gz').write_bytes(b)
 b=gzip.decompress((ROOT/'results/quality_records.csv.gz').read_bytes());assert hashlib.sha256(b).hexdigest()==source['quality_records_csv_sha256']
 (result/'quality_records.csv').write_bytes(b)
run('verify_menu_probability_quality.py','--plan',plan,'--results',result,'--out',a.out/'verification')
run('summarize_menu_quality.py','--results',result,'--out',a.out/'summary')
comparisons={}
for name in ['quality_summary.csv','paired_with_ridge.csv','ridge_fallback.csv','risk_definition.json']:
 comparisons[name]=(result/name).read_bytes()==(ROOT/'results'/name).read_bytes()
comparisons['quality_records.csv']=hashlib.sha256((result/'quality_records.csv').read_bytes()).hexdigest()==source['quality_records_csv_sha256']
for name in ['tableF3.csv','tableF3.md','all_availability_strata.csv','summary_verification.json']:
 comparisons['summary/'+name]=(a.out/'summary'/name).read_bytes()==(ROOT/'summary'/name).read_bytes()
assert all(comparisons.values()),comparisons
(a.out/'reproduction_check.json').write_text(json.dumps({'state':'passed','full_primary_replay':a.full,'scientific_outputs_byte_identical':comparisons},indent=2)+'\n')
print(json.dumps({'state':'passed','full_primary_replay':a.full,'identical_outputs':len(comparisons)}))
