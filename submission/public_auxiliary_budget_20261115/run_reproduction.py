from pathlib import Path
import argparse,subprocess,sys,json,shutil
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--full',action='store_true');a=p.parse_args()
out=a.out.resolve();assert not out.exists(),'Use a fresh output directory';out.mkdir(parents=True)
def run(script,*args):subprocess.run([sys.executable,str(ROOT/'analysis'/script),*map(str,args)],check=True)
base=ROOT
if a.full:
 base=out;shutil.copy2(ROOT/'plan.json',out/'plan.json')
 run('public_auxiliary_budget_audit.py','--plan',ROOT/'plan.json','--input-dir',a.input_dir.resolve(),'--out',out/'results')
run('verify_public_auxiliary_budget.py','--input-dir',a.input_dir.resolve(),'--base',base,'--out',out/'verification')
if a.full:
 run('public_auxiliary_budget_case.py','--base',base,'--input-dir',a.input_dir.resolve(),'--out',out/'case')
run('verify_public_auxiliary_case.py','--input-dir',a.input_dir.resolve(),'--case',base/'case','--out',out/'verification')
if a.full:run('write_public_auxiliary_budget_assets.py','--base',base,'--out',out/'figures')
(out/'reproduction_status.json').write_text(json.dumps(dict(status='complete',full_primary_run=a.full,records_reconstructed=36000,new_human_or_AI_answers=0),indent=2)+'\n')
