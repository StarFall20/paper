"""Verify retained records, or regenerate the full specified simulation."""
from pathlib import Path
import argparse,json,subprocess,sys,os,shutil

ROOT=Path(__file__).resolve().parent


def command(*args):
    subprocess.run([sys.executable,*map(str,args)],check=True,cwd=ROOT)


def compare(expected,actual):
    for k,v in expected.items():
        if isinstance(v,bool) or isinstance(v,str):
            assert actual[k]==v,(k,v,actual[k])
        elif isinstance(v,(int,float)):
            assert abs(actual[k]-v)<(1e-7 if isinstance(v,float) else .5),(k,v,actual[k])


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('regenerated'))
    p.add_argument('--full',action='store_true');a=p.parse_args()
    out=a.out.resolve();assert not out.exists(),'Choose a fresh output directory'
    out.mkdir(parents=True)
    os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
    command('verify_package.py')
    for case,script in [('attributes_only','learned_score_budget_boundary.py'),('proxy','learned_proxy_budget_boundary.py')]:
        supplied=ROOT/'evidence'/case
        if a.full:
            base=out/case;base.mkdir()
            shutil.copy2(supplied/'plan.json',base/'plan.json')
            for mode in ['population','phase']:
                command(ROOT/'analysis'/script,'--plan',base/'plan.json','--out',base/mode,'--mode',mode)
        else:base=supplied
        check=out/(case+'_verification')
        command(ROOT/'analysis/verify_learned_budget.py','--base',base,'--out',check)
        compare(json.loads((supplied/'verification/verification.json').read_text()),json.loads((check/'verification.json').read_text()))
        envelope=out/(case+'_envelope')
        command(ROOT/'analysis/learned_budget_envelope.py','--base',base,'--out',envelope)
        if case=='proxy':
            if a.full:
                audit=base/'all_power_original'
                command(ROOT/'analysis/learned_budget_power_audit.py','--base',base,'--out',audit)
                strict=base/'all_power'
                command(ROOT/'analysis/polish_learned_power_directions.py','--base',base,'--audit',audit,'--out',strict)
            else:strict=supplied/'all_power'
            check=out/'all_power_verification'
            command(ROOT/'analysis/verify_learned_power_audit.py','--base',base,'--audit',strict,'--out',check)
            compare(json.loads((supplied/'all_power_verification/verification.json').read_text()),json.loads((check/'verification.json').read_text()))
    result=dict(status='complete',full_primary_generation=a.full,
        retained_primary_method_records=32000,primary_model_draws=8000,
        secondary_all_power_existing_cohorts=4000,source_package_verified=True,
        originality_or_submission_threshold_certified=False)
    (out/'reproduction_status.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
