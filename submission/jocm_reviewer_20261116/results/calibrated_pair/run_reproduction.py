"""Regenerate the new scientific outputs in a separate directory."""
from pathlib import Path
import argparse, hashlib, json, os, subprocess, sys

root=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=root/'regenerated');a=p.parse_args()
out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
env['PYTHONPATH']=str(root/'analysis')+os.pathsep+env.get('PYTHONPATH','')
commands=[['calibrated_choice_pair.py','--out',str(out/'results')],
          ['verify_calibrated_choice_pair.py','--input',str(out/'results'),'--out',str(out/'verification')]]
for command in commands:
    subprocess.run([sys.executable,str(root/'analysis'/command[0]),*command[1:]],env=env,check=True)
rows=[]
for directory in ['results','verification']:
    for original in sorted((root/directory).glob('*')):
        new=out/directory/original.name
        assert new.is_file(),new
        expected=hashlib.sha256(original.read_bytes()).hexdigest()
        actual=hashlib.sha256(new.read_bytes()).hexdigest()
        assert expected==actual,(str(original.relative_to(root)),expected,actual)
        rows.append({'path':str(original.relative_to(root)),'sha256':actual})
receipt={'scientific_files':len(rows),'byte_identical':True,'records':rows,'claim':'Regeneration and independent verification, not a novelty or publication certificate.'}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'scientific_files':len(rows),'byte_identical':True}))
