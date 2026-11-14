"""Export vector PDFs using the existing publication plot builders and frozen inputs.

PNG and historical evidence files are not rewritten. No model draws or fits run.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, runpy, sys, tempfile
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--out',type=Path,default=ROOT/'manuscript/figures/publication_pdf')
args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/'analysis'))
matplotlib.rcParams['pdf.fonttype']=42
original=Figure.savefig
created=[]
class FigureComplete(Exception):pass
remaining=0
def vector_save(self,path,*pos,**kw):
 global remaining
 path=Path(path)
 if path.suffix!='.png':return
 target=args.out/(path.stem+'.pdf')
 original(self,target,format='pdf',bbox_inches=kw.get('bbox_inches'),
          metadata={'CreationDate':None,'ModDate':None})
 created.append(target);remaining-=1
 if remaining==0:raise FigureComplete
Figure.savefig=vector_save
builders=[]
def build(name,argv,count):
 global remaining
 source=ROOT/'analysis'/name;builders.append(source);remaining=count
 sys.argv=[str(source),*map(str,argv)]
 try:runpy.run_path(str(source),run_name='__main__')
 except FigureComplete:pass
 assert remaining==0,name
with tempfile.TemporaryDirectory() as td:
 temp=Path(td);(temp/'paper.md').write_text((ROOT/'manuscript/paper.md').read_text())
 build('build_publication_assets.py',['--results',ROOT/'results','--manuscript',temp],2)
 build('plot_variance_reference_audit.py',['--phase',ROOT/'results/score/phase',
       '--population',ROOT/'results/score/population_clean','--out',temp],1)
 build('build_menu_publication_assets.py',['--results',ROOT/'results/menu','--manuscript',temp],1)
 build('write_vanishing_power_report.py',['--results',ROOT/'results','--manuscript',temp],1)
 source=ROOT/'analysis/build_radial_publication_assets.py';builders.append(source)
 spec=importlib.util.spec_from_file_location('radial_vector_builder',source)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 remaining=1
 try:module.plot(ROOT,temp/'figure_radial_auxiliary.png')
 except FigureComplete:pass
 assert remaining==0
Figure.savefig=original
assert len(created)==6
record={'vector_PDFs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in created},
 'plot_builder_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in builders},
 'new_model_fits_or_draws':0,'historical_PNG_or_CSV_files_modified':False,
 'format':'PDF with TrueType font embedding requested (pdf.fonttype42)',
 'visual_inspection_still_required':True}
(args.out/'provenance.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
