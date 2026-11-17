"""Independent high-precision check of the retained analytic comparison."""
from decimal import Decimal as D, localcontext
from fractions import Fraction
from pathlib import Path
import argparse,hashlib,json

def main():
 p=argparse.ArgumentParser();p.add_argument('--certificate',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 saved=json.loads(a.certificate.read_text())
 assert saved['source_sha256']==hashlib.sha256((Path(__file__).parent/'check_existing_design.py').read_bytes()).hexdigest()
 with localcontext()as ctx:
  ctx.prec=80
  xl=[D(2),D(4)];xu=[D(1),D(1)];lam=D(15)/D(88)
  def derivative(beta,weight,gl):
   ul=[beta*x for x in xl];uu=[beta*x for x in xu]
   s=lambda z:D(1)/(D(1)+(-z).exp())
   human=sum(x*(s(u)-D(1))for x,u in zip(xl,ul))/2
   pred_u=sum(x*(s(u)-D('.5'))for x,u in zip(xu,uu))/2
   pred_l=sum(x*(s(u)-g)for x,u,g in zip(xl,ul,gl))/2
   return human+weight*(pred_u-pred_l)
  left=D(0);right=D(8).ln();gl=[D('.9')]*2
  assert derivative(left,lam,gl)<0<derivative(right,lam,gl)
  for _ in range(240):
   mid=(left+right)/2
   if derivative(mid,lam,gl)<0:left=mid
   else:right=mid
  beta=(left+right)/2;score=abs(derivative(beta,lam,gl));assert score<D('1e-65')
  endpoint=[D(25)/44]*2
  reference_error=abs(derivative(D(0),lam,gl)-derivative(D(0),D(1),endpoint));assert reference_error<D('1e-70')
  checks=0
  for b in range(-12,13):
   # Stable supervised logistic scores without subtracting a probability from 1.
   rows=[-x/(D(1)+(x*D(b)).exp())for x in xl]
   assert all(v<0 for v in rows)
   for w in [D(0),D('.01'),D('.25'),D('.5'),D('.75'),D('.99'),D(1)]:
    assert w*rows[0]+(1-w)*rows[1]<0;checks+=1
  gap=Fraction(saved['base_upper_support'])-Fraction(saved['base_corrected_moment']);assert gap==Fraction(3,88)
  result=dict(state='passed',independent_decimal_precision=80,base_root_beta=str(beta),base_root_original_loss_score_residual=str(score),reference_gradient_error=str(reference_error),supervised_weight_score_checks=checks,universal_obstruction_from_sign_proof_not_finite_grid=True,existing_design_count=1,new_independent_cohorts=0,new_human_or_AI_answers=0,upstream_EPI_algorithm_reproduced=False,formal_PPI_A_to_E_compared=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 (a.out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
