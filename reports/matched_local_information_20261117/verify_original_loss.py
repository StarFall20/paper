"""Independent Decimal check of the actual two-parameter compensated loss."""
from decimal import Decimal as D,localcontext
from pathlib import Path
import argparse,hashlib,json
P=Path(__file__).resolve().parent
def number(v):
    n,d=v.split('/') if '/' in v else (v,'1');return D(n)/D(d)
def main(certificate,out):
    with localcontext() as c:
        c.prec=90
        design=json.loads((P/'design.json').read_text());cert=json.loads(certificate.read_text())
        assert cert['state']=='passed' and cert['design_sha256']==hashlib.sha256((P/'design.json').read_bytes()).hexdigest()
        a=number(design['base_power']);L=[(number(r['x']),D(r['y']),number(r['g'])) for r in design['labelled']];U=[(number(r['x']),number(r['g'])) for r in design['auxiliary']]
        def evaluate(alpha,beta,lam):
            val=D(0);grad=[D(0),D(0)];H=[[D(0),D(0)],[D(0),D(0)]];scores=[[],[]]
            # Directly add the human loss and subtract/add both prediction losses.
            # No combined-moment or tanh formula from the primary checker is used.
            terms=[]
            for group,rows in enumerate([L,U]):
                for row in rows:
                    x=row[0];g=D('.5')+a/lam*(row[-1]-D('.5'));z=alpha+x*beta;s=1/(1+(-z).exp());sp=(1+z.exp()).ln();v=[D(1),x]
                    if group==0:
                        y=row[1];w=D(1)/len(L)
                        val+=w*((sp-y*z)-lam*(sp-g*z));d=(s-y)-lam*(s-g);h=(1-lam)*s*(1-s)
                    else:
                        w=D(1)/len(U);val+=w*lam*(sp-g*z);d=lam*(s-g);h=lam*s*(1-s)
                    scores[group].append([t*d for t in v])
                    for j in range(2):
                        grad[j]+=w*v[j]*d
                        for k in range(2):H[j][k]+=w*v[j]*v[k]*h
            return val,grad,H,scores
        def close(x,y,tol=D('1e-82')):assert abs(x-y)<tol,(x,y)
        def slope_fourth(lam):
            result=D(0)
            for group,rows in enumerate([L,U]):
                for row in rows:
                    x=row[0];s=D('.5');coefficient=(1-lam) if group==0 else lam
                    result+=coefficient*x**4*s*(1-s)*(1-6*s+6*s*s)/len(rows)
            return result
        close(slope_fourth(D(1))-slope_fourth(a),number(cert['base_vs_unit_fourth_slope_derivative_difference']))
        checks=0
        for row in cert['records']:
            lam=number(row['power']);v,g,h,scores=evaluate(D(0),D(0),lam)
            close(v,D(2).ln())
            for j in range(2):
                close(g[j],number(row['gradient'][j]))
                for k in range(2):close(h[j][k],number(str(row['Hessian'][j][k])))
            for group,key in enumerate(['labelled_row_scores','auxiliary_row_scores']):
                for actual,expected in zip(scores[group],row[key]):
                    for x,y in zip(actual,expected):close(x,number(str(y)))
            J=[[D(0),D(0)],[D(0),D(0)]]
            for rows in scores:
                n=len(rows);av=[sum(r[j] for r in rows)/n for j in range(2)]
                for j in range(2):
                    for k in range(2):J[j][k]+=sum((r[j]-av[j])*(r[k]-av[k]) for r in rows)/(n*(n-1))
            for j in range(2):
                for k in range(2):close(16*J[j][k],number(str(row['reference_sandwich'][j][k])))
            # Symmetry makes alpha=0 the exact conditional minimizer for every beta.
            for beta in [D('-12'),D('-1'),D('0'),D('1'),D('5'),D('20')]:
                close(evaluate(D(0),beta,lam)[1][0],D(0));checks+=1
        roots=[]
        for lam in [D('.5'),D('.525')]:
            lo,hi=D(0),D(100)
            assert evaluate(D(0),lo,lam)[1][1]<0<evaluate(D(0),hi,lam)[1][1]
            for _ in range(300):
                mid=(lo+hi)/2
                if evaluate(D(0),mid,lam)[1][1]>0:hi=mid
                else:lo=mid
            beta=(lo+hi)/2;v,g,h,_=evaluate(D(0),beta,lam)
            assert max(map(abs,g))<D('1e-85') and h[0][0]>0 and h[1][1]>0
            roots.append(dict(power=str(lam),alpha='0',beta=str(beta),original_gradient_max=str(max(map(abs,g))),loss=str(v)))
        # The original loss at large positive slopes has the exact saved ray slope.
        rays=[]
        for lam in [D('.55'),D('.6'),D(1)]:
            v1=evaluate(D(0),D(200),lam)[0];v2=evaluate(D(0),D(300),lam)[0]
            expected=D(11)/80-lam/4;observed=(v2-v1)/100
            close(observed,expected,D('1e-80'));rays.append(dict(power=str(lam),expected_recession=str(expected),original_loss_secant=str(observed)))
        result=dict(state='passed',precision=90,independent_original_objective_used=True,reference_scores_and_Hessian_and_sandwich_verified=True,fourth_slope_difference_verified=True,conditional_intercept_symmetry_checks=checks,finite_roots=roots,original_loss_rays=rays,universal_conclusions_require_derivation=True,new_deterministic_designs=1,new_independent_cohorts=0,new_human_or_AI_answers=0,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        out.mkdir(parents=True,exist_ok=True);(out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--certificate',type=Path,default=P/'results/exact_information.json');p.add_argument('--out',type=Path,default=P/'verification');ns=p.parse_args();main(ns.certificate,ns.out)
