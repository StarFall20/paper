"""Original-distance and exact original-row verification; no producer import."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,math
from flint import arb,ctx
from scipy.integrate import quad
ctx.prec=256
R=Path(__file__).resolve().parent
def a(x):
    q=F(x);return arb(q.numerator)/q.denominator
def density(x):return (-x*x/2).exp()/(2*arb.pi()).sqrt()
def cdf(x):return (-x/arb(2).sqrt()).erfc()/2
def interval(v,s,t):
    left,right=s*v,t*v
    val=1-(1+left*left)*cdf(left)-left*density(left)
    val-=(1+right*right)*cdf(-right)-right*density(right)
    der=-2*s*(left*cdf(left)+density(left))-2*t*(right*cdf(-right)-density(right))
    return val,der
def original(v,e,tau,power):
    result,der=arb(0),arb(0)
    for w,s,t in [(1-e,e,1/power-1+e),(e,1-e,1/power-e),(tau,-(1-e)/tau,e/tau)]:
        b,d=interval(v,s,t);result+=w*b;der+=w*d
    return result,der
def integrate(v,e,tau,power):
    total,err=0.,0.
    for w,s,t in [(1-e,e,1/power-1+e),(e,1-e,1/power-e),(tau,-(1-e)/tau,e/tau)]:
        left,right=s*v,t*v
        pdf=lambda h: math.exp(-h*h/2)/math.sqrt(2*math.pi)
        for fn,low,high in [(lambda h:(2*left*h-left*left)*pdf(h),-math.inf,left),
                            (lambda h:h*h*pdf(h),left,right),
                            (lambda h:(2*right*h-right*right)*pdf(h),right,math.inf)]:
            val,error=quad(fn,low,high,epsabs=2e-13,epsrel=2e-13,limit=250)
            total+=w*val;err+=w*error
    return total,err
def finite_pair(power):
    lam=F(power);base=F(1,2)
    Xl=[(F(1),F(-1)),(F(1),F(1))];Y=[F(0),F(1)];gl=[F(1,10),F(9,10)]
    Xu=[(F(1),F(x)) for x in [-2,0,0,0,0,0,0,2]];gu=[F(1,4)]+[F(1,2)]*6+[F(3,4)]
    gl=[F(1,2)+base/lam*(g-F(1,2)) for g in gl]
    gu=[F(1,2)+base/lam*(g-F(1,2)) for g in gu]
    assert all(0<g<1 for g in gl+gu)
    for y,g in zip(Y,gl):
        lower=-y+lam*g;upper=1-lam-y+lam*g
        assert lower>0 and upper>0 if y==0 else lower<0 and upper<0
    weights=[(1-lam)/2]*2+[lam/8]*8;X=Xl+Xu
    assert all(w>0 for w in weights)
    moment=[sum(x[j]*(y-lam*g)/2 for x,y,g in zip(Xl,Y,gl))+sum(lam*x[j]*g/8 for x,g in zip(Xu,gu)) for j in range(2)]
    gram=[[sum(w*x[j]*x[k] for w,x in zip(weights,X)) for k in range(2)] for j in range(2)]
    assert gram==[[F(1),F(0)],[F(0),F(1)]]
    assert moment==[F(1,2),F(29,80)]
    positive=sum(w*max(x[1],0) for w,x in zip(weights,X))-moment[1]
    negative=sum(w*max(-x[1],0) for w,x in zip(weights,X))+moment[1]
    assert positive==(11-20*lam)/80 and negative==(69-20*lam)/80
    # Reflection symmetry gives the unique conditional intercept minimum at zero.
    # On that line F_lambda is strictly increasing. A rational sigmoid witness
    # at beta=log(64) brackets its root when lambda=1/2.
    z=F(64);gradient_at_log64=(1-lam)/2*(z-1)/(z+1)+lam/4*(z*z-1)/(z*z+1)-F(29,80)
    assert positive>0 and gradient_at_log64>0 if lam==F(1,2) else positive<0
    return dict(power=power,positive_recession=str(positive),negative_recession=str(negative),
        positive_weight_Gram=[[str(v) for v in row] for row in gram],moment=[str(x) for x in moment],
        gradient_at_log64=str(gradient_at_log64),
        strict_attainment='unique_finite' if positive>0 else 'unbounded_below',
        all_row_losses_strictly_convex=True,labelled_categories=['decreasing' if y else 'increasing' for y in Y],
        auxiliary_categories=['coercive']*8,predictions_strictly_interior=True,
        reference_Hessian=[[str(v/4) for v in row] for row in gram])
def main():
    out=R/'results';plan=json.loads((R/'plan.json').read_text());pair=plan['population_pair']
    rows=json.loads((out/'population_pair.json').read_text());assert [r['power'] for r in rows]==pair['powers']
    e,tau,k=map(a,[pair['epsilon'],pair['tau'],pair['witness_dimension_ratio']]);checks=[]
    for row in rows:
        power=a(row['power']);lo,hi=map(a,row['optimizer_bracket_fraction']);mid=(lo+hi)/2
        val,der=original(mid,e,tau,power);lower,upper=map(a,row['boundary_interval_fraction'])
        assert original(lo,e,tau,power)[1]>0 and original(hi,e,tau,power)[1]<0
        assert lower<val and val+der*der/(2*e*(1-e))<upper
        assert lower>k if row['power']=='1/4' else upper<k
        value,error=integrate(float(mid),float(e),float(tau),float(power))
        discrepancy=abs(value-float(val));assert discrepancy<max(2e-12,20*error)
        checks.append(dict(power=row['power'],original_distance_integral=value,quadrature_error_diagnostic=error,
            difference_from_interval_value=discrepancy,derivative_bracket_verified=True,global_enclosure_verified=True,
            exact_saved_bounds=True))
    finite=[finite_pair(p) for p in plan['exact_finite_pair']['powers']]
    sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    provenance=json.loads((out/'provenance.json').read_text())
    assert provenance['plan_sha256']==sha(R/'plan.json') and provenance['producer_sha256']==sha(R/'calculate.py')
    result=dict(status='passed',independent_implementation=True,external_review=False,producer_imported=False,
        moment_precision_bits=ctx.prec,original_distance_checks=checks,exact_finite_checks=finite,
        inputs={str(f.relative_to(R)):sha(f) for f in [R/'plan.json',R/'calculate.py',out/'population_pair.json',out/'provenance.json']},
        verifier_sha256=sha(Path(__file__)),new_model_fits=0,new_Monte_Carlo_draws=0)
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
if __name__=='__main__':main()
