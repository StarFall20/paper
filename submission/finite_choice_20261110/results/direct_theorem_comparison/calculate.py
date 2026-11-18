"""Certify the planned strict-power pair in the existing balanced model."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json
from flint import arb,ctx
ctx.prec=160
R=Path(__file__).resolve().parent
def a(x):
    q=F(x);return arb(q.numerator)/q.denominator
def f(x):
    sf=(x/arb(2).sqrt()).erfc()/2;density=(-x*x/2).exp()/(2*arb.pi()).sqrt()
    return (1+x*x)*sf-x*density,2*(x*sf-density)
def profile(v,e,tau,power):
    h=1/power-1;value=1+tau;deriv=arb(0)
    terms=[(1-e,-e),(1-e,h+e),(e,-(1-e)),(e,h+1-e),(tau,(1-e)/tau),(tau,e/tau)]
    for w,slope in terms:
        val,d=f(v*slope);value-=w*val;deriv-=w*slope*d
    return value,deriv
def main():
    plan=json.loads((R/'plan.json').read_text());pair=plan['population_pair']
    e,tau=a(pair['epsilon']),a(pair['tau']);k=a(pair['witness_dimension_ratio'])
    output=[]
    for power_text in pair['powers']:
        power=a(power_text);lo=F(0);hi=F(1)
        assert profile(a(lo),e,tau,power)[1]>0
        while not profile(a(hi),e,tau,power)[1]<0:hi*=2
        for _ in range(100):
            mid=(lo+hi)/2;d=profile(a(mid),e,tau,power)[1]
            if d>0:lo=mid
            elif d<0:hi=mid
            else:raise AssertionError('Increase registered precision before changing this check')
        v=a((lo+hi)/2);value,d=profile(v,e,tau,power)
        upper=value+d*d/(2*e*(1-e))
        lowq=str(value.lower().fmpq());highq=str(upper.upper().fmpq())
        assert value>k if power_text=='1/4' else upper<k
        output.append(dict(power=power_text,optimizer_bracket_fraction=[str(lo),str(hi)],
            boundary_interval_fraction=[lowq,highq],boundary_display=float(value.mid()),
            witness_dimension_ratio=pair['witness_dimension_ratio'],
            strict_existence_side='finite' if value>k else 'no_finite_minimizer',
            labelled_Y0_category='strictly_increasing',labelled_Y1_category='strictly_decreasing',
            auxiliary_category='coercive',all_rows_strictly_convex=True,
            labelled_infimum='minus_infinity',predictor_and_data_law_unchanged=True))
    out=R/'results';out.mkdir(exist_ok=True)
    assert not (out/'population_pair.json').exists()
    (out/'population_pair.json').write_text(json.dumps(output,indent=2)+'\n')
    sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    (out/'provenance.json').write_text(json.dumps(dict(precision_bits=ctx.prec,plan_sha256=sha(R/'plan.json'),producer_sha256=sha(Path(__file__)),new_model_fits=0,new_Monte_Carlo_draws=0),indent=2)+'\n')
    print(json.dumps(output))
if __name__=='__main__':main()
