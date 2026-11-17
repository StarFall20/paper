"""Check the saved profile through original distance integrals; no producer import."""
from pathlib import Path
from fractions import Fraction
import csv, hashlib, itertools, json, math
from flint import arb, ctx
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parent
ctx.prec = 256
def number(x):
    f = Fraction(x)
    return arb(f.numerator) / f.denominator
def normal(x):
    return (-x*x/2).exp() / (2*arb.pi()).sqrt()
def cdf(x):
    return (-x/arb(2).sqrt()).erfc()/2
def original_interval(v, left_slope, right_slope):
    a, b = left_slope*v, right_slope*v
    # Direct truncated moments of the squared distance from H to [a,b].
    below = (1+a*a)*cdf(a)+a*normal(a)
    above = (1+b*b)*cdf(-b)-b*normal(b)
    value = 1-below-above
    derivative = -2*left_slope*(a*cdf(a)+normal(a))
    derivative -= 2*right_slope*(b*cdf(-b)-normal(b))
    return value, derivative
def original(v, r, e, tau):
    value, derivative = arb(0), arb(0)
    for scale, prob in [(1/r.sqrt(), r/(1+r)), (r.sqrt(), 1/(1+r))]:
        intervals = [(1-e, scale*e, scale*(1/tau+e)),
                     (e, scale*(1-e), scale*(1/tau+1-e)),
                     (tau, -scale*(1-e)/tau, scale*e/tau)]
        for weight, a, b in intervals:
            j, d = original_interval(v, a, b)
            value += prob*weight*j
            derivative += prob*weight*d
    return value, derivative
def quadrature(v, r, e, tau):
    total, error = 0., 0.
    for scale, prob in [(1/math.sqrt(r), r/(1+r)), (math.sqrt(r), 1/(1+r))]:
        intervals = [(1-e, scale*e*v, scale*(1/tau+e)*v),
                     (e, scale*(1-e)*v, scale*(1/tau+1-e)*v),
                     (tau, -scale*(1-e)*v/tau, scale*e*v/tau)]
        for weight, a, b in intervals:
            density = lambda h: math.exp(-h*h/2)/math.sqrt(2*math.pi)
            pieces = [(lambda h: (2*a*h-a*a)*density(h), -math.inf, a),
                      (lambda h: h*h*density(h), a, b),
                      (lambda h: (2*b*h-b*b)*density(h), b, math.inf)]
            for fn, low, high in pieces:
                ans, err = quad(fn, low, high, epsabs=2e-13, epsrel=2e-13, limit=250)
                total += prob*weight*ans
                error += abs(prob*weight)*err
    return total, error
def main():
    out=ROOT/'results';plan=json.loads((ROOT/'plan.json').read_text())
    rows=json.loads((out/'boundaries.json').read_text())
    ends=json.loads((out/'endpoints.json').read_text())
    provenance=json.loads((out/'provenance.json').read_text())
    sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    assert provenance['source_sha256']==sha(ROOT/'evaluate.py')
    assert provenance['plan_sha256']==sha(ROOT/'plan.json')
    expected=set(itertools.product(plan['scale_ratios'],plan['proxy_error_rates'],plan['auxiliary_ratios']))
    # Fractions are canonically written by the producer.
    expected={tuple(str(Fraction(x)) for x in t) for t in expected}
    assert {(x['scale_ratio'],x['epsilon'],x['tau']) for x in rows}==expected
    assert len(rows)==len(expected)==196
    checks=[]
    for row in rows:
        r,e,tau=map(number,[row['scale_ratio'],row['epsilon'],row['tau']])
        lo,hi=map(number,row['optimizer_interval_fraction']);mid=(lo+hi)/2
        value,deriv=original(mid,r,e,tau)
        assert original(lo,r,e,tau)[1]>0 and original(hi,r,e,tau)[1]<0
        # Printed Arb endpoint strings are themselves enclosing intervals.
        # Use their outer edges, rather than treating their midpoints as exact.
        lower=arb(row['boundary_interval'][0]).lower()
        upper=arb(row['boundary_interval'][1]).upper()
        assert lower<=value<=upper
        assert value+deriv*deriv/(2*e*(1-e))<=upper
        direct,err=quadrature(float(mid),float(r),float(e),float(tau))
        delta=abs(direct-float(value))
        assert delta<=max(2e-11,20*err)
        assert row['boundary_lower']<=float(value)<=row['boundary_upper']
        checks.append(dict(scale_ratio=row['scale_ratio'],epsilon=row['epsilon'],tau=row['tau'],
                           original_integral=direct,quadrature_error_estimate=err,
                           absolute_difference=delta,independent_256_bit_value=str(value),
                           derivative_bracket_verified=True,arb_enclosure_verified=True))
    assert len(ends)==28
    for row in ends:
        r,e=map(number,[row['scale_ratio'],row['epsilon']])
        a,b=1/r.sqrt(),r.sqrt();p=r/(1+r)
        mean=p*a+(1-p)*b;second=p*a*a+(1-p)*b*b
        assert abs(second-1)<arb('1e-72')
        rho=mean*mean/second
        assert abs(rho-4*r/(1+r)**2)<arb('1e-72')
        endpoint=rho/(2*arb.pi()*e*(1-e))
        root=(1-arb.pi()*e*(1-e)).sqrt()
        critical=(1+root)/(1-root)
        for key,value in [('sharp_minimum_moment_ratio',rho),('large_auxiliary_endpoint',endpoint),('critical_support_ratio',critical)]:
            assert row[key]['lower']<=value<=row[key]['upper']
        assert bool(endpoint<arb('1/2'))==bool(r>critical)==row['strict_endpoint_reversal']
    with (out/'independent_integrals.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,checks[0].keys());w.writeheader();w.writerows(checks)
    result=dict(status='passed',independent_implementation=True,external_peer_review=False,
        producer_imported=False,conditions_checked=len(rows),endpoint_cases_checked=len(ends),
        independent_truncated_moment_precision_bits=ctx.prec,derivative_brackets_verified=196,
        original_distance_integrals_checked=196,arb_global_enclosures_checked=196,
        max_quadrature_absolute_difference=max(x['absolute_difference'] for x in checks),
        max_quadrature_error_estimate=max(x['quadrature_error_estimate'] for x in checks),
        quadrature_diagnostics_are_not_interval_certificates=True,
        numerical_model_fits=0,Monte_Carlo_runs=0,
        inputs={str(f.relative_to(ROOT)):sha(f) for f in [ROOT/'plan.json',ROOT/'evaluate.py',out/'boundaries.json',out/'endpoints.json',out/'provenance.json']},
        verifier_sha256=sha(Path(__file__)))
    (out/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='inputs'}))
if __name__=='__main__':main()
