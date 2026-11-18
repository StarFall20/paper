"""Deterministic Arb enclosures for the existing balanced scale model."""
from pathlib import Path
from fractions import Fraction as F
import csv,hashlib,json,math
from flint import arb,ctx
ctx.prec=160
ROOT=Path(__file__).resolve().parent
def ball(x):
    if isinstance(x,F):return arb(str(x.numerator)+'/'+str(x.denominator))
    return arb(x)
def lower(x):return math.nextafter(float(x.lower()),-math.inf)
def upper(x):return math.nextafter(float(x.upper()),math.inf)
def bounds(x):return dict(lower=lower(x),upper=upper(x),arb=str(x))
SQRT2=arb(2).sqrt();NORM=(2*arb.pi()).sqrt()
def tail(x):
    sf=(x/SQRT2).erfc()/2;pdf=(-x*x/2).exp()/NORM
    return (1+x*x)*sf-x*pdf,2*(x*sf-pdf)
def profile(v,e,tau):
    h=1/tau
    terms=[(1-e,-e),(1-e,h+e),(e,-(1-e)),(e,h+1-e),(tau,(1-e)/tau),(tau,e/tau)]
    j=1+tau;d=arb(0)
    for w,a in terms:
        f,fp=tail(a*v);j-=w*f;d-=w*a*fp
    return j,d
def mixture(v,r,e,tau):
    root=r.sqrt();scales=[1/root,root];weights=[r/(1+r),1/(1+r)]
    j=arb(0);d=arb(0)
    for s,w in zip(scales,weights):
        a,b=profile(s*v,e,tau);j+=w*a;d+=w*s*b
    return j,d
def solve(rq,eq,tq):
    r,e,tau=map(ball,[rq,eq,tq]);c=e*(1-e)
    lo=F(0);hi=F(1)
    assert mixture(ball(lo),r,e,tau)[1]>0
    while not mixture(ball(hi),r,e,tau)[1]<0:hi*=2
    for _ in range(90):
        mid=(lo+hi)/2;d=mixture(ball(mid),r,e,tau)[1]
        if d>0:lo=mid
        elif d<0:hi=mid
        else:raise AssertionError('Root sign unresolved at registered precision')
    v=ball((lo+hi)/2);j,d=mixture(v,r,e,tau)
    up=j+d*d/(2*c)
    return dict(scale_ratio=str(rq),epsilon=str(eq),tau=str(tq),power=str(tq/(1+tq)),optimizer_interval_fraction=[str(lo),str(hi)],optimizer=float(v.mid()),boundary_lower=lower(j),boundary_upper=upper(up),boundary_interval=[str(j.lower()),str(up.upper())],derivative_interval=str(d),strong_concavity=str(c))
def main():
    plan=json.loads((ROOT/'plan.json').read_text());out=ROOT/'results';out.mkdir(exist_ok=True)
    assert not (out/'boundaries.json').exists(),'Use a fresh copy for a rerun.'
    rows=[];ends=[]
    for eq in map(F,plan['proxy_error_rates']):
        e=ball(eq);c=arb.pi()*e*(1-e);rs=(1+(1-c).sqrt())/(1-(1-c).sqrt())
        for rq in map(F,plan['scale_ratios']):
            r=ball(rq);rho=4*r/(1+r)**2;large=rho/(2*arb.pi()*e*(1-e))
            reversal=large<ball('1/2');no_reversal=large>=ball('1/2');assert reversal or no_reversal
            ends.append(dict(scale_ratio=str(rq),epsilon=str(eq),sharp_minimum_moment_ratio=bounds(rho),critical_support_ratio=bounds(rs),small_auxiliary_endpoint='1/2',large_auxiliary_endpoint=bounds(large),strict_endpoint_reversal=bool(reversal)))
            for tq in map(F,plan['auxiliary_ratios']):rows.append(solve(rq,eq,tq))
    (out/'boundaries.json').write_text(json.dumps(rows,indent=2)+'\n');(out/'endpoints.json').write_text(json.dumps(ends,indent=2)+'\n')
    with (out/'boundaries.csv').open('w',newline='') as f:
        keys=['scale_ratio','epsilon','tau','power','optimizer','boundary_lower','boundary_upper'];w=csv.DictWriter(f,keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in rows)
    candidate=[x for x in rows if x['scale_ratio']=='4' and x['epsilon']=='2/5'];print(json.dumps(dict(records=len(rows),endpoint_records=len(ends),candidate=candidate),indent=2))
    (out/'provenance.json').write_text(json.dumps(dict(precision_bits=ctx.prec,plan_sha256=hashlib.sha256((ROOT/'plan.json').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),numerical_model_fits=0,Monte_Carlo_runs=0,new_answers=0,all_196_conditions_retained=True),indent=2)+'\n')
if __name__=='__main__':main()
