"""Rational interval bounds for the null-signal oracle-power counterexample.

Only Python integers and Fractions enter the authoritative inequalities.
Normal values use alternating Taylor bounds and Machin's formula for pi.
"""
from fractions import Fraction as F
from math import factorial,isqrt
from pathlib import Path
import argparse,hashlib,json


class I:
    def __init__(self,lo,hi=None):self.lo=F(lo);self.hi=F(lo if hi is None else hi);assert self.lo<=self.hi
    def __add__(self,other):
        other=as_i(other);return I(self.lo+other.lo,self.hi+other.hi)
    __radd__=__add__
    def __neg__(self):return I(-self.hi,-self.lo)
    def __sub__(self,other):return self+-as_i(other)
    def __rsub__(self,other):return as_i(other)+-self
    def __mul__(self,other):
        other=as_i(other);v=[self.lo*other.lo,self.lo*other.hi,self.hi*other.lo,self.hi*other.hi];return I(min(v),max(v))
    __rmul__=__mul__
    def __truediv__(self,other):
        other=as_i(other);assert not other.lo<=0<=other.hi;return self*I(1/other.hi,1/other.lo)


def as_i(x):return x if isinstance(x,I) else I(x)


def arctan_bounds(x,N):
    assert 0<x<1 and N%2==0
    upper=sum(((-1)**k*x**(2*k+1)/F(2*k+1) for k in range(N+1)),F(0))
    lower=upper-x**(2*N+3)/F(2*N+3)
    return I(lower,upper)


def sqrt_bounds(x):
    assert x.lo>0;scale=10**60
    low=isqrt((x.lo*scale*scale).numerator//(x.lo*scale*scale).denominator)
    high=isqrt((x.hi*scale*scale).numerator//(x.hi*scale*scale).denominator)+1
    result=I(F(low,scale),F(high,scale));assert result.lo**2<=x.lo and result.hi**2>=x.hi
    return result


PI=16*arctan_bounds(F(1,5),50)-4*arctan_bounds(F(1,239),16)
C=sqrt_bounds(I(1)/(2*PI))


def exp_negative_bounds(z,N=60):
    assert z>=0 and N%2==0 and z/F(N+1)<1
    upper=sum(((-1)**k*z**k/F(factorial(k)) for k in range(N+1)),F(0))
    return I(upper-z**(N+1)/F(factorial(N+1)),upper)


def phi(x):return C*exp_negative_bounds(x*x/2)


def Phi(x,N=60):
    if x<0:return 1-Phi(-x,N)
    assert x*x/2/F(N+1)<1 and N%2==0
    upper=sum(((-1)**k*x**(2*k+1)/F(2**k*factorial(k)*(2*k+1)) for k in range(N+1)),F(0))
    lower=upper-x**(2*N+3)/F(2**(N+1)*factorial(N+1)*(2*N+3))
    return F(1,2)+C*I(lower,upper)


def tail(x):return (1+x*x)*Phi(-x)-x*phi(x)
def derivative(x):return 2*(x*Phi(-x)-phi(x))


def record(x):
    return dict(lower_rational=str(x.lo),upper_rational=str(x.hi),lower_approximate=float(x.lo),upper_approximate=float(x.hi))


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    # Shrunk prediction has confidence .2 and Brier risk .13.
    risk=F(13,100)
    J1=risk+tail(F(7,10))+tail(F(3,10))
    D1=2*risk+F(7,10)*derivative(F(7,10))+F(3,10)*derivative(F(3,10))
    maximum_derivative_squared=max(D1.lo*D1.lo,D1.hi*D1.hi)
    minimum_J_lower=J1.lo-maximum_derivative_squared/(4*risk)
    shrunk_boundary_upper=1-minimum_J_lower
    # Unshrunk lambda=.5: evaluate the scalar criterion at rescaled v=1.
    criterion=F(9,10)*(tail(F(-1,10))+tail(F(11,10)))+F(1,10)*(tail(F(-9,10))+tail(F(19,10)))+tail(F(9,10))+tail(F(1,10))
    oracle_boundary_lower=2-criterion.hi
    assert shrunk_boundary_upper<F(47,100)<F(1,2)<oracle_boundary_lower
    assert C.hi-C.lo<F(1,10**50)
    result=dict(state='certified_with_exact_rational_arithmetic',gamma=0,epsilon='1/10',tau=1,
        oracle_strength_power=['1','1/2'],shrunk_strength_power=['1/2','1'],
        dimension_ratio_witness='47/100',ordinary_boundary='1/2',
        shrunk_boundary_upper_rational=str(shrunk_boundary_upper),shrunk_boundary_upper_approximate=float(shrunk_boundary_upper),
        oracle_boundary_lower_rational=str(oracle_boundary_lower),oracle_boundary_lower_approximate=float(oracle_boundary_lower),
        full_fixed_dimensional_covariance_ratio='17/25',
        normal_constant_interval=record(C),unit_criterion_at_one=record(J1),unit_criterion_derivative_at_one=record(D1),
        oracle_criterion_at_one=record(criterion),
        argument='The shrunk unit criterion has second derivative at least2R,so inf J >= J(1)-Jprime(1)^2/(4R). The unshrunk criterion minimum is at most its value at v1. Rational interval bounds separate both from the ordinary boundary.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Certifies this population inequality using the manuscript boundary theorem. Does not independently certify the theorem itself or originality.')
    (a.out/'rational_counterexample_certificate.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k in ['state','shrunk_boundary_upper_approximate','oracle_boundary_lower_approximate','dimension_ratio_witness','ordinary_boundary','full_fixed_dimensional_covariance_ratio']}))


if __name__=='__main__':main()
