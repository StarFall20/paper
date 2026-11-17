"""Arb enclosures for an existing three-task compensated-logit comparison."""
from pathlib import Path
from fractions import Fraction
import argparse,json,math
import numpy as np
from scipy.special import ndtr,ndtri
from flint import arb,ctx
ctx.prec=160

def ball(q):
    if isinstance(q,Fraction):return arb(str(q.numerator)+'/'+str(q.denominator))
    return arb(q)
def exact_string(x):
    a,b=x.man_exp();return str(int(a))+'*2^'+str(int(b))
def bounds(x):return [str(x.lower()),str(x.upper())]
def up(x):return math.nextafter(float(x.upper()),math.inf)

class Dual:
    def __init__(self,x,d=0):self.x=ball(x);self.d=ball(d)
    def __add__(self,o):
        o=o if isinstance(o,Dual) else Dual(o);return Dual(self.x+o.x,self.d+o.d)
    __radd__=__add__
    def __neg__(self):return Dual(-self.x,-self.d)
    def __sub__(self,o):return self+-o if isinstance(o,Dual) else self+Dual(-ball(o))
    def __rsub__(self,o):return -self+o
    def __mul__(self,o):
        o=o if isinstance(o,Dual) else Dual(o);return Dual(self.x*o.x,self.d*o.x+self.x*o.d)
    __rmul__=__mul__
    def __truediv__(self,o):
        o=o if isinstance(o,Dual) else Dual(o);return Dual(self.x/o.x,(self.d*o.x-self.x*o.d)/(o.x*o.x))
    def __rtruediv__(self,o):return Dual(o)/self
    def sqrt(self):
        r=self.x.sqrt();return Dual(r,self.d/(2*r))
    def asin(self):return Dual(self.x.asin(),self.d/(1-self.x*self.x).sqrt())

def ordinary(a,b):
    a=Dual(a);b=Dual(b,1);c=Dual('3/2')-a-b
    aa=1-a*a;bb=1-b*b;cc=1-c*c
    inv=[(b*c-a)/(bb*cc).sqrt(),(a*c-b)/(aa*cc).sqrt(),(a*b-c)/(aa*bb).sqrt()]
    return Dual('3/2')+(sum(v.asin() for v in inv)-a.asin()-b.asin()-c.asin())/(2*arb.pi())

def model(a,lo,hi):
    a=Fraction(a);lo=Fraction(lo);hi=Fraction(hi)
    signs=[]
    for i in range(32):
        left=lo+(hi-lo)*Fraction(i,32);right=lo+(hi-lo)*Fraction(i+1,32)
        d=ordinary(a,ball(left).union(ball(right))).d
        assert d>0 or d<0,('root monotonicity unresolved',a,left,right,str(d))
        signs.append(1 if d>0 else -1)
    assert len(set(signs))==1
    flo=ordinary(a,lo).x-1;fhi=ordinary(a,hi).x-1
    assert (flo<0 and fhi>0) or (flo>0 and fhi<0)
    for _ in range(120):
        mid=(lo+hi)/2;fm=ordinary(a,mid).x-1
        if (fm>0)==(flo>0):lo=mid;flo=fm
        else:hi=mid;fhi=fm
    b=ball(lo).union(ball(hi));av=ball(a);c=ball('3/2')-av-b
    r=[[arb(1),av,b],[av,arb(1),c],[b,c,arb(1)]]
    l22=(1-av*av).sqrt();l32=(c-av*b)/l22;l33=(1-b*b-l32*l32).sqrt()
    assert l22>0 and l33>0
    l=[[arb(1),arb(0),arb(0)],[av,l22,arb(0)],[b,l32,l33]]
    rf=[[float(x.mid()) for x in row] for row in r];lf=[[float(x.mid()) for x in row] for row in l]
    re=sum(up(abs(x-arb(f)))**2 for row,fr in zip(r,rf) for x,f in zip(row,fr))**.5
    le=sum(up(abs(x-arb(f)))**2 for row,fr in zip(l,lf) for x,f in zip(row,fr))**.5
    return dict(first_correlation=str(a),root_interval_fraction=[str(lo),str(hi)],root_monotonic_sign=signs[0],ordinary_boundary_interval=bounds(ordinary(a,b).x),covariance=rf,cholesky=lf,covariance_error_fro_upper=math.nextafter(re,math.inf),cholesky_error_fro_upper=math.nextafter(le,math.inf),positive_cholesky_pivots=[bounds(l22),bounds(l33)])

def quantizer(n):
    assert n%2==0
    x=math.sqrt(3)*ndtri((np.arange(n)+.5)/n)
    for _ in range(3000):
        e=np.r_[-np.inf,(x[:-1]+x[1:])/2,np.inf]
        p=np.where(e[:-1]>=0,ndtr(-e[:-1])-ndtr(-e[1:]),ndtr(e[1:])-ndtr(e[:-1]))
        ph=np.exp(-e*e/2)/math.sqrt(2*math.pi)
        new=(ph[:-1]-ph[1:])/p;new=(new-new[::-1])/2
        if np.max(abs(new-x))<1e-12:x=new;break
        x=new
    e=np.r_[-np.inf,(x[:-1]+x[1:])/2,np.inf]
    positive=e[n//2:-1];e=np.r_[-np.inf,-positive[:0:-1],positive,np.inf]
    assert len(e)==n+1 and e[n//2]==0 and np.all(e[1:]>e[:-1])
    def phi(a):return arb(0) if math.isinf(a) else (-(arb(a)**2)/2).exp()/(2*arb.pi()).sqrt()
    def sf(a):
        if a==math.inf:return arb(0)
        if a==-math.inf:return arb(1)
        return (arb(a)/arb(2).sqrt()).erfc()/2
    ps=[];mus=[]
    for a,b in zip(e[:-1],e[1:]):
        p=sf(a)-sf(b);assert p>0
        ps.append(p);mus.append((phi(a)-phi(b))/p)
    pf=[float(p.mid()) for p in ps];xf=[float(m.mid()) for m in mus]
    variance=1-sum(p*m*m for p,m in zip(ps,mus));assert variance>0
    dp=sum(abs(p-arb(f)) for p,f in zip(ps,pf));dm=max(up(abs(m-arb(f))) for m,f in zip(mus,xf))
    assert max(abs(v) for v in xf)<8 and dp<arb('1e-12') and dm<1e-12
    return dict(bins=n,boundaries_hex=[float(a).hex() for a in e],nodes=xf,probabilities=pf,probability_sum_interval=bounds(sum(ps)),conditional_variance_interval=bounds(variance),conditional_variance_upper=up(variance),node_error_max_upper=dm,probability_l1_error_upper=up(dp),max_node_abs=max(abs(v) for v in xf),tail_cells_included=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bins',type=int,default=128);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    d=dict(arb_precision_bits=ctx.prec,models={'A':model('4/5','64/1000','66/1000'),'B':model('9/10','117/1000','119/1000')},quantizer=quantizer(a.bins))
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(bins=a.bins,product_nodes=a.bins**3,quantization_remainder_upper=3*d['quantizer']['conditional_variance_upper'],root_intervals={k:v['ordinary_boundary_interval'] for k,v in d['models'].items()})),flush=True)

if __name__=='__main__':main()
