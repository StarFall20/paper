"""Verify raw score signatures, attainment signs and exact approximation bounds."""
import argparse
from decimal import Decimal, localcontext
from fractions import Fraction as F
from hashlib import sha256
from math import lcm
from pathlib import Path
import json


ROOT=Path(__file__).resolve().parent


def sha(p):
    return sha256(p.read_bytes()).hexdigest()


def symmetric(rows):
    result=[]
    for value,weight in rows:
        x,w=F(value),F(weight)
        result.extend([(x,w/2),(-x,w/2)] if x else [(x,w)])
    assert sum(w for _,w in result)==1 and all(w>0 for _,w in result)
    return result


def human_rows(magnitudes, tilt):
    rows=[]
    for x,p in magnitudes:
        sign=(x>0)-(x<0)
        py=F(1,2)+tilt*sign/2
        assert 0<py<1
        rows.extend([(x,0,p*(1-py)),(x,1,p*py)])
    return rows


def covariance(rows):
    mass_count=lcm(*(p.denominator for p,_ in rows))
    mean=[sum(p*v[i] for p,v in rows) for i in (0,1)]
    out=[[sum(p*(v[i]-mean[i])*(v[j]-mean[j]) for p,v in rows)*mass_count/(mass_count-1)
           for j in (0,1)] for i in (0,1)]
    return out,mass_count


def check_signature(labelled, auxiliary, lam, scaled):
    covariate_shift=lambda x: (F(1,16)/lam if scaled else F(1,4))*x
    labelled_scores=[(mass,(F(1,2)-y+lam*covariate_shift(x),x*(F(1,2)-y+lam*covariate_shift(x))))
                      for x,y,mass in labelled]
    auxiliary_scores=[(mass,(-lam*covariate_shift(x),-lam*x*covariate_shift(x))) for x,mass in auxiliary]
    assert all(0<F(1,2)+covariate_shift(x)<1 for x,_,_ in labelled)
    assert all(0<F(1,2)+covariate_shift(x)<1 for x,_ in auxiliary)
    derivative=[sum(p*s[i] for p,s in labelled_scores+auxiliary_scores) for i in (0,1)]
    return derivative,labelled_scores,auxiliary_scores


def find_score_root(labelled,auxiliary,weight,target):
    def dec(x):
        return Decimal(x.numerator)/Decimal(x.denominator)
    with localcontext() as ctx:
        ctx.prec=110
        masses=[(dec(abs(x)),dec(p*(1-weight))) for x,p in labelled]
        masses.extend((dec(abs(x)),dec(p*weight)) for x,p in auxiliary)
        b=dec(target)
        def derivative(beta):
            return sum(m*x*(Decimal(1)/(Decimal(1)+(-x*beta).exp())-Decimal('0.5'))
                       for x,m in masses)-b
        assert derivative(Decimal(0)) < 0
        hi=Decimal(1)
        while derivative(hi)<=0:
            hi*=2
            assert hi<Decimal(10_000)
        lo=Decimal(0)
        for _ in range(240):
            middle=(hi+lo)/2
            if derivative(middle)>0:
                hi=middle
            else:
                lo=middle
        assert derivative(lo)<0 and derivative(hi)>0
        return format((lo+hi)/2,'.17g')


def check_design(case):
    k=case['K']
    degree=case['matched_total_degree']
    assert degree==2*k+1
    l=symmetric(case['labelled_magnitude_mass'])
    u=symmetric(case['auxiliary_magnitude_mass'])
    moments=[[sum(p*x**j for x,p in group) for j in range(degree+1)] for group in [l,u]]
    assert moments[0]==moments[1] and moments[0][2]>0
    abs_l,abs_u=(sum(abs(x)*p for x,p in group) for group in [l,u])
    gap=abs_l-abs_u
    assert gap==F(case['absolute_moment_gap']) and gap>0
    labelled=human_rows(l,F(case['labelled_outcome_tilt']))
    human_q=[sum(p*y*x**j for x,y,p in labelled) for j in (0,1)]
    assert human_q[0]==F(1,2) and human_q[1]==F(case['slope_moment'])
    fixed_signatures=[]
    scaled_signatures=[]
    margins={}
    for lam,state in [(F(1,4),'attained'),(F(1,2),'boundary'),(F(3,4),'unbounded')]:
        recession=(1-lam)*sum(p*max(x,0) for x,p in l)+lam*sum(p*max(x,0) for x,p in u)-human_q[1]
        assert recession==F(case['exact_slope_recession'][str(lam)])
        assert (recession>0 if state=='attained' else recession==0 if state=='boundary' else recession<0)
        margins[str(lam)]=str(recession)
        fixed_signatures.append(check_signature(labelled,u,lam,False)[0])
        scaled_signatures.append(check_signature(labelled,u,lam,True))
    assert fixed_signatures==[[0,-human_q[1]]]*3
    assert scaled_signatures==[scaled_signatures[0]]*3
    derivative,l_scores,u_scores=scaled_signatures[0]
    assert derivative==[0,-human_q[1]]
    hess=[[F(1,4),F(0)],[F(0),moments[0][2]/4]]
    cov_l,n_l=covariance(l_scores)
    cov_u,n_u=covariance(u_scores)
    meat=[[cov_l[i][j]/n_l+cov_u[i][j]/n_u for j in (0,1)] for i in (0,1)]
    assert meat[0][0]>0 and meat[0][0]*meat[1][1]-meat[0][1]**2>0
    sandwich=[[meat[i][j]/(hess[i][i]*hess[j][j]) for j in (0,1)] for i in (0,1)]
    # Conditional intercept zero, strict slope increase, positive and negative slope-ray limits.
    ordinary=(abs_l/2)-human_q[1]
    assert ordinary==gap/4 and ordinary>0
    return dict(K=k,degree=degree,one_quarter_finite_slope=find_score_root(l,u,F(1,4),human_q[1]),
                ordinary_human_has_finite_optimum=True,first_common_derivative=[str(z) for z in derivative],
                common_hessian=[[str(z) for z in row] for row in hess],
                common_sandwich_diagonal=[str(sandwich[i][i]) for i in (0,1)],
                equal_weight_sample_sizes_by_replication=[n_l,n_u],
                absolute_moment_gap=str(gap),recession=margins)


def check_continuous_bound(case):
    k=case['K']
    coeffs=[F(z) for z in case['approx_even_chebyshev_coefficients']]
    assert len(coeffs)==k+1
    intervals=case['grid_n']
    e=F(0)
    for i in range(intervals+1):
        x=F(i,intervals)
        t0,t1=F(1),x
        value=coeffs[0]
        for j in range(2,2*k+1):
            t0,t1=t1,2*x*t1-t0
            if j%2==0:
                value+=coeffs[j//2]*t1
        e=max(e,abs(value-x))
    m2=sum(abs(a)*(((2*j)**4-(2*j)**2)//3) for j,a in enumerate(coeffs))
    upper=e+m2/(8*intervals**2)
    lower=F(case['absolute_moment_gap'])/2
    assert 0<lower<=e<=upper and upper-lower < F(1,500)
    return dict(K=k,degree=2*k,lower_exact=str(lower),upper_exact=str(upper),
                lower=float(lower),upper=float(upper),grid_error=float(e),
                upper_interpolation_allowance=float(upper-e),lower_to_upper_ratio=float(lower/upper))


def verify(args):
    package=json.loads(args.designs.read_text())
    plan=json.loads((ROOT/'plan.json').read_text())
    assert package['plan_sha256']==sha(ROOT/'plan.json')
    assert package['script_sha256']==sha(ROOT/'calculate.py')
    analytic=[check_design(r) for r in package['exact_constructions']]
    approx=[check_design(r) for r in package['near_extremal_grid_constructions']]
    bounds=[check_continuous_bound(r) for r in package['near_extremal_grid_constructions']]
    assert [r['K'] for r in analytic]==plan['exact_orders_K']
    assert [r['K'] for r in approx]==plan['approximation_orders_K']
    receipt=dict(state='passed',finite_designs_checked=len(analytic)+len(approx),
                 exact_orders=analytic,near_extremal_orders=approx,
                 continuous_best_approximation_intervals=bounds,
                 designs_sha256=sha(args.designs),verifier_sha256=sha(Path(__file__)))
    args.out.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(verified_designs=receipt['finite_designs_checked'],brackets=bounds)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--designs',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    verify(parser.parse_args())
