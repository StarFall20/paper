"""Verify original score rows, derivatives, recession and finite slopes."""
import argparse
from collections import Counter
from decimal import Decimal, localcontext
from fractions import Fraction as F
from hashlib import sha256
import json
from itertools import product
from math import factorial
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(p):
    return sha256(p.read_bytes()).hexdigest()


def mean(rows, total, value):
    return sum(count * value(row) for *row, count in rows) / F(total)


def covariance(scores, n):
    mu = [sum(count*s[i] for s,count in scores.items()) / n for i in range(2)]
    return tuple(sum(count*(s[i]-mu[i])*(s[j]-mu[j]) for s,count in scores.items())/(n-1)
                 for i,j in product(range(2), repeat=2))


def aux_scores(rows, power, predictor):
    result = Counter()
    for x,z,count in rows:
        residual = power*(F(1,2)-predictor(z))
        result[(residual,x*residual)] += count
    return result


def derivatives_at_zero(order):
    c = [F(1,2)]
    for d in range(order-1):
        c.append((c[d] - sum(c[j]*c[d-j] for j in range(d+1)))/(d+1))
    return {d: c[d-1]*factorial(d-1) for d in range(1,order+1)}


def dec(p):
    return Decimal(p.numerator) / Decimal(p.denominator)


def source_one_dimensional(labelled, aux, power, predictor, m, n):
    """Use each of the three original means to evaluate the slope score and loss."""
    groups = ((labelled, m, Decimal(1), True),
              (labelled, m, -dec(power), False),
              (aux, n, dec(power), False))

    def evaluate(beta, loss=False):
        answer = Decimal(0)
        for rows, size, sign, observed in groups:
            total = Decimal(0)
            for row in rows:
                x,z,*outcome,count = row
                x = dec(x)
                u = beta*x
                y = Decimal(outcome[0]) if observed else dec(predictor(z))
                t = (-abs(u)).exp()
                if loss:
                    value = max(Decimal(0),u)+(1+t).ln()-u*y
                else:
                    prob = 1/(1+t) if u>=0 else t/(1+t)
                    value = x*(prob-y)
                total += count*value
            answer += sign*total/size
        return answer

    return evaluate


def finite_slope(evaluate):
    lo, hi = Decimal(0), Decimal(1)
    assert evaluate(lo)<0
    while evaluate(hi)<=0:
        hi *= 2
        assert hi<Decimal(2)**32
    for _ in range(165):
        mid = (lo+hi)/2
        if evaluate(mid)>0:
            hi = mid
        else:
            lo = mid
    root = (lo+hi)/2
    assert abs(evaluate(root)) < Decimal('1e-38')
    return root


def verify_case(case, plan):
    m,n = (case[field] for field in ('labelled_count','auxiliary_count'))
    power = F(case['power'])
    degree = case['matched_degree']
    assert degree==2*case['K']+1 and degree>2
    offset = F(plan['prediction_offset'])
    predictor = lambda z: F(1,2)+offset*z
    labelled = [(F(x),z,y,count) for x,z,y,count in case['labelled']]
    assert sum(row[-1] for row in labelled)==m and all(row[-1]>0 for row in labelled)
    labelled_counts = {(x,z,y): count for x,z,y,count in labelled}
    assert len(labelled_counts)==len(labelled)
    for (x,z,y),count in labelled_counts.items():
        assert (-x,-z,1-y) in labelled_counts and labelled_counts[-x,-z,1-y]==count
        assert (x,z,1-y) in labelled_counts and (z==0 or x==0)
    brier = mean(labelled,m,lambda row:(predictor(row[1])-row[2])**2)
    assert brier==F(7,32)
    for z in (-1,0,1):
        subgroup = [row for row in labelled if row[1]==z]
        assert mean(subgroup,sum(row[-1] for row in subgroup),lambda row:F(row[2]))==predictor(z)
    labelled_scores = Counter()
    for x,z,y,count in labelled:
        value = (1-power)*F(1,2)-y+power*predictor(z)
        labelled_scores[(value,x*value)] += count
    labelled_cov = covariance(labelled_scores,m)
    label_mean = mean(labelled,m,lambda row:row[0]*row[2])
    lpred = mean(labelled,m,lambda row:row[0]*predictor(row[1]))
    derivative = derivatives_at_zero(degree)
    outputs = {}
    summaries = []
    all_profiles = {(x,z) for x,z,_,_ in labelled}
    for state,raw in case['auxiliary'].items():
        aux = [(F(x),z,count) for x,z,count in raw]
        aux_count = {(x,z):count for x,z,count in aux}
        assert len(aux_count)==len(aux) and sum(aux_count.values())==n
        assert all(count>0 and count==aux_count.get((-x,-z)) for (x,z),count in aux_count.items())
        assert {(x,z) for x,z in aux_count if z}=={(F(0),1),(F(0),-1)}
        all_profiles.update(aux_count)
        aux_pred = mean(aux,n,lambda row:row[0]*predictor(row[1]))
        target = label_mean - power*lpred + power*aux_pred
        assert aux_pred==lpred==0 and target>0
        score = (F(0),-target)
        aux_row_scores = aux_scores(aux,power,predictor)
        cov = covariance(aux_row_scores,n)
        moment = {j:(1-power)*mean(labelled,m,lambda row:row[0]**j)+
                  power*mean(aux,n,lambda row:row[0]**j) for j in range(degree+1)}
        jet = {(a,d-a):derivative[d]*moment[d-a] for d in range(2,degree+1) for a in range(d+1)}
        h = (jet[2,0],jet[0,2])
        assert jet[1,1]==0 and min(h)>0
        sandwich = tuple((labelled_cov[2*i+j]/m + cov[2*i+j]/n)/(h[i]*h[j])
                         for i,j in product(range(2),repeat=2))
        assert sandwich[0]>0 and sandwich[0]*sandwich[3]>sandwich[1]*sandwich[2]
        summaries.append((score,aux_row_scores,jet,sandwich,(0,target/h[1])))

        def recession(direction):
            return (mean(labelled,m,lambda row:max(F(0),direction*row[0])-direction*row[0]*row[2])
                    +power*mean(aux,n,lambda row:max(F(0),direction*row[0])-direction*row[0]*predictor(row[1]))
                    -power*mean(labelled,m,lambda row:max(F(0),direction*row[0])-direction*row[0]*predictor(row[1])))

        rate = recession(1)
        assert recession(-1)>0
        assert state==('finite' if rate>0 else 'unbounded' if rate<0 else 'boundary')
        zero_mass = (1-power)*mean(labelled,m,lambda row:F(row[0]==0))+power*mean(aux,n,lambda row:F(row[0]==0))
        solve = source_one_dimensional(labelled,aux,power,predictor,m,n)
        check = dict(recession_positive=str(rate),recession_negative=str(recession(-1)))
        if state=='finite':
            check['finite_slope'] = format(finite_slope(solve),'.18g')
            check['ordinary_slope'] = format(finite_slope(source_one_dimensional(labelled,aux,F(0),predictor,m,n)),'.18g')
        else:
            nonzero = min(abs(x) for x,z in all_profiles if x)
            t = Decimal(120)/dec(nonzero)
            q = solve(t,loss=True)
            gap = q-t*dec(rate)-dec(zero_mass)*Decimal(2).ln()
            assert abs(gap)<Decimal('1e-45')
            check['finite_intercept_limit_constant'] = format(dec(zero_mass)*Decimal(2).ln(),'.18g')
        outputs[state]=check
    assert all(s==summaries[0] for s in summaries)
    assert F(outputs['finite']['recession_positive'])==-F(outputs['unbounded']['recession_positive'])
    # One symmetric empirical covariate law covers the common labelled and all auxiliary rows.
    assert all(F(0)<predictor(z)<1 and (x==0 or z==0) for x,z in all_profiles)
    for x in {x for x,_ in all_profiles}:
        z_support = {z for x_obs,z in all_profiles if x_obs==x}
        assert sum(predictor(z) for z in z_support)/len(z_support)==F(1,2)
    return dict(family=case['family'], K=case['K'], power=case['power'], degree=degree, m=m, N=n,
                common_reference_score=[str(s) for s in summaries[0][0]],
                common_reference_hessian_diagonal=[str(v) for v in h],
                common_reference_newton_slope=str(summaries[0][-1][1]),
                labelled_brier=str(brier), ordinary_baseline_brier='1/4',
                outcomes=outputs)


def main(designs,out):
    plan_path=ROOT/'plan.json'
    plan=json.loads(plan_path.read_text())
    cases=json.loads(designs.read_text())
    assert cases['plan_sha256']==sha(plan_path)
    assert cases['source_sha256']==plan['source_sha256']
    with localcontext() as ctx:
        ctx.prec=75
        results=[verify_case(case,plan) for case in cases['cases']]
    assert len(results)==30
    selected=plan['compact_case']
    compact=[r for r in results if all(r[k]==v for k,v in selected.items())][0]
    assert compact['m']==compact['N']==32
    assert compact['outcomes']['finite']['recession_positive']=='1/32'
    assert compact['outcomes']['unbounded']['recession_positive']=='-1/32'
    audit=dict(status='passed', design_cases=len(results), fitted_powers=plan['powers'],
               independent_recession_reconstruction=True, exact_scores_hessians_sandwich_jets=True,
               max_order=max(r['degree'] for r in results), compact_case=compact, records=results,
               designs_sha256=sha(designs), verifier_sha256=sha(Path(__file__)))
    out.write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:audit[k] for k in ('status','design_cases','max_order','compact_case')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--designs',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    main(args.designs,args.out)
