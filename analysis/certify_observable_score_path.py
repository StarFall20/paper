"""Exact rational finite-design example; no optimizer or quadrature is used."""
from fractions import Fraction as F
from pathlib import Path
import argparse,json,hashlib


def mean(values):return sum(values,F(0))/len(values)
def covariance(a,b,divisor):
    ma,mb=mean(a),mean(b)
    return sum((x-ma)*(y-mb) for x,y in zip(a,b))/divisor
def serial(x):
    if isinstance(x,F):return str(x)
    if isinstance(x,list):return [serial(t) for t in x]
    if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
    return x


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    x=[F(2),F(4),F(1),F(1)];g=[F(9,10),F(9,10),F(1,2),F(1,2)]
    true=[-x[0]/2,-x[1]/2];pred=[xx*(F(1,2)-gg) for xx,gg in zip(x,g)]
    cross=covariance(true,pred[:2],2);predvar=covariance(pred,pred,3)
    alpha=cross/(2*predvar);assert alpha==F(15,88)
    rows=[];common=None
    for fraction in [F(0),F(1,4),F(1,2),F(3,4),F(1)]:
        power=alpha+fraction*(1-alpha);gs=[F(1,2)+alpha/power*(gg-F(1,2)) for gg in g]
        labelled=[xx*((1-power)/2-1+power*gg) for xx,gg in zip(x[:2],gs[:2])]
        auxiliary=[power*xx*(F(1,2)-gg) for xx,gg in zip(x[2:],gs[2:])]
        score=mean(labelled)+mean(auxiliary)
        middle=covariance(labelled,labelled,1)/2+covariance(auxiliary,auxiliary,1)/2
        transformed_pred=[xx*(F(1,2)-gg) for xx,gg in zip(x,gs)]
        recomputed=covariance(true,transformed_pred[:2],2)/(2*covariance(transformed_pred,transformed_pred,3))
        assert recomputed==power
        weights=[(1-power)/2,(1-power)/2,power/2,power/2]
        moment=mean([xx*(1-power*gg) for xx,gg in zip(x[:2],gs[:2])])+power*mean([xx*gg for xx,gg in zip(x[2:],gs[2:])])
        upper=sum(w*xx for w,xx in zip(weights,x));gap=upper-moment
        if common is None:common=(labelled,auxiliary,score,middle)
        assert common==(labelled,auxiliary,score,middle)
        rows.append(dict(fraction=fraction,power=power,prediction=gs,score=score,
            empirical_score_middle=middle,recomputed_variance_power=recomputed,
            moment=moment,upper_support=upper,positive_recession_cost=gap,
            finite_unique_minimizer=(0<moment<upper)))
    assert rows[0]['positive_recession_cost']==F(3,88) and rows[0]['finite_unique_minimizer']
    assert rows[-1]['positive_recession_cost']==-F(35,44) and not rows[-1]['finite_unique_minimizer']
    cutoff=F(9,44)
    assert alpha<cutoff<1 and common[2]==-F(57,44)
    # All x are positive. A finite scalar logistic derivative spans exactly
    # (0, upper_support). The affine gap is 9/44 - power along this path.
    for row in rows:assert row['positive_recession_cost']==cutoff-row['power']
    result=dict(date='2026-11-12',x=x,labelled_y=[1,1],base_prediction=g,
        observed_anchor=0,anchor_probability=F(1,2),base_variance_power=alpha,
        finite_power_cutoff=cutoff,finite_region='15/88 <= power < 9/44',
        boundary_power='9/44 has no finite stationary point',
        exterior_region='9/44 < power <= 1 is unbounded below',
        all_computed_score_rows_identical=True,all_empirical_score_middles_identical=True,
        all_recomputed_variance_powers_equal_path_powers=True,
        all_predictors_strictly_inside_unit_interval=True,
        true_parameter_or_oracle_prediction_used=False,
        rows=rows,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Deterministic scalar finite-design illustration at an observed reference point. It does not establish common sampling covariance, predictor consistency, inference validity or novelty of general convex support geometry.')
    (a.out/'observable_rational_example.json').write_text(json.dumps(serial(result),indent=2)+'\n')
    print(json.dumps(dict(base_power=str(alpha),cutoff=str(cutoff),base_recession_gap=str(rows[0]['positive_recession_cost']),unit_recession_gap=str(rows[-1]['positive_recession_cost']),all_exact_identities=True)))


if __name__=='__main__':main()
