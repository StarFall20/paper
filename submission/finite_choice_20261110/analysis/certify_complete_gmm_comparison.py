"""Exact comparison of two existing objectives on the retained scalar design.

The rational witness and two tail limits prove global attainment of the
specified complete GMM criterion. They do not certify uniqueness, optimal
inference, a production CC-PPI fit, or a new estimator.
"""
from fractions import Fraction as F
from pathlib import Path
import argparse,hashlib,json

def mean(a):return sum(a,F(0))/len(a)
def cov(a,b):
    ma,mb=mean(a),mean(b)
    return mean([(x-ma)*(y-mb) for x,y in zip(a,b)])
def encode(a):
    if isinstance(a,F):return str(a)
    if isinstance(a,list):return [encode(x) for x in a]
    if isinstance(a,dict):return {k:encode(v) for k,v in a.items()}
    return a

def calculation():
    xl=[F(2),F(4)];xu=[F(1),F(1)];y=[F(1),F(1)]
    base=F(15,88);gl=[F(1,2)+base*F(2,5)]*2;gu=[F(1,2)]*2
    sl=[x*(F(1,2)-yy) for x,yy in zip(xl,y)]
    hl=[x*(F(1,2)-g) for x,g in zip(xl,gl)]
    hu=[x*(F(1,2)-g) for x,g in zip(xu,gu)]
    ridge=F(1,len(xl)**2);rho=F(len(xl),len(xu))
    s11=cov(sl,sl)+ridge;s12=-cov(sl,hl)
    s22=cov(hl,hl)+rho*cov(hu,hu)+ridge
    determinant=s11*s22-s12*s12
    assert s11>0 and determinant>0
    w11=s22/determinant;w12=-s12/determinant;w22=s11/determinant
    assert [s11,s12,s22]==[F(1,2),-F(3,88),F(493,1936)]
    assert [w11,w12,w22]==[F(1972,977),F(264,977),F(3872,977)]
    def criterion(m1,m2):return (w11*m1*m1+2*w12*m1*m2+w22*m2*m2)/2
    t=F(3,2)
    probabilities=[t**int(x)/(1+t**int(x)) for x in xl]
    pu=[t/(1+t)]*2
    m1=mean([x*(p-yy) for x,p,yy in zip(xl,probabilities,y)])
    m2=mean([x*(p-g) for x,p,g in zip(xu,pu,gu)])-mean([x*(p-g) for x,p,g in zip(xl,probabilities,gl)])
    witness=criterion(m1,m2)
    plus_m1=F(0);plus_m2=mean([x*(1-g) for x,g in zip(xu,gu)])-mean([x*(1-g) for x,g in zip(xl,gl)])
    minus_m1=-mean(xl);minus_m2=-mean([x*g for x,g in zip(xu,gu)])+mean([x*g for x,g in zip(xl,gl)])
    plus=criterion(plus_m1,plus_m2);minus=criterion(minus_m1,minus_m2)
    assert witness==F(43618624129,38838705425)
    assert plus==F(1225,977) and minus==F(10729,977)
    assert witness<min(plus,minus)
    moment=mean([x*(yy-g) for x,yy,g in zip(xl,y,gl)])+mean([x*g for x,g in zip(xu,gu)])
    recession=mean(xu)-moment
    assert recession==-F(35,44)
    # The supervised score is -1/(1+t^2)-2/(1+t^4)<0 for every t>0.
    # A finite point below both tail limits makes a nonempty GMM sublevel
    # set compact. Continuity then proves finite global attainment.
    return dict(date='2026-11-12',labelled_x=xl,auxiliary_x=xu,labelled_y=y,
        labelled_prediction=gl,auxiliary_prediction=gu,observed_reference=0,
        centered_covariance_divisors='sample sizes m and N',covariance_ridge=ridge,
        Sigma=dict(entry11=s11,entry12=s12,entry22=s22,determinant=determinant),
        W=dict(entry11=w11,entry12=w12,entry22=w22),
        witness=dict(exp_coefficient=t,moment1=m1,moment2=m2,criterion=witness),
        positive_infinite_coefficient=dict(moment1=plus_m1,moment2=plus_m2,criterion=plus),
        negative_infinite_coefficient=dict(moment1=minus_m1,moment2=minus_m2,criterion=minus),
        strict_gap_to_both_tails=min(plus,minus)-witness,
        complete_criterion_has_finite_global_minimizer=True,uniqueness_claimed=False,
        supervised_MLE_has_finite_minimizer=False,unit_corrected_recession_cost=recession,
        unit_corrected_loss_unbounded_below=True,
        exact_positive_definiteness_and_tail_comparison=True,
        production_CC_PPI_algorithm_reproduced=False,
        reason='The reference is fixed at zero, not a finite supervised MLE. The weight is the specified uncalibrated centered covariance inverse with ridge. No spectral calibration is applied.',
        scope='Illustration for existing complete GMM; no dominance, sampling, or novelty claim.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);result=calculation()
    (a.out/'complete_gmm_exact_comparison.json').write_text(json.dumps(encode(result),indent=2)+'\n')
    print(json.dumps(dict(exact_finite_GMM_witness=True,supervised_MLE_separated=True,
        unit_corrected_loss_unbounded_below=True,new_estimator_claim=False)))

if __name__=='__main__':main()
