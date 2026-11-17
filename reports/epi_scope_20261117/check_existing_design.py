"""Exact source-scope comparison on the scalar design already in Appendix B5.

This is an analytic obstruction check, not an EPI implementation or a new
estimator. The positive-weight separation implication is classical.
"""
from fractions import Fraction as F
from pathlib import Path
import argparse, hashlib, json

def mean(values):return sum(values,F(0))/len(values)
def encode(obj):
    if isinstance(obj,F):return str(obj)
    if isinstance(obj,dict):return {k:encode(v)for k,v in obj.items()}
    if isinstance(obj,list):return [encode(v)for v in obj]
    return obj

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    xl=[F(2),F(4)];xu=[F(1),F(1)];y=[F(1),F(1)]
    power=F(15,88);gl=[F(9,10)]*2;gu=[F(1,2)]*2
    moment=mean([x*(yy-power*g)for x,yy,g in zip(xl,y,gl)])+power*mean([x*g for x,g in zip(xu,gu)])
    upper=(1-power)*mean(xl)+power*mean(xu)
    assert moment==F(21,8)and upper==F(117,44)and upper-moment==F(3,88)
    def corrected_score(t,lam,pred_l):
        sigmoid=lambda x:t**int(x)/(1+t**int(x))
        return (1-lam)*mean([x*sigmoid(x)for x in xl])+lam*mean([x*sigmoid(x)for x in xu])-mean([x*(yy-lam*g)for x,yy,g in zip(xl,y,pred_l)])-lam*mean([x*g for x,g in zip(xu,gu)])
    at_zero=corrected_score(F(1),power,gl);at_log8=corrected_score(F(8),power,gl)
    assert at_zero==-F(57,44)and at_log8>0
    # Positive curvature and these exact signs imply a unique finite root.
    endpoint_gl=[F(25,44)]*2
    endpoint_b=mean([x*(yy-g)for x,yy,g in zip(xl,y,endpoint_gl)])+mean([x*g for x,g in zip(xu,gu)])
    endpoint_recession=mean(xu)-endpoint_b
    assert endpoint_recession==-F(35,44)
    assert corrected_score(F(1),F(1),endpoint_gl)==at_zero
    # For t=exp(beta)>0, supervised row scores are
    # -2/(1+t^2), -4/(1+t^4). Every normalized nonnegative weighting
    # has a strictly negative score, independently of all auxiliary terms.
    epi=dict(strict_separator=1,signed_margins=[2,4],supervised_row_score_expressions=['-2/(1+t^2)','-4/(1+t^4)'],parameterization='t=exp(beta)>0',all_finite_coefficients_excluded=True,all_normalized_nonnegative_weights_excluded=True,auxiliary_constraints_can_remove_obstruction=False,upstream_algorithm_executed=False,positive_weight_generalization_is_classical=True)
    result=dict(date='2026-11-17',state='passed',existing_design='Current Appendix B5 and B6',labelled_x=xl,auxiliary_x=xu,labelled_y=y,base_predictions=gl,auxiliary_predictions=gu,base_power=power,base_corrected_moment=moment,base_upper_support=upper,base_support_gap=upper-moment,base_gradient_zero=at_zero,base_gradient_log8=at_log8,base_root_log_bracket=['0','log(8)'],base_unique_finite_root=True,unit_transformed_predictions=endpoint_gl,unit_corrected_moment=endpoint_b,unit_corrected_recession=endpoint_recession,unit_corrected_unbounded_below=True,reference_gradients_equal=True,epi_obstruction=epi,new_independent_cohorts=0,new_human_or_AI_answers=0,new_theoretical_novelty_claim=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.out/'exact_existing_design.json').write_text(json.dumps(encode(result),indent=2)+'\n')
    print(json.dumps(dict(state='passed',exact_support_gap=str(upper-moment),exact_negative_recession=str(endpoint_recession),EPI_constraints_obstructed_by_supervised_score=True,new_method_or_novelty_claim=False)))

if __name__=='__main__':main()
