"""Bound the .90 population contrast without the polygon integration."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import ndtr


def tail_pair(grid):
    cdf = ndtr(-grid)
    density = np.exp(-grid*grid/2)/np.sqrt(2*np.pi)
    return (1+grid*grid)*cdf-grid*density, -2*(density-grid*cdf)


def rectangle(u, lam, rho, tau):
    """The transformed parallelograms lie inside these rectangles."""
    widths = np.sqrt(2 * np.array([1+rho,1-rho]))
    label_width = (1-lam)*widths
    centers = np.diag(widths)
    a = np.stack((centers+label_width,-centers+label_width),axis=-1)
    auxiliary = lam/tau*widths
    label_parts, label_derivative = tail_pair(u*a)
    aux_parts, aux_derivative = tail_pair(u*auxiliary)
    score = 2-label_parts.sum()/2+tau*(2-2*aux_parts.sum())
    slope = -(label_derivative*a).sum()/2-2*tau*(aux_derivative*auxiliary).sum()
    return float(score), float(slope)


def ordinary_pair_score(u, lam, tau):
    labelled, _ = tail_pair(np.array([-u*lam, u*(2-lam)]))
    auxiliary, _ = tail_pair(u*lam/tau)
    return float(2*(1-labelled.sum())+2*tau*(1-2*auxiliary))


def main(out):
    chosen = 0.9
    tau = 1.
    u = 0.405
    lower_scale = 0.48
    lower = ordinary_pair_score(lower_scale,chosen,tau)
    value, derivative = rectangle(u,chosen,.99,tau)
    curvature = 2*chosen**2
    upper = value+derivative**2/(2*curvature)
    assert upper < .663 < lower
    result=dict(
        state="passed", power=chosen, auxiliary_ratio=tau, upper_evaluation_scale=u,
        lower_evaluation_scale=lower_scale,
        independent_boundary_lower_bound=lower, correlated_boundary_upper_bound=upper,
        upper_objective_value=value, upper_objective_slope=derivative,
        minimum_upper_objective_concavity=curvature,
        numerical_rounding_for_text=dict(correlated_upper=.635, independent_lower=.750),
    )
    assert upper<result["numerical_rounding_for_text"]["correlated_upper"]
    assert lower>result["numerical_rounding_for_text"]["independent_lower"]
    out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    main(parser.parse_args().out)
