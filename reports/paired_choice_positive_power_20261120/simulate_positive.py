"""Paired finite-sample check with nonzero weight on both pools' log partitions."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csc_matrix, eye, hstack, vstack
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"paired_choice_boundary_20261120"))
from simulate import design_pair, minimum_box_radius


def probability_margin(weighted, target):
    """Independent primal margin formulation in the original probabilities."""
    rows, dimension = weighted.shape
    unit = csc_matrix(np.ones((rows,1)))
    ident = eye(rows,format="csc")
    inequalities = vstack([hstack([-ident,unit]), hstack([ident,unit])],format="csc")
    equals = hstack([csc_matrix(weighted.T), csc_matrix((dimension,1))],format="csc")
    moment = 0.5*(weighted.T @ np.ones(rows)+target)
    result = linprog(np.r_[np.zeros(rows),-1.],
                     A_ub=inequalities,b_ub=np.r_[np.zeros(rows),np.ones(rows)],
                     A_eq=equals,b_eq=moment,bounds=[(None,None)]*rows+[(None,0.5)],method="highs")
    assert result.success
    rechecked = 1-2*result.x[-1]
    assert np.max(np.abs(weighted.T@result.x[:-1]-moment)) < 1e-6
    return float(rechecked)


def main(out):
    plan = json.loads((HERE/"plan.json").read_text())["paired_simulation"]
    boundary = json.loads((HERE/"boundaries.json").read_text())
    power = plan["primary_power"]
    relevant = [r["critical_slopes_per_labelled_person"] for r in boundary
                if r["power"]==power and r["correlation"] in plan["correlations"]]
    assert len(relevant)==2
    ratio = round(sum(relevant)/2.,3)
    out.mkdir(parents=True, exist_ok=True)
    lock = dict(power=power, exact_calculated_boundaries=relevant, nominal_ratio=ratio,
                source_plan=plan, recorded_before_simulation=True)
    (out/"simulation_design.json").write_text(json.dumps(lock,indent=2)+"\n")
    rows=[]
    for m in plan["labelled_and_auxiliary_respondents"]:
        slopes=round(m*ratio)
        for allocation in range(plan["coupled_draws_per_sample_size"]):
            for rho in plan["correlations"]:
                human, label, auxiliary = design_pair(plan["seed"],plan["family"],m,allocation,slopes,rho)
                weighted = np.vstack(((1-power)*human,power*auxiliary))
                target = human.T@label
                state,radius,residual,recession,gap=minimum_box_radius(weighted,target)
                ordinary, ordinary_radius, *_=minimum_box_radius(human,target)
                secondary = ""
                if allocation<8:
                    secondary = abs(probability_margin(weighted,target)-radius)
                    assert secondary < 1e-7
                rows.append(dict(respondents=m,slopes=slopes,correlation=rho,allocation=allocation,
                                 state=state,radius=radius,primal_error=residual,recession=recession,
                                 duality_gap=gap,ordinary_state=ordinary,ordinary_radius=ordinary_radius,
                                 independent_radius_difference=secondary))
        print(f"Finished {m} people per sample",flush=True)
    with (out/"cohort_results.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    paired=[]
    for m in plan["labelled_and_auxiliary_respondents"]:
        subsets=[[r for r in rows if r["respondents"]==m and r["correlation"]==rho] for rho in plan["correlations"]]
        a,b=subsets
        assert all(r["allocation"]==s["allocation"] for r,s in zip(a,b,strict=True))
        paired.append(dict(
            respondents=m,slopes=a[0]["slopes"],draws=len(a),
            independent_finite=sum(r["state"]=="interior" for r in a),
            correlated_finite=sum(r["state"]=="interior" for r in b),
            finite_to_exterior=sum(r["state"]=="interior" and s["state"]=="exterior" for r,s in zip(a,b,strict=True)),
            exterior_to_finite=sum(r["state"]=="exterior" and s["state"]=="interior" for r,s in zip(a,b,strict=True)),
            ordinary_finite=sum(r["ordinary_state"]=="interior" for subset in subsets for r in subset),
            near_boundary=sum(r["state"]=="near_boundary" for subset in subsets for r in subset),
        ))
    report=dict(records=len(rows),paired=paired,checks=dict(
        max_primal_residual=max(r["primal_error"] for r in rows),
        max_duality_gap=max(r["duality_gap"] for r in rows),
        independently_rechecked=sum(r["independent_radius_difference"]!="" for r in rows),
        max_independent_radius_diff=max(r["independent_radius_difference"] for r in rows if r["independent_radius_difference"]!=""),
    ))
    (out/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    with threadpool_limits(limits=1):
        main(parser.parse_args().out)
