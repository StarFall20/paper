"""A sensitivity check using the same fixed menu forecast on both samples."""

from argparse import ArgumentParser, Namespace
from collections import Counter
from hashlib import sha256
import importlib.util
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

def fixed_weight_tools(archive):
    path = archive / "results/public_common_menu_weight_limits/verify.py"
    spec = importlib.util.spec_from_file_location("public_menu_weights", path)
    implementation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(implementation)
    return implementation


def check(archive, food, out):
    tools = fixed_weight_tools(archive)
    powers = (*tools.PROBES, 1.0)
    earlier = archive / "results/public_common_menu_weight_limits/results.json"
    old_rows = {(r["trial"], r["label_people"], r["draw"]): r
                for r in tools.read(earlier)["records"]}
    prior = archive / "results/fixed_label_auxiliary_design"
    fixed = tools.load(prior / "run.py", "same_predictor_fixed_label")
    previous = archive / "results/public_common_menu_diagnostics"
    signature = tools.load(previous / "verify.py", "same_predictor_menu_keys").signatures
    plan = tools.read(prior / "plan.json")
    food_path = archive / "results/power_frontier"
    files = dict(food_model=food_path / "analysis/existing_panel_model.py",
                 food_plan=food_path / "inputs/model_plan.json",
                 food_prediction=food_path / "inputs/preflight.json")
    assert all(tools.digest(path) == plan["source_files_sha256"][key] for key,path in files.items())
    result = []
    for _, trial, order, x, g, y, offered, people in fixed.food(Namespace(food_data=food), files, plan["label_allocations"]):
        assert offered.all()
        pool = np.isin(people, order)
        keys = [signature(xx, oo) for xx,oo in zip(x,offered)]
        table = {key:i for i,key in enumerate(sorted({keys[j] for j in np.flatnonzero(pool)}))}
        design = np.array([table.get(key,-1) for key in keys])
        pools = [pool & (design==i) for i in range(len(table))]
        representatives = [np.flatnonzero(p)[0] for p in pools]
        x_menu, forecasts = x[representatives], np.array([g[p].mean(axis=0) for p in pools])
        assert len(table)==34 and np.array_equal(x[pool], x_menu[design[pool]])
        for budget in plan["label_budgets"]:
            label = np.isin(people, order[:budget])
            x_l, g_l = x[label], forecasts[design[label]]
            b_l = x_l[np.arange(len(x_l)), y[label]].mean(axis=0)
            model_l = np.einsum("ta,tad->d", g_l, x_l)/len(x_l)
            original_model_l = np.einsum("ta,tad->d", g[label], x_l)/len(x_l)
            ordinary = fixed.support(x_l, offered[label], np.full(label.sum(),1/label.sum()), b_l)[0]
            rng=np.random.default_rng(plan["auxiliary_generator_seed"]+10007*trial+101*budget)
            records=[]
            for draw in range(plan["auxiliary_draws"]):
                auxiliary=np.isin(people,rng.choice(order[budget:],size=plan["auxiliary_people"],replace=False))
                ids=design[auxiliary]
                counts=np.bincount(ids,minlength=len(table))
                assert counts.sum()==384 and np.all(counts>0)
                w=counts/counts.sum()
                delta=np.einsum("t,ta,tad->d",w,forecasts,x_menu)-model_l
                x_u=x[auxiliary]
                cap, direction=tools.cap_weights(x_l,x_menu,w,b_l,delta)
                cap_raw,_=tools.cap_weights(x_l,x_u,np.full(len(x_u),1/len(x_u)),delta=delta,b_l=b_l)
                assert abs(cap-cap_raw)<1e-8
                if cap<1-1e-8:
                    a,b=tools.original_recession(x_l,x_u,b_l,delta,direction)
                    assert b<0 and abs(a/(a-b)-cap)<1e-8
                rows = (x_l, x_u, b_l, delta)
                old = old_rows[(trial, budget, draw)]
                assert old["ordinary"] == ordinary
                records.append(dict(draw=draw, weight_limit=cap,
                                    earlier_weight_limit=old["cap"], rows=rows))
            def count_outside(key, power):
                return sum(row[key] + 1e-8 < power for row in records)

            mixed = any(0 < count_outside(key, power) < len(records)
                        for power in powers for key in ("weight_limit", "earlier_weight_limit"))
            if ordinary == "interior" and mixed:
                for row in records:
                    l,u,base,change=row["rows"]
                    z=np.concatenate((l,u))
                    statuses={}
                    for power in powers:
                        if power == 1:
                            features, weights = u, np.full(len(u), 1 / len(u))
                        else:
                            features = z
                            weights = np.r_[np.full(len(l), (1-power)/len(l)), np.full(len(u), power/len(u))]
                        original_change = change + model_l - original_model_l
                        statuses[str(power)] = {}
                        for model, cap_key, correction in (
                            ("shared_forecast", "weight_limit", change),
                            ("auxiliary_only_shared", "earlier_weight_limit", original_change),
                        ):
                            state = fixed.support(features, np.ones(features.shape[:2], bool),
                                                  weights, base + power * correction)[0]
                            assert state == ("outside" if power > row[cap_key] + 1e-8 else "interior")
                            statuses[str(power)][model] = state
                    row["verified_probe_states"]=statuses
            for row in records:
                row.pop("rows")
            result.append(dict(trial=trial,budget=budget,ordinary=ordinary,records=records))
    groups = [r for r in result if r["records"][0].get("verified_probe_states")]
    summary = {model: Counter({str(q): 0 for q in powers})
               for model in ("shared_forecast", "auxiliary_only_shared")}
    for group in groups:
        for a,b in combinations(group["records"],2):
            for power in powers:
                key=str(power)
                for model in summary:
                    summary[model][key] += (a["verified_probe_states"][key][model]
                                            != b["verified_probe_states"][key][model])
    payload=dict(status="passed", workbook_sha256=sha256(food.read_bytes()).hexdigest(),
                 label_groups=len(result), total_allocations=384,
                 original_weight_audit_sha256=tools.digest(earlier),
                 ordinary_finite_groups=sum(g["ordinary"]=="interior" for g in result),
                 verified_ordinary_finite_mixed_groups=len(groups),
                 direct_probability_state_checks=len(groups)*8*len(powers)*len(summary),
                 changed_state_pairs=summary, groups=result)
    out.write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps({k:v for k,v in payload.items() if k!="groups"},indent=2))


if __name__ == "__main__":
    p=ArgumentParser()
    for name in ("archive","food","out"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    with threadpool_limits(limits=1):
        check(a.archive,a.food,a.out)
