"""Validate the full correction-weight interval on the fixed RADIANT menus."""

import argparse
from collections import Counter
import csv
from hashlib import sha256
import importlib.util
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
PROBES = (0.75, 0.90, 0.95, 0.99)
TOL = 1e-8


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cap_weights(x_l, x_u, frequencies, b_l, delta):
    """LP in nonnegative masses and the scalar weight, with all alternatives available."""
    x = np.concatenate((x_l, x_u))
    menus, choices, dim = x.shape
    features = x.reshape(-1, dim)
    menu_weights = np.r_[np.full(len(x_l), 1 / len(x_l)), np.zeros(len(x_u))]
    weight_rates = np.r_[-menu_weights[:len(x_l)], frequencies]
    incidence = sparse.csr_matrix((np.ones(menus*choices),
                                  (np.repeat(np.arange(menus), choices), np.arange(menus*choices))))
    equality = sparse.bmat([
        [incidence, sparse.csr_matrix(-weight_rates[:, None])],
        [sparse.csr_matrix(features.T), sparse.csr_matrix(-delta[:, None])],
    ]).tocsr()
    rhs = np.r_[menu_weights, b_l]
    fit = linprog(np.r_[np.zeros(menus*choices), -1.0], A_eq=equality, b_eq=rhs,
                  bounds=[(0, None)]*(menus*choices) + [(0., 1.)], method="highs-ds")
    assert fit.success, fit.message
    assert np.max(np.abs(equality @ fit.x - rhs)) < TOL
    return float(fit.x[-1]), fit.eqlin.marginals[-dim:]


def original_recession(x_l, x_u, b_l, delta, direction):
    label_support = (x_l @ direction).max(axis=1).mean()
    aux_support = (x_u @ direction).max(axis=1).mean()
    return float(label_support - b_l @ direction), float(aux_support - (b_l + delta) @ direction)


def compute(archive, food, out):
    original = archive / "results/fixed_label_auxiliary_design"
    earlier = archive / "results/public_common_menu_diagnostics"
    fit_module = load(original / "run.py", "previous_fixed_labels")
    plan = read(original / "plan.json")
    menu_keys = load(earlier / "verify.py", "previous_menu_keys").signatures
    prior = read(earlier / "results.json")["common_forecast_by_exact_auxiliary_menu"]
    previous = {float(power): {
        (r["trial"], r["label_people"], r["draw"]): r["state"] for r in state["allocation_records"]}
        for power, state in prior.items()}
    original_ordinary = {}
    with (original / "results/allocations.csv").open() as f:
        for row in csv.DictReader(f):
            if row["dataset"] == "RADIANT":
                original_ordinary[(int(row["allocation"]), int(row["label_people"]))] = row["labelled_only"]
    food_path = archive / "results/power_frontier"
    paths = dict(food_model=food_path / "analysis/existing_panel_model.py",
                 food_plan=food_path / "inputs/model_plan.json",
                 food_prediction=food_path / "inputs/preflight.json")
    for name, path in paths.items():
        assert digest(path) == plan["source_files_sha256"][name]
    from argparse import Namespace
    records, groups = [], []
    source = fit_module.food(Namespace(food_data=food), paths, plan["label_allocations"])
    for _, trial, order, x, g, y, offered, people in source:
        assert offered.all()
        reference_pool = np.isin(people, order)
        labels = [menu_keys(xx, available) for xx, available in zip(x, offered)]
        levels = {name: i for i, name in enumerate(sorted(set(labels[i] for i in np.flatnonzero(reference_pool))))}
        ids = np.array([levels.get(label, -1) for label in labels])
        first = [np.flatnonzero(reference_pool & (ids == i))[0] for i in range(len(levels))]
        x_menu, g_menu = x[first], np.array([g[reference_pool & (ids == i)].mean(axis=0) for i in range(len(levels))])
        assert len(levels) == 34 and np.array_equal(x[reference_pool], x_menu[ids[reference_pool]])
        for budget in plan["label_budgets"]:
            label = np.isin(people, order[:budget])
            x_l = x[label]
            b_l = x_l[np.arange(len(x_l)), y[label]].mean(axis=0)
            model_l = np.einsum("tj,tjd->d", g[label], x_l) / len(x_l)
            ordinary = fit_module.support(x_l, offered[label], np.full(label.sum(), 1/label.sum()), b_l)[0]
            assert ordinary == original_ordinary[(trial, budget)]
            rng = np.random.default_rng(plan["auxiliary_generator_seed"] + 10007*trial + 101*budget)
            local = []
            for draw in range(plan["auxiliary_draws"]):
                people_u = rng.choice(order[budget:], size=plan["auxiliary_people"], replace=False)
                aux = np.isin(people, people_u)
                counts = np.bincount(ids[aux], minlength=len(levels))
                assert (counts > 0).all() and len(counts) == 34 and counts.sum() == 384
                f = counts/counts.sum()
                x_u = x[aux]
                g_u = g_menu[ids[aux]]
                delta = np.einsum("t,tj,tjd->d", f, g_menu, x_menu) - model_l
                delta_raw = np.einsum("tj,tjd->d", g_u, x_u)/len(x_u) - model_l
                assert np.max(abs(delta - delta_raw)) < TOL
                cap, ray = cap_weights(x_l, x_menu, f, b_l, delta)
                cap_raw, _ = cap_weights(x_l, x_u, np.full(len(x_u), 1/len(x_u)), b_l, delta_raw)
                assert abs(cap-cap_raw) < TOL
                k = (trial, budget, draw)
                assert (cap < 1 - TOL) == (previous[1.0][k] == "outside")
                assert (cap < .5 - TOL) == (previous[.5][k] == "outside")
                rec = None
                if cap < 1 - TOL:
                    r0, r1 = original_recession(x_l, x_u, b_l, delta_raw, ray)
                    assert r0 >= -TOL and r1 < 0
                    assert abs(r0/(r0-r1) - cap) < TOL
                    rec = [r0, r1]
                results = {}
                if ordinary == "interior" and any(
                    previous[1.0][(trial,budget,j)] != previous[1.0][k] for j in range(plan["auxiliary_draws"])
                ):
                    for power in PROBES:
                        feat = np.concatenate((x_l, x_u))
                        weights = np.r_[np.full(len(x_l), (1-power)/len(x_l)), np.full(len(x_u), power/len(x_u))]
                        observed = fit_module.support(feat, np.ones(feat.shape[:2], bool), weights, b_l + power * delta_raw)[0]
                        expected = "interior" if power < cap else "outside"
                        assert abs(power-cap) > TOL and observed == expected
                        results[str(power)] = observed
                row = dict(trial=trial, label_people=budget, draw=draw, ordinary=ordinary, cap=cap,
                           cap_original=cap_raw, direction_recession_endpoints=rec, checks=results)
                local.append(row)
                records.append(row)
            a = [r for r in local if r["cap"] < 1 - TOL]
            b = [r for r in local if r["cap"] >= 1 - TOL]
            if a and b:
                groups.append(dict(trial=trial, label_people=budget, ordinary=ordinary, outside_unit=len(a),
                                   interior_unit=len(b), opposite_unit=len(a)*len(b),
                                   min_outside_unit_cap=min(r["cap"] for r in a),
                                   max_outside_unit_cap=max(r["cap"] for r in a)))
    assert len(records) == 384 and len(groups) == 5
    subsets = [g for g in groups if g["ordinary"] == "interior"]
    compared = [p for p in records if p["checks"]]
    summary = dict(
        total_allocations=len(records), unique_label_groups=len(original_ordinary), mixed_unit_groups=len(groups),
        mixed_unit_groups_ordinary_finite=len(subsets), unit_pairs_ordinary_finite=sum(r["opposite_unit"] for r in subsets),
        max_compression_error=max(abs(r["cap"]-r["cap_original"]) for r in records),
        validated_supporting_directions=sum(r["direction_recession_endpoints"] is not None for r in records),
        full_intervals=[min(p["cap"] for p in records), max(p["cap"] for p in records)],
        direct_state_checks=sum(len(r["checks"]) for r in compared),
        opposite_pairs_for_weight={str(power): sum(
            a["trial"] == b["trial"] and a["label_people"] == b["label_people"]
            and a["checks"][str(power)] != b["checks"][str(power)]
            for a,b in combinations(compared, 2)) for power in PROBES},
    )
    result = dict(
        status="passed", public_workbook_sha256=digest(food), plan_sha256=digest(HERE / "plan.json"),
        archived_draws_sha256=digest(earlier / "results.json"), records=records, mixed_groups=groups, summary=summary,
    )
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"summary": summary, "mixed_groups": groups}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("archive", "food", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        compute(args.archive, args.food, args.out)
