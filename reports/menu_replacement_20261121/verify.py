"""Audit how much of a fixed auxiliary menu distribution can be replaced."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csc_matrix, eye, hstack, kron, vstack
from threadpoolctl import threadpool_limits


TOL = 1e-8
POWERS = (0.75, 0.90)
FOOD_SHA256 = "447cb59fab8500639d9f5478dad4aeeeb64bc09f1d0d0848ae0219e1c42c88a0"


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def replacement_radii(labelled, menus, predictions, nominal, power, human, labelled_prediction):
    """One probability-mass feasibility program for each pure menu replacement."""
    labels, alternatives, dimension = labelled.shape
    count = len(menus)
    mu = np.einsum("ma,mad->md", predictions, menus)
    centred = np.concatenate((labelled, menus - mu[:, None, :]))
    total = labels + count
    moments = csc_matrix(centred.reshape(-1, dimension).T)
    row_sums = kron(eye(total, format="csc"), np.ones((1, alternatives)), format="csc")
    masses = np.r_[np.full(labels, (1 - power) / labels), power * nominal]
    target = np.r_[masses, human - power * labelled_prediction]
    matrix = vstack((row_sums, moments), format="csc")
    cost = np.r_[np.zeros(total * alternatives), -1.0]
    bounds = [(0, None)] * (total * alternatives) + [(0, 1)]

    radii = []
    worst_direction = None
    for menu_id in range(count):
        direction = np.r_[np.zeros(labels), -nominal, np.zeros(dimension)]
        direction[labels + menu_id] += 1
        a_eq = hstack((matrix, csc_matrix((-power * direction)[:, None])), format="csc")
        sol = linprog(cost, A_eq=a_eq, b_eq=target, bounds=bounds, method="highs-ds")
        if not sol.success:
            raise RuntimeError(sol.message)
        radius = float(sol.x[-1])
        assert np.max(np.abs(a_eq @ sol.x - target)) < TOL
        if radius < 1 - TOL:
            # Use original menu scores to check the signed directional slope independently of row masses.
            witness = sol.eqlin.marginals[-dimension:]
            assert np.max(np.abs(witness)) > TOL
            nominal_recession = (
                (1 - power) * np.max(labelled @ witness, axis=1).mean()
                + power * nominal @ (np.max(menus @ witness, axis=1) - mu @ witness)
                - (human - power * labelled_prediction) @ witness
            )
            auxiliary_recession = np.max(menus @ witness, axis=1) - mu @ witness
            slope = power * (auxiliary_recession[menu_id] - nominal @ auxiliary_recession)
            assert slope < -TOL
            assert abs(nominal_recession + radius * slope) < TOL
            if not radii or radius < min(radii):
                worst_direction = witness
        radii.append(radius)
    return np.asarray(radii), worst_direction


def original_status(support, labelled, menus, predictions, nominal, power, human, labelled_prediction):
    features = np.concatenate((labelled, menus))
    weights = np.r_[np.full(len(labelled), (1 - power) / len(labelled)), power * nominal]
    shift = np.einsum("m,ma,mad->d", nominal, predictions, menus) - labelled_prediction
    return support(features, np.ones(features.shape[:2], bool), weights, human + power * shift)[0]


def toy_check():
    labelled = np.array([[[0.0], [1.0]], [[0.0], [1.0]]])
    menus = np.array([[[0.0], [2.0]], [[0.0], [0.25]]])
    probabilities = np.full((2, 2), 0.5)
    radii, witness = replacement_radii(labelled, menus, probabilities, np.full(2, 0.5),
                                       0.75, np.array([0.5]), np.array([0.9]))
    assert np.max(np.abs(radii - [1, 79 / 105])) < TOL and witness[0] < 0
    return {"threshold_fraction": "79/105", "linear_program_radii": radii.tolist()}


def analyze(archive, food):
    from importlib.util import module_from_spec, spec_from_file_location

    setup = archive / "results/public_common_menu_shared_predictor/verify.py"
    spec = spec_from_file_location("shared_prediction_replay", setup)
    implementation = module_from_spec(spec)
    spec.loader.exec_module(implementation)
    tools = implementation.fixed_weight_tools(archive)
    source = archive / "results/fixed_label_auxiliary_design"
    fixed = tools.load(source / "run.py", "replacement_radius_choices")
    signature = tools.load(
        archive / "results/public_common_menu_diagnostics/verify.py", "replacement_radius_menu_signatures"
    ).signatures
    old = tools.read(setup.with_name("results.json"))
    previous = {(g["trial"], g["budget"]): g for g in old["groups"]}
    plan = tools.read(source / "plan.json")
    model_dir = archive / "results/power_frontier"
    files = dict(food_model=model_dir / "analysis/existing_panel_model.py",
                 food_plan=model_dir / "inputs/model_plan.json",
                 food_prediction=model_dir / "inputs/preflight.json")
    assert sha(food) == FOOD_SHA256
    assert all(sha(path) == plan["source_files_sha256"][key] for key, path in files.items())

    rows = []
    for _, trial, order, x, g, y, offered, people in fixed.food(
        argparse.Namespace(food_data=food), files, plan["label_allocations"]
    ):
        assert offered.all()
        pool = np.isin(people, order)
        keys = [signature(features, allow) for features, allow in zip(x, offered, strict=True)]
        mapping = {key: index for index, key in enumerate(sorted({keys[j] for j in np.flatnonzero(pool)}))}
        design = np.array([mapping.get(key, -1) for key in keys])
        slices = [pool & (design == index) for index in range(len(mapping))]
        menus = x[[np.flatnonzero(s)[0] for s in slices]]
        predictions = np.array([g[s].mean(axis=0) for s in slices])
        assert len(menus) == 34 and np.array_equal(x[pool], menus[design[pool]])
        for budget in plan["label_budgets"]:
            label = np.isin(people, order[:budget])
            labelled = x[label]
            human = labelled[np.arange(len(labelled)), y[label]].mean(axis=0)
            labelled_prediction = np.einsum("ta,tad->d", predictions[design[label]], labelled) / len(labelled)
            ordinary_state = fixed.support(labelled, offered[label], np.full(len(labelled), 1 / len(labelled)), human)[0]
            prior = previous[trial, budget]
            assert ordinary_state == prior["ordinary"]
            if ordinary_state != "interior":
                continue
            # Use draw zero for every group, before inspecting the radius or revised fit status.
            rng = np.random.default_rng(plan["auxiliary_generator_seed"] + 10007 * trial + 101 * budget)
            auxiliary = np.isin(people, rng.choice(order[budget:], size=plan["auxiliary_people"], replace=False))
            counts = np.bincount(design[auxiliary], minlength=len(menus))
            assert np.all(counts > 0) and counts.sum() == 384
            nominal = counts / counts.sum()
            for power in POWERS:
                def classify(frequencies):
                    return original_status(fixed.support, labelled, menus, predictions, frequencies,
                                           power, human, labelled_prediction)

                state = classify(nominal)
                cap = prior["records"][0]["weight_limit"]
                assert state == ("outside" if power > cap + TOL else "interior")
                row = dict(trial=trial, label_respondents=budget, weight=power, nominal_state=state)
                if state == "interior":
                    radii, direction = replacement_radii(labelled, menus, predictions, nominal, power,
                                                         human, labelled_prediction)
                    critical_menu = int(np.argmin(radii))
                    radius = float(radii[critical_menu])
                    row.update(radius=radius, per_menu_radii=radii.tolist())
                    if radius < 1 - TOL:
                        vertex = np.eye(len(menus))[critical_menu]
                        step = min(0.001, radius / 2, (1 - radius) / 2)
                        below = classify(nominal + (radius - step) * (vertex - nominal))
                        above = classify(nominal + (radius + step) * (vertex - nominal))
                        assert (below, above) == ("interior", "outside")
                        row.update(critical_menu_index=critical_menu, verifying_direction=direction.tolist(),
                                   below_state=below, above_state=above)
                rows.append(row)

    summaries = []
    for power in POWERS:
        subset = [r for r in rows if r["weight"] == power]
        finite = [r for r in subset if r["nominal_state"] == "interior"]
        radii = [r["radius"] for r in finite]
        summaries.append(dict(
            weight=power, label_groups=len(subset), nominal_states=Counter(r["nominal_state"] for r in subset),
            vertex_programs=34 * len(finite), below_and_above_checks=2 * sum(r < 1 - TOL for r in radii),
            smallest_radius=min(radii), median_radius=float(np.median(radii)),
            smaller_than_one=sum(r < 1 - TOL for r in radii),
            at_most_one_quarter=sum(r <= 0.25 for r in radii),
        ))
    assert len(rows) == 74
    lower = {(r["trial"], r["label_respondents"]): r for r in rows if r["weight"] == POWERS[0]}
    nested = 0
    for high in (r for r in rows if r["weight"] == POWERS[1] and "radius" in r):
        low = lower[high["trial"], high["label_respondents"]]
        assert "radius" in low
        for low_radius, high_radius in zip(low["per_menu_radii"], high["per_menu_radii"], strict=True):
            assert high_radius <= low_radius + TOL
            nested += 1
    return dict(toy=toy_check(), food_workbook_sha256=sha(food), summaries=summaries,
                paired_vertex_monotonicity_checks=nested, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("archive", "food", "out"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        result = analyze(args.archive.resolve(), args.food.resolve())
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"toy": result["toy"], "summaries": result["summaries"]}, indent=2))


if __name__ == "__main__":
    main()
