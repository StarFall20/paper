"""Compare local derivatives for fixed-label auxiliary replacements."""

import argparse
from collections import Counter
import csv
import importlib.util
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from threadpoolctl import threadpool_limits


def diagnostics(features, available, weights, target):
    masses = available / available.sum(axis=1, keepdims=True)
    means = np.einsum("ij,ijk->ik", masses, features)
    contrasts = features - means[:, None, :]
    hessian = np.einsum("i,ij,ijk,ijl->kl", weights, masses, contrasts, contrasts)
    return weights @ means - target, hessian


def discrepancies(gradients, hessians):
    basis = cholesky(hessians.mean(axis=0), lower=True)
    g = [solve_triangular(basis, row, lower=True) for row in gradients]
    h = [solve_triangular(basis, solve_triangular(basis, row, lower=True).T, lower=True)
         for row in hessians]
    gradient_norms, hessian_norms = np.linalg.norm(g, axis=1), np.linalg.norm(h, axis=(1, 2))
    for i, j in itertools.combinations(range(len(g)), 2):
        denom = gradient_norms[i] + gradient_norms[j]
        score = np.linalg.norm(g[i] - g[j]) / denom if denom else 0.0
        curvature = np.linalg.norm(h[i] - h[j]) / (hessian_norms[i] + hessian_norms[j])
        yield i, j, (score, curvature, max(score, curvature))


def numerical_check(menu, gradient, hessian):
    features, available, weights, target = menu

    def objective(beta):
        v = np.exp(features @ beta)
        return weights @ np.log(np.where(available, v, 0).sum(axis=1)) - target @ beta

    def direct_score(beta):
        v = np.where(available, np.exp(features @ beta), 0)
        contribution = np.sum(v[..., None] * features, axis=1) / v.sum(axis=1)[:, None]
        return weights @ contribution - target

    axes = np.eye(len(gradient))
    ng = np.array([objective(1e-20j * column).imag / 1e-20 for column in axes])
    step = 2e-5
    nh = np.column_stack([(direct_score(step * e) - direct_score(-step * e)) / (2 * step) for e in axes])
    return float(np.max(abs(gradient - ng))), float(np.max(abs(hessian - nh)))


def check_expected(summary, reference_path):
    expected = json.loads(reference_path.read_text())
    for key in ("corrected_designs", "all_pair_comparisons", "support_witness_designs_replayed",
                "archived_cases_sha256"):
        assert summary[key] == expected[key], key
    for actual, saved in zip(summary["cells"], expected["cells"], strict=True):
        for key in ("dataset", "power", "label_groups", "pairs", "opposite", "flipped_groups",
                    "flipped_where_ordinary_finite", "exterior_lower_decrement"):
            assert actual[key] == saved[key], key
        for key in ("median_same", "median_opposite", "closest_opposite_range"):
            np.testing.assert_allclose(actual[key], saved[key], rtol=1e-8, atol=1e-11)
    for actual, saved in zip(summary["mixed_group_witnesses"], expected["mixed_group_witnesses"], strict=True):
        for key in ("dataset", "allocation", "labels", "power", "draw_a", "draw_b", "state_a", "state_b"):
            assert actual[key] == saved[key], key
        for key in ("gradient_distance", "hessian_distance", "decrement_a", "decrement_b"):
            np.testing.assert_allclose(actual[key], saved[key], rtol=1e-8, atol=1e-11)


def run(args):
    source = args.fixed

    module = importlib.util.spec_from_file_location("fixed_auxiliary_groups", source / "run.py")
    fixed = importlib.util.module_from_spec(module)
    module.loader.exec_module(fixed)
    plan = fixed.read(source / "plan.json")
    records = source / "results/allocations.csv"
    prior = fixed.read(source / "results/summary.json")
    assert prior["allocations_sha256"] == fixed.digest(records)
    with records.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    recorded = {(r["dataset"], int(r["allocation"]), int(r["label_people"]),
                 float(r["power"]), int(r["draw"])): r for r in rows}
    assert len(recorded) == prior["corrected_allocation_records"] == 1536
    inputs = SimpleNamespace(food_data=args.food_data, swiss_data=args.swiss_data,
                             food_archive=args.food_archive, swiss_archive=args.swiss_archive)
    files = fixed.audited_files(inputs, plan)
    all_pairs, witnesses, scores_checked = [], [], 0
    errors = dict(coordinate_invariance=0.0, original_row_gradient=0.0, original_row_hessian=0.0)
    for dataset, trial, order, features, predictions, choices, available, ids in itertools.chain(
        fixed.food(inputs, files, plan["label_allocations"]),
        fixed.swiss(inputs, files, plan["label_allocations"]),
    ):
        k = features.shape[-1]
        change = np.diag(np.geomspace(0.25, 4.0, k)) + 0.15 * np.roll(np.eye(k), 1, axis=1)
        for budget in plan["label_budgets"]:
            take = np.isin(ids, order[:budget])
            rng = np.random.default_rng(plan["auxiliary_generator_seed"] + 10007 * trial + 101 * budget)
            auxiliary = [
                np.isin(ids, rng.choice(order[budget:], size=plan["auxiliary_people"], replace=False))
                for _ in range(plan["auxiliary_draws"])
            ]
            for power in plan["powers"]:
                menus = [fixed.power_inputs(features, available, choices, predictions, take, part, power)
                         for part in auxiliary]
                observed = [recorded[dataset, trial, budget, power, draw] for draw in range(len(menus))]
                assert all(int(r["label_tasks"]) == take.sum() and int(r["aux_tasks"]) == part.sum()
                           for r, part in zip(observed, auxiliary))
                grads, hs = map(np.asarray, zip(*(diagnostics(*menu) for menu in menus)))
                assert np.linalg.eigvalsh(hs).min() > 0
                dec = np.sqrt(np.einsum("ni,ni->n", grads, np.linalg.solve(hs, grads[..., None])[..., 0]))
                scores_checked += len(menus)
                # Right-multiply the original features to test whitening under a different utility encoding.
                scaled = [diagnostics(x @ change, av, weight, target @ change)
                          for x, av, weight, target in menus]
                scaled_g, scaled_h = map(np.asarray, zip(*scaled))
                group_pairs = []
                for (i, j, raw), (u, v, alternate) in zip(
                    discrepancies(grads, hs), discrepancies(scaled_g, scaled_h), strict=True
                ):
                    assert (i, j) == (u, v)
                    error = float(np.max(abs(np.array(raw) - alternate)))
                    errors["coordinate_invariance"] = max(errors["coordinate_invariance"], error)
                    si, sj = observed[i]["status"], observed[j]["status"]
                    assert {si, sj} <= {"interior", "outside"}
                    outside_smaller = None
                    if si != sj:
                        exterior, interior = (i, j) if si == "outside" else (j, i)
                        outside_smaller = bool(dec[exterior] < dec[interior])
                    row = dict(dataset=dataset, allocation=trial, labels=budget, power=power,
                               draw_a=i, draw_b=j, state_a=si, state_b=sj, gradient_distance=float(raw[0]),
                               hessian_distance=float(raw[1]), distance=float(raw[2]),
                               decrement_a=float(dec[i]), decrement_b=float(dec[j]),
                               exterior_lower_decrement=outside_smaller)
                    group_pairs.append(row)
                all_pairs.extend(group_pairs)
                opposite = [r for r in group_pairs if r["state_a"] != r["state_b"]]
                if not opposite:
                    continue
                choice = min(opposite, key=lambda r: (r["distance"], r["draw_a"], r["draw_b"]))
                replay = []
                for index in (choice["draw_a"], choice["draw_b"]):
                    state, margin, recession = fixed.support(*menus[index])
                    assert state == observed[index]["status"]
                    eg, eh = numerical_check(menus[index], grads[index], hs[index])
                    errors["original_row_gradient"] = max(errors["original_row_gradient"], eg)
                    errors["original_row_hessian"] = max(errors["original_row_hessian"], eh)
                    replay.append(dict(draw=index, state=state, fraction=margin, negative_recession=recession))
                witnesses.append(dict(**choice, support_replay=replay,
                                      ordinary_fit=observed[0]["labelled_only"]))
    assert len(all_pairs) == 5376 and scores_checked == len(recorded)
    assert errors["coordinate_invariance"] < 1e-10
    assert errors["original_row_gradient"] < 1e-10
    assert errors["original_row_hessian"] < 2e-7
    cells = []
    for dataset, power in itertools.product(("RADIANT", "Swissmetro"), (1.0, 0.5)):
        subset = [r for r in all_pairs if r["dataset"] == dataset and r["power"] == power]
        opposite = [r for r in subset if r["state_a"] != r["state_b"]]
        same = [r for r in subset if r["state_a"] == r["state_b"]]
        nearest = [r for r in witnesses if r["dataset"] == dataset and r["power"] == power]
        low, high = min(nearest, key=lambda r: r["distance"]), max(nearest, key=lambda r: r["distance"])
        cells.append(dict(dataset=dataset, power=power, label_groups=len(subset) // 28,
                          pairs=len(subset), opposite=len(opposite), flipped_groups=len(nearest),
                          flipped_where_ordinary_finite=sum(r["ordinary_fit"] == "interior" for r in nearest),
                          median_same=float(np.median([r["distance"] for r in same])),
                          median_opposite=float(np.median([r["distance"] for r in opposite])),
                          closest_opposite_range=[low["distance"], high["distance"]],
                          closest_opposite=low,
                          exterior_lower_decrement=sum(r["exterior_lower_decrement"] for r in opposite)))
    got = Counter((r["power"] for r in witnesses))
    assert got == {1: 12, 0.5: 2}
    args.out.mkdir(parents=True)
    with (args.out / "pair_distances.csv").open("w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=list(all_pairs[0]))
        writer.writeheader()
        writer.writerows(all_pairs)
    summary = dict(status="passed", corrected_designs=scores_checked, all_pair_comparisons=len(all_pairs),
                   support_witness_designs_replayed=2 * len(witnesses), maximum_errors=errors,
                   cells=cells, mixed_group_witnesses=witnesses,
                   archived_cases_sha256=fixed.digest(records))
    if args.reference:
        check_expected(summary, args.reference)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "mixed_group_witnesses"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("fixed", "food-archive", "food-data", "swiss-archive", "swiss-data", "out"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    with threadpool_limits(limits=1):
        run(parser.parse_args())
