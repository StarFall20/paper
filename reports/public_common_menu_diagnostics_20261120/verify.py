"""Replay the published auxiliary draws and test equality of complete menu sets."""

from argparse import ArgumentParser, Namespace
from collections import Counter
import csv
import hashlib
from itertools import chain, combinations
import json
from pathlib import Path

import numpy as np

def signatures(x, available):
    menu = x[available]
    contrast = menu - menu[0]
    # Include availability so that offered mode identities remain distinguishable.
    return tuple(available.tolist()), tuple(contrast.ravel().tolist())


def group_key(dataset, trial, budget, draw, power):
    return dataset, int(trial), int(budget), int(draw), float(power)


def analyze(archive, food_data, swiss_data, out):
    match = archive / "results/fixed_label_auxiliary_design"
    options = Namespace(
        food_archive=archive / "results/power_frontier",
        food_data=food_data,
        swiss_archive=archive / "results/swissmetro_corrected_path",
        swiss_data=swiss_data,
    )
    import importlib.util
    path = match / "run.py"
    spec = importlib.util.spec_from_file_location("fixed_label_public", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan = module.read(match / "plan.json")
    files = module.audited_files(options, plan)
    source_result = match / "results/allocations.csv"
    archived = {}
    with source_result.open() as handle:
        for row in csv.DictReader(handle):
            key = group_key(row["dataset"], row["allocation"], row["label_people"], row["draw"], row["power"])
            assert key not in archived
            archived[key] = row
    tally = {}
    examples = []
    common_forecast = {}
    replay = chain(module.food(options, files, plan["label_allocations"]),
                   module.swiss(options, files, plan["label_allocations"]))
    for dataset, trial, order, x, g, y, availability, people in replay:
        rows = [signatures(xx, a) for xx, a in zip(x, availability)]
        if dataset == "RADIANT":
            reference_pool = np.isin(people, order)
            levels = sorted({rows[i] for i in np.flatnonzero(reference_pool)})
            by_menu = {level: index for index, level in enumerate(levels)}
            ids = np.array([by_menu.get(row, -1) for row in rows])
            values = np.array([g[reference_pool & (ids == i)].mean(axis=0) for i in range(len(levels))])
            exemplar = [np.flatnonzero(reference_pool & (ids == i))[0] for i in range(len(levels))]
            # All available alternatives share the stored standardized feature convention.
            assert all(np.array_equal(x[index], x[exemplar[ids[index]]])
                       for index in np.flatnonzero(reference_pool))
        for budget in plan["label_budgets"]:
            labelled = np.isin(people, order[:budget])
            label_set = {rows[i] for i in np.flatnonzero(labelled)}
            rng = np.random.default_rng(plan["auxiliary_generator_seed"] + 10007*trial + 101*budget)
            draws = []
            common_draws = []
            for draw in range(plan["auxiliary_draws"]):
                selected = rng.choice(order[budget:], size=plan["auxiliary_people"], replace=False)
                subset = np.isin(people, selected)
                assert not np.any(labelled & subset)
                frequencies = Counter(rows[i] for i in np.flatnonzero(subset))
                ref = archived[group_key(dataset, trial, budget, draw, 0.5)]
                assert int(subset.sum()) == int(ref["aux_tasks"])
                draws.append(frequencies)
                if dataset == "RADIANT":
                    common_draws.append((subset, np.array([frequencies[level] for level in levels])))
            for power in plan["powers"]:
                summary = tally.setdefault((dataset, power), Counter())
                for left, right in combinations(range(len(draws)), 2):
                    summary["all_pairs"] += 1
                    a, b = draws[left], draws[right]
                    statuses = [archived[group_key(dataset, trial, budget, d, power)]["status"] for d in (left, right)]
                    opp = set(statuses) == {"interior", "outside"}
                    summary["interior_exterior_pairs"] += opp
                    if a.keys() != b.keys():
                        continue
                    summary["same_aux_menu_pairs"] += 1
                    summary["same_menu_pairs_and_changed_counts"] += a != b
                    summary["same_menu_opposite_states"] += opp
                    summary["same_menu_including_labelled"] += a.keys() == label_set
                    summary["all_samples_same_menu_opposite_states"] += opp and a.keys() == label_set
                    if opp:
                        # Verify the original, uncompressed probability equations in both fits.
                        for draw in (left, right):
                            regen_rng = np.random.default_rng(plan["auxiliary_generator_seed"] + 10007*trial + 101*budget)
                            for _ in range(draw + 1):
                                selected = regen_rng.choice(order[budget:], size=plan["auxiliary_people"], replace=False)
                            unseen = np.isin(people, selected)
                            state, margin, ray = module.support(*module.power_inputs(x, availability, y, g, labelled, unseen, power))
                            assert state == archived[group_key(dataset, trial, budget, draw, power)]["status"]
                        examples.append(dict(dataset=dataset, trial=trial, budget=budget, power=power,
                                             auxiliary_draws=[left, right], states=statuses,
                                             menu_count=len(a), labelled_menu_count=len(label_set),
                                             labelled_also_same=label_set == a.keys(),
                                             equal_auxiliary_task_counts=sum(a.values()) == sum(b.values())))
                if dataset != "RADIANT":
                    continue
                x_label = x[labelled]
                label_mean = x_label[np.arange(labelled.sum()), y[labelled]].mean(axis=0)
                label_proxy_mean = np.einsum("ij,ijp->p", g[labelled], x_label) / labelled.sum()
                x_all = x[exemplar]
                aux_proxy = np.einsum("ij,ijp->ip", values, x_all)
                fixed = common_forecast.setdefault(power, {
                    "counts": Counter(), "allocation_records": [],
                })
                statuses = []
                for draw, (subset, counts) in enumerate(common_draws):
                    assert np.all(counts > 0)
                    selected_weights = counts / counts.sum()
                    target = label_mean + power * (selected_weights @ aux_proxy - label_proxy_mean)
                    features = x_all if power == 1 else np.concatenate((x_label, x_all))
                    chosen = availability[exemplar] if power == 1 else np.concatenate((availability[labelled], availability[exemplar]))
                    weights = selected_weights if power == 1 else np.r_[np.full(labelled.sum(), (1-power)/labelled.sum()), power*selected_weights]
                    state, margin, ray = module.support(features, chosen, weights, target)
                    statuses.append(state)
                    row = dict(trial=trial, label_people=budget, draw=draw, state=state, recession=ray,
                               probability_margin=margin)
                    fixed["allocation_records"].append(row)
                    fixed["counts"][state] += 1
                    uncompressed_g = g.copy()
                    uncompressed_g[subset] = values[ids[subset]]
                    stats = module.power_inputs(x, availability, y, uncompressed_g, labelled, subset, power)
                    assert np.max(abs(target - stats[-1])) < 1e-12
                    status, original_margin, original_ray = module.support(*stats)
                    assert status == state
                    if margin is not None:
                        assert abs(original_margin - margin) < 1e-8
                    if ray is not None:
                        assert abs(original_ray - ray) < 1e-8
                fixed["counts"]["groups"] += 1
                fixed["counts"]["mixed_groups"] += {"interior", "outside"} <= set(statuses)
                fixed["counts"]["opposite_state_pairs"] += sum(
                    {a,b} == {"interior", "outside"} for a,b in combinations(statuses, 2))
    tally_fields = ("all_pairs", "interior_exterior_pairs", "same_aux_menu_pairs",
                    "same_menu_pairs_and_changed_counts", "same_menu_opposite_states",
                    "same_menu_including_labelled", "all_samples_same_menu_opposite_states")
    payload = dict(
        scope="Complete alternative sets with identical available identities and exact encoded relative attributes. "
              "No attribute rounding. Fixed archived auxiliary draws.",
        plan_sha256=hashlib.sha256((match / "plan.json").read_bytes()).hexdigest(),
        archived_states_sha256=hashlib.sha256(source_result.read_bytes()).hexdigest(),
        summaries=[dict(dataset=dataset, power=power, **{key: c[key] for key in tally_fields})
                   for (dataset, power), c in tally.items()],
        reverified_same_menu_opposite_state_pairs=examples,
        common_forecast_by_exact_auxiliary_menu=common_forecast,
        common_forecast_definition="One frozen mean vector per menu across the RADIANT correction pool. "
                                   "Leave the labelled forecasts intact and fix the per-menu vector in all auxiliary draws. "
                                   "Every finite state is rechecked using both compressed and original task rows.",
    )
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summaries"], indent=2))
    print(json.dumps({weight:v["counts"] for weight,v in common_forecast.items()},indent=2))


if __name__ == "__main__":
    p = ArgumentParser()
    for name in ("archive", "food-data", "swiss-data", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    analyze(args.archive.resolve(), args.food_data.resolve(), args.swiss_data.resolve(), args.out)
