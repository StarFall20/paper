"""Test finite corrected logit while varying frequencies on a fixed menu library."""

import argparse
import csv
import hashlib
import itertools
import io
import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import qr
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
INSIDE = "interior"
OUTSIDE = "outside_mean_support"
OPPOSITE = {INSIDE, OUTSIDE}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_data(path, plan, numeric_only):
    raw_manifest = (path / "source_manifest.json").read_bytes()
    assert sha(raw_manifest) == plan["source_manifest_sha256"]
    source = json.loads(raw_manifest)
    entry, = [entry for entry in source["files"] if "train_cot" in entry["path"]]
    raw = (path / entry["path"]).read_bytes()
    assert len(raw) == entry["size"]
    checksum = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw)
    assert checksum.hexdigest() == entry["git_blob_sha1"]
    data, = numeric_only(io.BytesIO(raw)).load()
    menus, y, z = (np.asarray(data[key], float) for key in ("X", "y", "y_aug"))
    assert menus.shape == (4605, 2, 12) and y.shape == (4605,) and z.shape == (1200,)
    assert np.isfinite(menus).all() and set(y) == set(z) == {0., 1.}
    contrasts = menus[:, 1] - menus[:, 0]
    assert np.max(abs(contrasts[:, 0])) < 1e-12
    return menus, contrasts[:, 1:], y, z, source["commit"]


def frozen_forecast(x, response, design, penalty):
    scale = x.std(axis=0)
    assert (scale > 0).all()
    train = x / scale

    def value(beta):
        eta = train @ beta
        loss = np.mean(np.logaddexp(0, eta) - response * eta)
        score = train.T @ (expit(eta) - response) / len(train) + penalty * beta
        return loss + penalty * (beta @ beta) / 2, score

    fit = minimize(value, np.zeros(x.shape[1]), jac=True, method="BFGS",
                   options={"gtol": 1e-10})
    gradient = float(max(abs(value(fit.x)[1])))
    assert gradient < 1e-8
    beta = fit.x / scale
    predictions = expit(design @ beta)
    assert np.min(predictions) > 0 and np.max(predictions) < 1
    return predictions, {"coefficient": beta.tolist(), "gradient_max": gradient,
                         "min": float(min(predictions)), "max": float(max(predictions))}


def frequency_scenarios(g, plan):
    size = len(g)
    total = plan["auxiliary_rows_after_replication"]
    assert size % 5 == 0 and total == 4 * size
    rng = np.random.default_rng(plan["frequency_seed"])
    p = np.full(size, 1 / size)
    counts = {"original": np.full(size, 4, dtype=np.int64)}
    for index in (1, 2):
        counts[f"multinomial_{index}"] = 1 + rng.multinomial(total - size, p)
    ordered = np.argsort(g, kind="stable")
    ends = {"low": ordered[:size // 5], "high": ordered[-size // 5:]}
    for severity, (base, top) in {"moderate": (3, 8), "strong": (1, 16)}.items():
        for name, index in ends.items():
            freq = np.full(size, base, dtype=np.int64)
            freq[index] = top
            counts[f"{name}_{severity}"] = freq
    assert list(counts) == plan["frequency_scenarios"]
    assert all((v > 0).all() and v.sum() == total for v in counts.values())
    return counts


def moment(x_l, y_l, g_l, x_u, g_u, f):
    human = x_l.T @ y_l / len(y_l)
    delta = x_u.T @ (f * g_u) - x_l.T @ g_l / len(g_l)
    return human, delta


def support_with_interior_certificate(audit, loss):
    # A strictly positive feasible probability vector proves interiority. The
    # nonlinear fit only proposes one. An exact linear correction on a
    # full-rank set of columns enforces moment feasibility numerically.
    keep = loss.weights > 0
    x = loss.x[keep]
    w = loss.weights[keep]
    fit = minimize(loss.value, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B",
                   bounds=[(-30., 30.)] * x.shape[1],
                   options={"gtol": 1e-8, "ftol": 1e-13, "maxiter": 45})
    probability = expit(x @ fit.x)
    margin = min(min(probability), 1 - max(probability))
    if margin > 1e-6 and max(abs(fit.jac)) < 1e-5:
        matrix = x.T * w
        _, _, pivots = qr(matrix, mode="economic", pivoting=True)
        basis = pivots[:x.shape[1]]
        rank_block = matrix[:, basis]
        if np.linalg.cond(rank_block) < 1e6:
            probability[basis] += np.linalg.solve(rank_block, loss.moment - matrix @ probability)
            error = float(max(abs(matrix @ probability - loss.moment)))
            margin = float(min(min(probability), 1 - max(probability)))
            if margin > 1e-6 and error < 1e-10:
                return {"status": INSIDE, "interior_margin": margin, "equality_residual": error,
                        "certificate": "positive_probability_vector"}
    state = audit.support(loss)
    state["certificate"] = "binary_probability_program"
    return state


def audit_witness(audit, alternate, lam, x_l, y_l, g_l, x_u, g_u, counts, state):
    f = counts / counts.sum()
    human, delta = moment(x_l, y_l, g_l, x_u, g_u, f)
    x = np.vstack((x_l, x_u))
    w = np.r_[np.full(len(y_l), (1 - lam) / len(y_l)), lam * f]
    point = human + lam * delta
    loss = audit.WeightedLogit(x, w, point)
    keep = w > 0
    alternatives = np.stack((np.zeros_like(x[keep]), x[keep]), axis=1)
    available = np.ones(alternatives.shape[:2], dtype=bool)
    status, margin, recession = alternate.support(alternatives, available, w[keep], point)
    assert status == {INSIDE: "interior", OUTSIDE: "outside"}[state["status"]]
    x_expanded = np.repeat(x_u, counts, axis=0)
    g_expanded = np.repeat(g_u, counts)

    def direct(beta):
        eta_l, eta_u = x_l @ beta, x_expanded @ beta
        human_loss = np.mean(np.logaddexp(0, eta_l) - y_l * eta_l)
        soft_l = np.mean(np.logaddexp(0, eta_l) - g_l * eta_l)
        soft_u = np.mean(np.logaddexp(0, eta_u) - g_expanded * eta_u)
        score = x_l.T @ (expit(eta_l) - y_l) / len(y_l)
        score += lam * (x_expanded.T @ (expit(eta_u) - g_expanded) / len(g_expanded)
                        - x_l.T @ (expit(eta_l) - g_l) / len(g_l))
        return human_loss + lam * (soft_u - soft_l), score

    probe = np.linspace(-.75, .5, x.shape[1])
    original, score = direct(probe)
    formula, model_score = loss.value(probe)
    assert abs(original - formula) < 1e-11 and max(abs(score - model_score)) < 1e-11
    result = {"alternate_probability_status": status, "alternate_margin": margin,
              "alternate_recession": recession, "probe_loss_error": abs(original - formula),
              "probe_score_error": float(max(abs(score - model_score)))}
    if status == "interior":
        fit = minimize(loss.value, np.zeros(x.shape[1]), jac=True, hess=loss.hessian,
                       method="trust-exact", options={"gtol": 1e-10, "maxiter": 200})
        actual, score = direct(fit.x)
        min_eig = float(np.linalg.eigvalsh(loss.hessian(fit.x))[0])
        assert max(abs(score)) < 1e-8 and min_eig > 1e-12
        result.update(coefficient=fit.x.tolist(), loss=float(actual),
                      direct_score_max=float(max(abs(score))), information_min=min_eig)
    else:
        direction = np.asarray(state["direction"])
        raw_recession = (np.maximum(x_l @ direction, 0).mean()
                         - np.mean(y_l * (x_l @ direction)))
        raw_recession += lam * (np.mean(np.maximum(x_expanded @ direction, 0)
                                       - g_expanded * (x_expanded @ direction))
                                - np.mean(np.maximum(x_l @ direction, 0)
                                          - g_l * (x_l @ direction)))
        assert raw_recession < -1e-8
        assert abs(raw_recession + state["separation_gap"]) < 1e-10
        sequence = [float(direct(step * direction)[0]) for step in [100, 1000, 10000]]
        assert sequence[-1] < sequence[0] and sequence[-1] < sequence[1]
        result.update(direction=direction.tolist(), direct_recession=float(raw_recession),
                      direct_loss_at_100_1000_10000=sequence)
    return result


def run(args):
    archive = args.archive.resolve()
    sys.path[:0] = [str(archive / "results/official_ppi_runtime/reference/analysis"),
                    str(archive / "analysis"), str(archive / "results/fixed_label_auxiliary_design")]
    from public_conjoint_optimizer_audit import NumericOnly
    import power_tuned_choice_audit as audit
    import run as alternate

    assert Path(alternate.__file__).resolve() == archive / "results/fixed_label_auxiliary_design/run.py"
    plan = json.loads((HERE / "plan.json").read_text())
    menus, x, y, ai, revision = read_data(args.data, plan, NumericOnly)
    train, aux, labels = (slice(*plan[key]) for key in ("training_rows", "auxiliary_library", "label_pool"))
    assert train.stop == aux.start and aux.stop == labels.start
    x_u = x[aux]
    assert np.linalg.matrix_rank(x_u) == x.shape[1]
    distinct_full_menus = len(np.unique(menus[aux].reshape(len(x_u), -1), axis=0))
    rng = np.random.default_rng(plan["allocation_seed"])
    block = plan["source_block_size"]
    blocks = np.arange(labels.start, labels.stop).reshape(-1, block)
    allocations = [blocks[rng.permutation(len(blocks))].ravel()
                   for _ in range(plan["allocation_count"])]
    ordinary = {}
    for draw, order in enumerate(allocations):
        for budget in plan["label_budgets"]:
            idx = order[:budget]
            ordinary[draw, budget] = audit.support(audit.WeightedLogit(
                x[idx], np.full(budget, 1 / budget), x[idx].T @ y[idx] / budget))
    rows, checks, summaries, forecasts = [], {}, {}, {}
    scenarios = plan["frequency_scenarios"]
    pairs = list(itertools.combinations(scenarios, 2))
    target = args.out.resolve()
    target.mkdir(parents=True, exist_ok=True)
    for forecast, response in zip(plan["frozen_forecasts"], [y[train], ai], strict=True):
        g, info = frozen_forecast(x[train], response, x, plan["ridge_penalty_multiplier"])
        scenarios_counts = frequency_scenarios(g[aux], plan)
        info["frequency_designs"] = {name: {"counts_sha256": sha(counts.astype("<i8").tobytes()),
                                            "tv_from_original": float(abs(counts - 4).sum() / (2 * counts.sum()))}
                                     for name, counts in scenarios_counts.items()}
        forecasts[forecast] = info
        for budget in plan["label_budgets"]:
            for lam in plan["weight_grid"]:
                key = (forecast, budget, lam)
                group_count, pair_count, vs_original = 0, 0, 0
                tally = {name: {} for name in scenarios}
                for draw, order in enumerate(allocations):
                    idx = order[:budget]
                    ordinary_state = ordinary[draw, budget]["status"]
                    states = {}
                    design = np.vstack((x[idx], x_u))
                    for scenario, counts in scenarios_counts.items():
                        f = counts / counts.sum()
                        human, delta = moment(x[idx], y[idx], g[idx], x_u, g[aux], f)
                        w = np.r_[np.full(budget, (1-lam) / budget), lam * f]
                        loss = audit.WeightedLogit(design, w, human + lam * delta)
                        state = (support_with_interior_certificate(audit, loss) if args.fast
                                 else audit.support(loss))
                        states[scenario] = state
                        rows.append(dict(forecast=forecast, budget=budget, weight=lam, allocation=draw,
                                         scenario=scenario, ordinary=ordinary_state, state=state["status"],
                                         margin=state.get("interior_margin"), gap=state.get("separation_gap"),
                                         certificate=state.get("certificate", "binary_probability_program")))
                        if ordinary_state == INSIDE:
                            s = state["status"]
                            tally[scenario][s] = tally[scenario].get(s, 0) + 1
                    if ordinary_state != INSIDE:
                        continue
                    opposed = [(a, b) for a, b in pairs if {states[a]["status"], states[b]["status"]} == OPPOSITE]
                    group_count += bool(opposed)
                    pair_count += len(opposed)
                    vs_original += sum(a == "original" for a, _ in opposed)
                    if opposed and key not in checks:
                        a, b = opposed[0]
                        checks[key] = {"allocation": draw, "allocation_sha256": sha(idx.astype("<i8").tobytes()),
                                       "comparisons": {name: audit_witness(
                                           audit, alternate, lam, x[idx], y[idx], g[idx], x_u, g[aux],
                                           scenarios_counts[name], states[name]) for name in (a, b)}}
                eligible = sum(ordinary[i, budget]["status"] == INSIDE for i in range(len(allocations)))
                summaries[key] = dict(eligible=eligible, total_allocations=len(allocations),
                                      opposite_groups=group_count, opposite_pairs=pair_count,
                                      pairs_with_original=vs_original, states=tally)
                print(f"{forecast}, m={budget}, weight={lam}: {group_count}/{eligible} groups, "
                      f"{vs_original} against original", flush=True)
    result = dict(source_commit=revision, exact_ordered_full_menus=distinct_full_menus,
                  exact_utility_contrasts=len(np.unique(x_u, axis=0)),
                  evaluated_objectives=len(rows), forecasts=forecasts,
                  ordinary_states={str(b): {state: sum(ordinary[i,b]["status"] == state
                                                       for i in range(len(allocations)))
                                            for state in sorted({ordinary[i,b]["status"]
                                                                 for i in range(len(allocations))})}
                                   for b in plan["label_budgets"]},
                  groups=[dict(forecast=f, budget=b, weight=w, **summary)
                          for (f, b, w), summary in summaries.items()],
                  witnesses=[dict(forecast=f, budget=b, weight=w, **check)
                             for (f, b, w), check in checks.items()])
    (target / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (target / "fit_states.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fast", action="store_true")
    with threadpool_limits(limits=1):
        run(parser.parse_args())
