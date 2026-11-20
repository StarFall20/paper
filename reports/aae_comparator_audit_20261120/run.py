"""Equation-level comparison on fixed public choice records."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import least_squares, linprog, minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OFFICIAL = REPO / "reports/official_ppi_runtime_audit_20261117"
OLD = REPO / "reports/public_holdout_20261114/results"
sys.path[:0] = [str(OFFICIAL / "reference/analysis"), str(REPO / "analysis")]

from public_conjoint_optimizer_audit import load_data, save
from public_conjoint_matched_audit import nuisance
from power_tuned_choice_audit import WeightedLogit, objective, support


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_fit(x, w, b, known_inside=False):
    loss = WeightedLogit(x, w, b)
    if not known_inside:
        check = support(loss)
        if check["status"] != "interior":
            return None, check["status"]
    fit = minimize(
        loss.value, np.zeros(x.shape[1]), jac=True, hess=loss.hessian,
        method="trust-exact", options={"gtol": 1e-10, "maxiter": 200},
    )
    beta = fit.x
    score = loss.value(beta)[1]
    info = loss.hessian(beta)
    # A Newton polish removes optimizer stopping variation on near-flat losses.
    if np.max(abs(score)) < 1e-7:
        beta -= np.linalg.solve(info, score)
    assert np.isfinite(beta).all()
    assert np.max(abs(loss.value(beta)[1])) < 1e-8
    assert np.linalg.eigvalsh(loss.hessian(beta))[0] > 1e-12
    return beta, "finite_root"


def models(x, y, g, m):
    n = len(y)
    ratio = m / n
    ux = x[m:]
    pooled = np.full(n, 1 / n)
    aux = np.full(len(ux), 1 / len(ux))
    delta = x[:m].T @ (y[:m] - g[:m]) / m
    center = x.T @ g / n
    acenter = ux.T @ g[m:] / len(ux)
    return delta, {
        "auxiliary_AAE": (ux, aux, acenter, True),
        "pooled_AAE": (x, pooled, center, True),
        "pooled_human_and_soft": (x, pooled, center + ratio * delta, True),
        "pooled_PPI": (x, pooled, center + delta, False),
    }


def direct_score(method, x, y, g, m, beta):
    p = expit(x @ beta)
    label = x[:m].T @ (p[:m] - y[:m]) / m
    lsoft = x[:m].T @ (p[:m] - g[:m]) / m
    usoft = x[m:].T @ (p[m:] - g[m:]) / (len(g) - m)
    ratio = m / len(g)
    return {
        "auxiliary_AAE": usoft,
        "pooled_AAE": ratio * lsoft + (1 - ratio) * usoft,
        "pooled_human_and_soft": ratio * label + (1 - ratio) * usoft,
        "pooled_PPI": label + (1 - ratio) * (usoft - lsoft),
    }[method]


def calculate_frontier(x, w, center, delta):
    """Primal ray endpoint and independent normalized supporting hyperplane."""
    n, d = x.shape
    target = sparse.csr_matrix((x * w[:, None]).T)
    ray_column = sparse.csr_matrix(-delta[:, None])
    primal = linprog(
        np.r_[np.zeros(n), -1.], A_eq=sparse.hstack([target, ray_column]),
        b_eq=center, bounds=[(0., 1.)] * n + [(0., None)], method="highs",
    )
    assert primal.success, primal.message
    endpoint = float(primal.x[-1])
    residual = float(np.max(abs(target @ primal.x[:-1] - center - endpoint * delta)))
    assert residual < 1e-8 and endpoint > 0

    # h_K(d) - d'center is the dual objective when d'delta = 1.
    dual = linprog(
        np.r_[-center, w], A_ub=sparse.hstack([sparse.csr_matrix(x), -sparse.eye(n)]),
        b_ub=np.zeros(n), A_eq=np.r_[delta, np.zeros(n)][None, :],
        b_eq=[1.], bounds=[(None, None)] * d + [(0., None)] * n, method="highs",
    )
    assert dual.success, dual.message
    direction = dual.x[:d]
    normalized_error = float(abs(direction @ delta - 1))
    actual_dual = float(w @ np.maximum(x @ direction, 0.) - direction @ center)
    assert normalized_error < 1e-8 and abs(endpoint - actual_dual) < 1e-8
    return dict(
        frontier=endpoint, primal_residual=residual, dual_frontier=actual_dual,
        dual_normalization_error=normalized_error,
        unit_recession_gap=float(w @ np.maximum(x @ direction, 0.) - direction @ (center + delta)),
        direction=direction.tolist(),
    )


def same_label_refit(x, y, z, m, prediction):
    extra = np.column_stack([np.ones(len(x)), z]) if prediction == "AI" else np.ones((len(x), 1))
    design = np.column_stack([x, extra])
    xl = design[:m]
    beta, status = finite_fit(xl, np.full(m, 1 / m), xl.T @ y[:m] / m)
    if beta is None:
        return dict(refit_status=status)
    g = expit(design @ beta)
    assert np.min(g) > 0 and np.max(g) < 1
    delta, specs = models(x, y, g, m)
    assert np.max(abs(delta)) < 1e-8
    shifts = []
    for name in ["auxiliary_AAE", "pooled_AAE"]:
        xx, w, center, _ = specs[name]
        b0, _ = finite_fit(xx, w, center, True)
        b1, _ = finite_fit(xx, w, center + delta, True)
        shifts.append(float(max(abs(b0-b1))))
    assert max(shifts) < 1e-6
    return dict(refit_status=status, max_abs_focal_residual=float(max(abs(delta))),
                aux_beta_difference=shifts[0], pooled_beta_difference=shifts[1])


def evaluate(x, y, beta):
    u = x @ beta
    return float(np.mean(np.logaddexp(0, u) - y * u)), float(np.mean((expit(u) - y) ** 2))


def run(data, out):
    plan = json.loads((HERE / "plan.json").read_text())
    assert sha(data / "source_manifest.json") == plan["source_manifest_sha256"]
    assert sha(OLD / "allocation_order_sha256.json") == plan["existing_order_log_sha256"]
    assert sha(OLD / "fit_records.csv") == plan["existing_fit_records_sha256"]
    official_plan = OFFICIAL / "plan.json"
    assert sha(official_plan) == "79fe84e95fe21668dfd26cbf13137d0121b1c9f5fd79dc2a22074908062d0f6c"
    selected = {(r["trial"], r["prediction"]): r["existing_unit_support"]
                for r in json.loads(official_plan.read_text())["public_cases"]}
    assert len(selected) == 16
    order_hashes = json.loads((OLD / "allocation_order_sha256.json").read_text())
    old = pd.read_csv(OLD / "fit_records.csv").set_index(["trial", "label_rows", "prediction", "method"])
    source_x, source_y, z, _, _ = load_data(data)
    n = plan["source_prefix_rows"]
    test_x, test_y = source_x[n:], source_y[n:]
    x, y = source_x[:n], source_y[:n]
    assert len(test_y) == plan["evaluation_suffix_rows"] and len(z) == n
    assert len(order_hashes) == plan["allocations"] == 500
    rng = np.random.RandomState(plan["allocation_seed"])
    block = plan["source_block_size"]
    rows, witnesses, calibrated = [], [], []
    score_error, form_error, old_loss_error, old_brier_error, root_replay_error = [], [], [], [], []
    for trial in range(plan["allocations"]):
        order = (rng.permutation(n // block)[:, None]*block + np.arange(block)).reshape(-1)
        assert hashlib.sha256(order.astype("<i8").tobytes()).hexdigest() == order_hashes[trial]
        xx, yy, zz = x[order], y[order], z[order]
        for m in plan["label_budgets"]:
            for prediction in plan["prediction_routines"]:
                g, fallback = nuisance(xx, yy, zz, m, prediction == "AI")
                assert np.min(g) > 0 and np.max(g) < 1
                delta, specs = models(xx, yy, g, m)
                assert (np.linalg.matrix_rank(xx) == xx.shape[1] ==
                        np.linalg.matrix_rank(xx[m:]))
                selected_case = m == 50 and (trial, prediction) in selected
                for name, (features, w, moment, known_inside) in specs.items():
                    beta, status = finite_fit(features, w, moment, known_inside)
                    row = dict(trial=trial, label_rows=m, prediction=prediction, method=name,
                               status=status, fallback_rows=fallback, logloss=None, brier=None,
                               focal_residual_inf=float(max(abs(delta))))
                    if beta is not None:
                        row["logloss"], row["brier"] = evaluate(test_x, test_y, beta)
                        err = float(max(abs(direct_score(name, xx, yy, g, m, beta))))
                        score_error.append(err)
                        assert err < 1e-8
                        if selected_case:
                            check = least_squares(
                                lambda b: direct_score(name, xx, yy, g, m, b),
                                np.zeros(xx.shape[1]), gtol=1e-13, xtol=1e-13, ftol=1e-13,
                            )
                            assert max(abs(check.fun)) < 1e-8
                            root_replay_error.append(float(max(abs(check.x - beta))))
                        if name == "pooled_human_and_soft":
                            ref = old.loc[(trial, m, prediction, "soft_pooling")]
                            old_loss_error.append(abs(row["logloss"]-ref["logloss"]))
                            old_brier_error.append(abs(row["brier"]-ref["brier"]))
                    rows.append(row)

                # Compare independently written objective forms, with the same nonzero probe.
                ratio = m/n
                probe = np.linspace(-.6, .4, xx.shape[1])
                b = specs["pooled_PPI"][2]
                val = WeightedLogit(xx, specs["pooled_PPI"][1], b).value(probe)[0]
                orig = objective(xx, yy, g, m, 1-ratio).value(probe)[0]
                base = WeightedLogit(xx, specs["pooled_AAE"][1], specs["pooled_AAE"][2]).value(probe)[0]
                mix = WeightedLogit(xx, specs["pooled_human_and_soft"][1], specs["pooled_human_and_soft"][2]).value(probe)[0]
                form_error += [abs(val-orig), abs((1-ratio)*base+ratio*val-mix)]

                if selected_case:
                    expected = selected[(trial, prediction)]
                    for name in ("auxiliary_AAE", "pooled_AAE"):
                        features, w, moment, _ = specs[name]
                        proof = calculate_frontier(features, w, moment, delta)
                        if name == "auxiliary_AAE":
                            assert (proof["frontier"] < 1) == (expected == "outside_mean_support")
                        else:
                            assert (proof["frontier"] > 1) == (status == "finite_root")
                        witnesses.append(dict(trial=trial, prediction=prediction, menu_group=name,
                                              existing_unit_status=expected, **proof))
                    calibrated.append(dict(trial=trial, prediction=prediction, existing_unit_status=expected,
                                           **same_label_refit(xx, yy, zz, m, prediction)))
        if (trial + 1) % 100 == 0:
            print(f"{trial+1}/500 fixed allocations", flush=True)

    assert len(rows) == 8000 and len(witnesses) == 32 and len(calibrated) == 16
    assert len(old_loss_error) == 2000 and max(old_loss_error) < 1e-7 and max(old_brier_error) < 1e-7
    assert max(form_error) < 1e-12 and max(root_replay_error) < 1e-6
    frame = pd.DataFrame(rows)
    assert set(frame.status) <= {"finite_root", "outside_mean_support", "boundary_or_near_boundary"}
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "fit_records.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    summary = frame.groupby(["label_rows", "prediction", "method"]).agg(
        allocations=("trial", "size"), finite=("logloss", "count"), mean_logloss=("logloss", "mean"),
        mean_brier=("brier", "mean")).reset_index()
    summary.to_csv(out / "summary.csv", index=False)
    frame.groupby(["label_rows", "prediction", "method", "status"]).size().rename("count").to_csv(out / "status_counts.csv")
    paired = []
    for (m, pred), group in frame.groupby(["label_rows", "prediction"]):
        pivot = group.pivot(index="trial", columns="method", values="logloss")
        for a, b in [("pooled_AAE", "auxiliary_AAE"), ("pooled_PPI", "pooled_AAE"),
                     ("pooled_human_and_soft", "pooled_AAE")]:
            diff = (pivot[a]-pivot[b]).dropna()
            paired.append(dict(label_rows=m, prediction=pred, first=a, second=b, shared_finite=len(diff),
                               mean_logloss_difference=float(diff.mean()), first_lower=int((diff < 0).sum())))
    pd.DataFrame(paired).to_csv(out / "paired_logloss.csv", index=False)
    save(out / "witnesses.json", dict(frontier=witnesses, same_label_refits=calibrated))
    checks = dict(
        plan_sha256=sha(HERE / "plan.json"), runner_sha256=sha(Path(__file__)),
        canonical_reader_sha256=sha(OFFICIAL / "reference/analysis/public_conjoint_optimizer_audit.py"),
        canonical_nuisance_sha256=sha(OFFICIAL / "reference/analysis/public_conjoint_matched_audit.py"),
        canonical_support_sha256=sha(REPO / "analysis/power_tuned_choice_audit.py"),
        fit_count=len(rows), order_hashes_matched=len(order_hashes), direct_probability_scores=len(score_error),
        max_direct_score=max(score_error), max_objective_form_difference=max(form_error),
        max_previous_soft_pool_logloss_difference=max(old_loss_error),
        max_previous_soft_pool_brier_difference=max(old_brier_error),
        least_squares_roots_rechecked=len(root_replay_error), max_alternative_root_difference=max(root_replay_error),
        primal_dual_frontiers=len(witnesses),
        max_primal_dual_frontier_difference=max(abs(a["frontier"]-a["dual_frontier"]) for a in witnesses),
    )
    save(out / "verification.json", checks)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.data, args.out)
