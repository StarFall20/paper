"""Audit the residual path with fixed menus and two prediction-fold conventions."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import least_squares, linprog
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
MATCHED = HERE.parent / "aae_comparator_audit_20261120"
if not MATCHED.is_dir():
    MATCHED = HERE.parent / "matched_all_soft_choice"
sys.path.insert(0, str(MATCHED))
from run import OLD, evaluate, finite_fit, load_data, nuisance, save, sha


def grouped_predictions(x, y, z, m, block, use_ai):
    """Hold out entire released blocks. All fallback probabilities use training labels."""
    assert m % block == 0
    leave = np.full(m, np.nan)
    external = np.zeros(len(y) - m)
    offset = np.arange(block)
    counts = {"validation_fallback_rows": 0, "auxiliary_fallback_fold_rows": 0}
    for train_blocks, test_blocks in KFold(5, shuffle=True, random_state=0).split(
        np.arange(m // block)
    ):
        train = (train_blocks[:, None] * block + offset).ravel()
        test = (test_blocks[:, None] * block + offset).ravel()
        assert not np.intersect1d(train // block, test // block).size
        for group in ([0, 1] if use_ai else [None]):
            tr = train[z[train] == group] if use_ai else train
            te = test[z[test] == group] if use_ai else test
            au = m + np.flatnonzero(z[m:] == group) if use_ai else np.arange(m, len(y))
            if len(tr) >= 2 and np.unique(y[tr]).size == 2:
                fit = LogisticRegression(C=.05, max_iter=2000, random_state=0).fit(x[tr], y[tr])
                if te.size:
                    leave[te] = fit.predict_proba(x[te])[:, 1]
                if au.size:
                    external[au - m] += fit.predict_proba(x[au])[:, 1]
            else:
                base = tr if tr.size else train
                prior = (y[base].sum() + .5) / (len(base) + 1)
                leave[te] = prior
                external[au - m] += prior
                counts["validation_fallback_rows"] += len(te)
                counts["auxiliary_fallback_fold_rows"] += len(au)
    return np.r_[leave, external / 5], counts


def support_endpoint(x, weights, center, residual, support_rows):
    """Return independent primal and dual checks with exact duplicate contrasts collapsed."""
    features, inverse, counts, inequalities = support_rows
    scores = sparse.csr_matrix(features.T * (counts / len(x)))
    variables = scores.shape[1]
    result = linprog(
        np.r_[np.zeros(variables), -1.0],
        A_eq=sparse.hstack([scores, sparse.csr_matrix(-residual[:, None])]),
        b_eq=center, bounds=[(0.0, 1.0)] * variables + [(0.0, None)],
        method="highs",
    )
    assert result.success, result.message
    endpoint = float(result.x[-1])
    endpoint_prob = result.x[:-1][inverse]
    n_contrasts, dimensions = features.shape
    dual = linprog(
        np.r_[-center, counts / len(x)], A_ub=inequalities, b_ub=np.zeros(n_contrasts),
        A_eq=np.r_[residual, np.zeros(n_contrasts)][None, :], b_eq=[1.0],
        bounds=[(None, None)] * dimensions + [(0.0, None)] * n_contrasts,
        method="highs",
    )
    assert dual.success, dual.message
    direction = dual.x[:dimensions]
    upper = weights @ np.maximum(x @ direction, 0.0) - center @ direction
    gap = abs(endpoint - upper)
    normal_error = abs(residual @ direction - 1.0)
    primal_error = np.max(abs(x.T @ (weights * endpoint_prob) - center - endpoint * residual))
    assert endpoint > 0 and max(gap, normal_error, primal_error) < 1e-8
    return endpoint, endpoint_prob, (float(gap), float(normal_error), float(primal_error))


def direct_score(x, y, g, m, multiplier, beta):
    return (x.T @ (expit(x @ beta) - g) / len(y)
            - multiplier * (x[:m].T @ (y[:m] - g[:m])) / m)


def run(data, scheme, out):
    plan = json.loads((HERE / "plan.json").read_text())
    assert sha(data / "source_manifest.json") == plan["source_manifest_sha256"]
    assert sha(MATCHED / "plan.json") == plan["matched_run_plan_sha256"]
    records_file = MATCHED / "results/fit_records.csv.gz"
    assert sha(records_file) == plan["matched_records_sha256"]
    matched_plan = json.loads((MATCHED / "plan.json").read_text())
    assert sha(OLD / "allocation_order_sha256.json") == matched_plan["existing_order_log_sha256"]
    assert sha(OLD / "fit_records.csv") == matched_plan["existing_fit_records_sha256"]
    previous = pd.read_csv(records_file).set_index(["trial", "label_rows", "prediction", "method"])
    ridge = (pd.read_csv(OLD / "fit_records.csv")
             .query("method == 'human_ridge'")
             .set_index(["trial", "label_rows", "prediction"]))
    order_hashes = json.loads((OLD / "allocation_order_sha256.json").read_text())
    source_x, source_y, z, _, _ = load_data(data)
    n = matched_plan["source_prefix_rows"]
    x, y, test_x, test_y = source_x[:n], source_y[:n], source_x[n:], source_y[n:]
    assert len(test_y) == matched_plan["evaluation_suffix_rows"]
    weights = np.full(n, 1.0 / n)
    rng = np.random.RandomState(matched_plan["allocation_seed"])
    block = matched_plan["source_block_size"]
    support_features, support_inverse, support_counts = np.unique(
        x, axis=0, return_inverse=True, return_counts=True,
    )
    inequalities = sparse.hstack([
        sparse.csr_matrix(support_features), -sparse.eye(len(support_counts)),
    ])
    rows, checks = [], []
    for trial in range(matched_plan["allocations"]):
        order = (rng.permutation(n // block)[:, None] * block + np.arange(block)).reshape(-1)
        assert hashlib.sha256(order.astype("<i8").tobytes()).hexdigest() == order_hashes[trial]
        xx, yy, zz = x[order], y[order], z[order]
        support = support_features, support_inverse[order], support_counts, inequalities
        for m in matched_plan["label_budgets"]:
            for prediction in matched_plan["prediction_routines"]:
                use_ai = prediction == "AI"
                if scheme == "block":
                    g, fallback = grouped_predictions(xx, yy, zz, m, block, use_ai)
                else:
                    g, count = nuisance(xx, yy, zz, m, use_ai)
                    fallback = {"validation_fallback_rows": count}
                assert np.isfinite(g).all() and (g > 0).all() and (g < 1).all()
                center = xx.T @ g / n
                residual = xx[:m].T @ (yy[:m] - g[:m]) / m
                endpoint, endpoint_prob, errors = support_endpoint(
                    xx, weights, center, residual, support,
                )
                protected = min(1.0, .5 * endpoint)
                fraction = protected / endpoint
                mix = (1.0 - fraction) * g + fraction * endpoint_prob
                witness_error = float(np.max(abs(xx.T @ mix / n - center - protected * residual)))
                assert (mix > 0).all() and (mix < 1).all() and witness_error < 1e-8
                key = (trial, m, prediction)
                ridge_record = ridge.loc[key]
                assert ridge_record.status == "finite_penalized_root"
                methods = {
                    "all_soft": 0.0, "human_and_soft": m / n, "fixed_quarter": .25,
                    "fixed_half": .5, "protected_half_margin": protected, "full": 1.0,
                }
                archived_names = {
                    "all_soft": "pooled_AAE", "human_and_soft": "pooled_human_and_soft",
                    "full": "pooled_PPI",
                }
                metrics = {
                    "max_direct_score": 0.0, "largest_alternative_root_error": 0.0,
                    "max_archived_loss_error": 0.0, "alternative_roots": 0,
                    "archived_checks": 0, "finite_scores_checked": 0,
                }
                all_soft = None
                for name, multiplier in methods.items():
                    assert abs(multiplier - endpoint) > 1e-7
                    if multiplier < endpoint:
                        target = center + multiplier * residual
                        beta, status = finite_fit(xx, weights, target, known_inside=True)
                        assert status == "finite_root"
                        score_error = float(max(abs(direct_score(xx, yy, g, m, multiplier, beta))))
                        assert score_error < 1e-8
                        metrics["max_direct_score"] = max(metrics["max_direct_score"], score_error)
                        metrics["finite_scores_checked"] += 1
                        if trial % 25 == 0:
                            refit = least_squares(
                                lambda b: direct_score(xx, yy, g, m, multiplier, b),
                                np.zeros(xx.shape[1]), gtol=1e-13, xtol=1e-13, ftol=1e-13,
                            )
                            error = float(max(abs(refit.x - beta)))
                            raw_u = test_x @ refit.x
                            independent_loss = float(np.mean(np.logaddexp(0, (1 - 2 * test_y) * raw_u)))
                            assert error < 1e-6 and abs(independent_loss - evaluate(test_x, test_y, beta)[0]) < 1e-8
                            metrics["largest_alternative_root_error"] = max(metrics["largest_alternative_root_error"], error)
                            metrics["alternative_roots"] += 1
                    else:
                        beta, status = all_soft, "outside_all_soft_fallback"
                    if name == "all_soft":
                        all_soft = beta
                    loss, brier = evaluate(test_x, test_y, beta)
                    if scheme == "row" and name in archived_names:
                        old = previous.loc[(*key, archived_names[name])]
                        assert (status == "finite_root") == (old.status == "finite_root")
                        if status == "finite_root":
                            error = max(abs(loss - old.logloss), abs(brier - old.brier))
                            assert error < 1e-8
                            metrics["max_archived_loss_error"] = max(metrics["max_archived_loss_error"], error)
                        metrics["archived_checks"] += 1
                    rows.append(dict(
                        trial=trial, label_rows=m, prediction=prediction, method=name,
                        multiplier=multiplier, endpoint=endpoint, status=status, logloss=loss,
                        brier=brier, ridge_logloss=ridge_record.logloss,
                    ))
                checks.append(dict(
                    trial=trial, label_rows=m, prediction=prediction, endpoint=endpoint,
                    protected=protected, primal_dual_gap=errors[0], dual_error=errors[1],
                    primal_error=errors[2], witness_error=witness_error, **fallback, **metrics,
                ))
        if (trial + 1) % 100 == 0:
            print(f"{scheme}: {trial + 1}/500 allocations", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    rows = pd.DataFrame(rows)
    checks = pd.DataFrame(checks)
    assert len(rows) == 12000 and len(checks) == 2000
    all_soft = rows.loc[rows.method.eq("all_soft"), ["trial", "label_rows", "prediction", "logloss"]]
    rows = rows.merge(all_soft.rename(columns={"logloss": "soft_logloss"}), validate="many_to_one")
    rows["diff_soft"] = rows.logloss - rows.soft_logloss
    rows["diff_ridge"] = rows.logloss - rows.ridge_logloss
    rows.to_csv(out / "policy_rows.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    checks.to_csv(out / "frontier_checks.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    groups = ["label_rows", "prediction", "method"]
    summary = rows.groupby(groups).agg(
        cases=("trial", "size"), exterior=("status", lambda s: int(s.ne("finite_root").sum())),
        mean_loss=("logloss", "mean"), median_loss=("logloss", "median"),
        p90_loss=("logloss", lambda v: v.quantile(.90)), mean_brier=("brier", "mean"),
        mean_vs_soft=("diff_soft", "mean"), mean_vs_ridge=("diff_ridge", "mean"),
        better_than_soft=("diff_soft", lambda v: int(v.lt(-1e-10).sum())),
        reduced_multiplier=("multiplier", lambda v: int(v.lt(1).sum())),
    ).reset_index()
    summary.to_csv(out / "summary.csv", index=False)
    save(out / "verification.json", dict(
        scheme=scheme, runner_sha256=sha(HERE / "evaluate_paths.py"), plan_sha256=sha(HERE / "plan.json"),
        allocations=len(order_hashes), configurations=len(rows), frontiers=len(checks),
        original_score_checks=int(checks.finite_scores_checked.sum()),
        archived_checks=int(checks.archived_checks.sum()), alternative_roots=int(checks.alternative_roots.sum()),
        maxima={column: float(checks[column].max()) for column in [
            "primal_dual_gap", "dual_error", "primal_error", "witness_error", "max_direct_score",
            "largest_alternative_root_error", "max_archived_loss_error",
        ]},
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scheme", choices=["row", "block"], required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.data, args.scheme, args.out)
