"""Recoverability-aware model-family triage for the smart-device benchmark.

The rule separates two signals on a respondent-level development split:

* ``flex_gain``: out-of-fold log-loss gain of a flexible predictor over the
  additive MNL, which detects unexplained observable structure or any other
  predictive departure;
* ``cluster_score``: respondent-level score overdispersion after the additive
  MNL, which detects persistent individual taste variation.

Thresholds are calibrated once on independent additive datasets and frozen
before the mechanism conditions are evaluated.  The selected family is then
refit on development respondents and evaluated on a third respondent holdout.
This makes the proposed object a mechanism triage rule rather than another
in-sample model ranking.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.dirname(__file__))
from run_simulation import (  # noqa: E402
    CONDITIONS,
    N_ALTERNATIVES,
    feature_matrix,
    fit_latent_class,
    fit_mnl,
    make_data,
    score,
)


@dataclass(frozen=True)
class Thresholds:
    structured_gain: float
    flex_gain: float
    cluster_score: float


def respondent_split(n: int, seed: int):
    ids = np.arange(n)
    np.random.default_rng(seed).shuffle(ids)
    n_train = int(0.60 * n)
    n_val = int(0.20 * n)
    return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]


def row_indices(ids: np.ndarray, tasks: int) -> np.ndarray:
    # ``choices[ids].reshape(-1)`` is respondent-major, so preserve the same
    # order when selecting flattened design rows.
    return (ids[:, None] * tasks + np.arange(tasks)[None, :]).reshape(-1)


def mnl_probs(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    utility = np.einsum("tjp,p->tj", X, beta)
    utility -= utility.max(axis=1, keepdims=True)
    exp_u = np.exp(utility)
    return exp_u / exp_u.sum(axis=1, keepdims=True)


def logloss_from_probs(prob: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.clip(prob[np.arange(len(y)), y], 1e-12, 1.0)).mean())


def cluster_score_overdispersion(X: np.ndarray, y: np.ndarray, beta: np.ndarray,
                                 n_resp: int, tasks: int) -> float:
    """Score overdispersion on a respondent-level validation block."""
    p = mnl_probs(X, beta).reshape(n_resp, tasks, N_ALTERNATIVES)
    x = X.reshape(n_resp, tasks, N_ALTERNATIVES, X.shape[-1])
    price = x[:, :, :, 4]
    chosen = price[np.arange(n_resp)[:, None], np.arange(tasks)[None, :], y.reshape(n_resp, tasks)]
    expected = (p * price).sum(axis=2)
    scores = (chosen - expected).sum(axis=1)
    information = ((p * price ** 2).sum(axis=2) - expected ** 2).sum(axis=1)
    return float(np.sum(scores ** 2) / np.clip(np.sum(information), 1e-12, None))


def fit_flexible(x: np.ndarray, z: np.ndarray, ids: np.ndarray, choices: np.ndarray,
                 tasks: int, seed: int):
    raw = np.concatenate([x.reshape(len(x) * tasks, -1),
                          np.repeat(z, tasks, axis=0)], axis=1)
    rows = row_indices(ids, tasks)
    model = HistGradientBoostingClassifier(
        max_iter=150, max_leaf_nodes=15, learning_rate=0.05,
        l2_regularization=1e-3, random_state=seed,
    )
    model.fit(raw[rows], choices[ids].reshape(-1))
    return model, raw


def get_diagnostics(x, z, choices, train_ids, val_ids, tasks, seed):
    """Fit only on train respondents and return development diagnostics."""
    base_features = feature_matrix(x, z, structured=False)
    train_rows = row_indices(train_ids, tasks)
    val_rows = row_indices(val_ids, tasks)
    y_train = choices[train_ids].reshape(-1)
    y_val = choices[val_ids].reshape(-1)
    beta = fit_mnl(base_features[train_rows], y_train)
    base_prob = mnl_probs(base_features[val_rows], beta)
    base_loss = logloss_from_probs(base_prob, y_val)
    structured_features = feature_matrix(x, z, structured=True)
    structured_beta = fit_mnl(structured_features[train_rows], y_train)
    structured_prob = mnl_probs(structured_features[val_rows], structured_beta)
    structured_loss = logloss_from_probs(structured_prob, y_val)
    flexible, raw = fit_flexible(x, z, train_ids, choices, tasks, seed)
    flex_prob = np.zeros((len(val_rows), N_ALTERNATIVES))
    pred = flexible.predict_proba(raw[val_rows])
    flex_prob[:, flexible.classes_.astype(int)] = pred
    flex_loss = logloss_from_probs(flex_prob, y_val)
    q = cluster_score_overdispersion(
        base_features[val_rows], y_val, beta, len(val_ids), tasks,
    )
    return {"structured_gain": base_loss - structured_loss,
            "flex_gain": base_loss - flex_loss, "cluster_score": q,
            "base_loss": base_loss, "structured_loss": structured_loss,
            "flex_loss": flex_loss}


def calibrate(reps: int = 100, n: int = 400, tasks: int = 12,
              seed: int = 20261030) -> tuple[Thresholds, list[dict]]:
    diagnostics = []
    for rep in range(reps):
        x, z, choices, _ = make_data(seed + rep, n, tasks, CONDITIONS[0])
        tr, va, _ = respondent_split(n, seed + 10000 + rep)
        d = get_diagnostics(x, z, choices, tr, va, tasks, seed + 20000 + rep)
        d["replication"] = rep
        diagnostics.append(d)
    gains = np.asarray([d["flex_gain"] for d in diagnostics])
    structured = np.asarray([d["structured_gain"] for d in diagnostics])
    scores = np.asarray([d["cluster_score"] for d in diagnostics])
    # A one-sided 95% calibration controls the additive-condition false
    # expansion rate at the calibration resolution; these thresholds are
    # frozen for all mechanism conditions.
    thresholds = Thresholds(
        structured_gain=float(np.quantile(structured, 0.95)),
        flex_gain=float(np.quantile(gains, 0.95)),
        cluster_score=float(np.quantile(scores, 0.95)),
    )
    return thresholds, diagnostics


def classify(diagnostic: dict, thresholds: Thresholds) -> tuple[str, str]:
    structured = diagnostic["structured_gain"] > thresholds.structured_gain
    flex = diagnostic["flex_gain"] > thresholds.flex_gain
    het = diagnostic["cluster_score"] > thresholds.cluster_score
    if (structured or flex) and het:
        return "unresolved", "both_signals"
    if het:
        return "heterogeneity", "cluster_signal"
    if structured:
        return "observed_structure", "flexible_gain"
    if flex:
        return "unresolved", "flexible_gain_without_candidate"
    return "base", "no_signal"


def target_action(condition) -> str:
    """Mechanism label used only for the simulation audit, never in fitting."""
    has_observable = condition.nonlinear or condition.threshold or condition.interaction
    if condition.heterogeneity and has_observable:
        return "unresolved"
    if condition.heterogeneity:
        return "heterogeneity"
    if has_observable:
        return "observed_structure"
    return "base"


def fit_and_score_family(family: str, x, z, choices, v, train_ids, test_ids, tasks, seed):
    base = feature_matrix(x, z, structured=False)
    structured = feature_matrix(x, z, structured=True)
    tr = row_indices(train_ids, tasks); te = row_indices(test_ids, tasks)
    ytr = choices[train_ids].reshape(-1); yte = choices[test_ids].reshape(-1)
    vte = v[test_ids].reshape(-1, N_ALTERNATIVES)
    if family == "base":
        beta = fit_mnl(base[tr], ytr)
        pr = mnl_probs(base[te], beta)
        return (*score(base[te], yte, vte, beta), beta)
    if family == "observed_structure":
        beta = fit_mnl(structured[tr], ytr)
        return (*score(structured[te], yte, vte, beta), beta)
    if family == "heterogeneity":
        x4 = base.reshape(len(x), tasks, N_ALTERNATIVES, base.shape[-1])
        beta_lc, prior = fit_latent_class(x4[train_ids], choices[train_ids], seed=seed)
        probs = []
        for b in beta_lc:
            probs.append(mnl_probs(base[te], b).reshape(len(test_ids), tasks, N_ALTERNATIVES))
        pr = np.tensordot(np.asarray(prior), np.asarray(probs), axes=(0, 0)).reshape(-1, N_ALTERNATIVES)
        pred = pr.argmax(1)
        return (float((pred == yte).mean()), logloss_from_probs(pr, yte),
                float(((pr - np.eye(N_ALTERNATIVES)[yte]) ** 2).sum(1).mean()),
                float(np.sqrt(np.mean((pr.mean(0) - np.bincount(yte, minlength=N_ALTERNATIVES) / len(yte)) ** 2))),
                float(np.mean(vte.max(1) - vte[np.arange(len(yte)), pred])), beta_lc)
    if family == "unresolved":
        model, raw = fit_flexible(x, z, train_ids, choices, tasks, seed)
        p = np.zeros((len(te), N_ALTERNATIVES)); pred_p = model.predict_proba(raw[te])
        p[:, model.classes_.astype(int)] = pred_p
        pred = p.argmax(1)
        return (float((pred == yte).mean()), logloss_from_probs(p, yte),
                float(((p - np.eye(N_ALTERNATIVES)[yte]) ** 2).sum(1).mean()),
                float(np.sqrt(np.mean((p.mean(0) - np.bincount(yte, minlength=N_ALTERNATIVES) / len(yte)) ** 2))),
                float(np.mean(vte.max(1) - vte[np.arange(len(yte)), pred])), None)
    raise ValueError(family)


def run(reps: int = 30, calibration_reps: int = 100, n: int = 400,
        tasks: int = 12, out: str = "results/recoverability_triage.csv",
        calibration_out: str = "results/recoverability_triage_calibration.csv"):
    thresholds, calibration = calibrate(calibration_reps, n, tasks)
    rows = []
    for d in calibration:
        rows.append({"stage": "calibration", "condition": "additive",
                     "replication": d["replication"], **d,
                     "threshold_flex_gain": thresholds.flex_gain,
                     "threshold_cluster_score": thresholds.cluster_score,
                     "action": "calibration", "reason": "calibration"})
    for ci, condition in enumerate(CONDITIONS):
        for rep in range(reps):
            seed = 20400000 + ci * 1000 + rep
            x, z, choices, v = make_data(seed, n, tasks, condition)
            tr, va, te = respondent_split(n, seed + 77)
            d = get_diagnostics(x, z, choices, tr, va, tasks, seed + 101)
            action, reason = classify(d, thresholds)
            development_ids = np.concatenate([tr, va])
            metrics_cache = {}
            for family in ("base", "observed_structure", "unresolved"):
                metrics_cache[family] = fit_and_score_family(
                    family, x, z, choices, v, development_ids, te, tasks,
                    seed + 202 + len(metrics_cache),
                )
            if action == "heterogeneity":
                metrics_cache[action] = fit_and_score_family(
                    action, x, z, choices, v, development_ids, te, tasks,
                    seed + 205,
                )
            # A conventional validation-only selector is a useful comparator:
            # it chooses the best held-out log loss among base, structured MNL,
            # and the flexible predictor, without the mechanism diagnostics.
            validation_losses = {
                "base": d["base_loss"],
                "observed_structure": d["structured_loss"],
                "unresolved": d["flex_loss"],
            }
            validation_action = min(validation_losses, key=validation_losses.get)
            metrics = metrics_cache[action]
            validation_metrics = metrics_cache[validation_action]
            rows.append({"stage": "evaluation", "condition": condition.name,
                         "replication": rep, **d,
                         "threshold_flex_gain": thresholds.flex_gain,
                         "threshold_cluster_score": thresholds.cluster_score,
                         "action": action, "reason": reason,
                         "target_action": target_action(condition),
                         "validation_action": validation_action,
                         "test_accuracy": metrics[0], "test_logloss": metrics[1],
                         "test_brier": metrics[2], "test_share_rmse": metrics[3],
                         "test_decision_regret": metrics[4],
                         "validation_test_logloss": validation_metrics[1],
                         "validation_test_decision_regret": validation_metrics[4],
                         "base_test_logloss": metrics_cache["base"][1],
                         "base_test_decision_regret": metrics_cache["base"][4],
                         "structured_test_logloss": metrics_cache["observed_structure"][1],
                         "structured_test_decision_regret": metrics_cache["observed_structure"][4]})
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    with open(calibration_out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(calibration[0].keys())); writer.writeheader(); writer.writerows(calibration)
    print("thresholds", asdict(thresholds))
    eval_rows = [r for r in rows if r["stage"] == "evaluation"]
    for condition in [c.name for c in CONDITIONS]:
        subset = [r for r in eval_rows if r["condition"] == condition]
        actions = {a: sum(r["action"] == a for r in subset) / len(subset)
                   for a in ("base", "observed_structure", "heterogeneity", "unresolved")}
        action_accuracy = np.mean([r["action"] == r["target_action"] for r in subset])
        triage_regret = np.mean([float(r["test_decision_regret"]) for r in subset])
        validation_regret = np.mean([float(r["validation_test_decision_regret"]) for r in subset])
        base_regret = np.mean([float(r["base_test_decision_regret"]) for r in subset])
        print(condition, actions, "action_accuracy", round(float(action_accuracy), 3),
              "mean_regret", round(float(triage_regret), 4),
              "validation_regret", round(float(validation_regret), 4),
              "base_regret", round(float(base_regret), 4))
    print(f"wrote {len(rows)} rows to {out}")
    return thresholds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--calibration-reps", type=int, default=100)
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--tasks", type=int, default=12)
    parser.add_argument("--out", default="results/recoverability_triage.csv")
    parser.add_argument("--calibration-out", default="results/recoverability_triage_calibration.csv")
    args = parser.parse_args()
    run(**vars(args))
