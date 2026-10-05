"""Monte Carlo benchmark for the JOCM revision.

The script keeps the simulation runnable in a minimal environment. It compares
an additive MNL with a structured MNL that receives prespecified nonlinear and
interaction terms. The ML-assisted selector uses a respondent-level random
forest diagnostic to rank candidate utility terms, followed by nested
behavioural refitting and validation.
"""
from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
except ImportError as exc:  # pragma: no cover - dependency is pinned for the release
    RandomForestClassifier = None
    _SKLEARN_IMPORT_ERROR = exc

N_ALTERNATIVES = 3
N_ATTRIBUTES = 5
OPT_OUT_INDEX = 2
PRICE_INDEX = 4


@dataclass(frozen=True)
class Condition:
    name: str
    nonlinear: bool
    threshold: bool
    interaction: bool
    heterogeneity: bool


CONDITIONS = [
    Condition("additive", False, False, False, False),
    Condition("nonlinear", True, False, False, False),
    Condition("threshold", False, True, False, False),
    Condition("interaction", False, False, True, False),
    Condition("heterogeneity", False, False, False, True),
    Condition("nonlinear_threshold", True, True, False, False),
    Condition("nonlinear_interaction", True, False, True, False),
    Condition("combined", True, True, True, True),
]


def softmax(u):
    z = u - u.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def make_data(seed, n=400, tasks=12, condition=None):
    rng = np.random.default_rng(seed)
    j = N_ALTERNATIVES
    # alternative attributes: quality, support, evidence, data, price
    x = rng.normal(size=(n, tasks, j, N_ATTRIBUTES))
    x[:, :, OPT_OUT_INDEX, :] = 0.0  # opt-out has no product attributes
    x[:, :, :, PRICE_INDEX] = np.abs(x[:, :, :, PRICE_INDEX]) + 0.5
    # Restore the opt-out row after the product-price transform. Otherwise the
    # global price floor would silently assign a product price to opt-out.
    x[:, :, OPT_OUT_INDEX, :] = 0.0
    z = rng.normal(size=(n, 2))  # observed income and digital experience
    z[:, 0] = (z[:, 0] > 0).astype(float)
    z[:, 1] = (z[:, 1] > 0).astype(float)
    base = np.array([0.55, 0.35, 0.45, 0.25, -0.85])
    v = np.einsum("ntjp,p->ntj", x, base)
    v[:, :, OPT_OUT_INDEX] += -0.35
    if condition.nonlinear:
        v += 0.55 * x[:, :, :, 0] ** 2 - 0.30 * x[:, :, :, PRICE_INDEX] ** 2
    if condition.threshold:
        v += -0.90 * np.maximum(x[:, :, :, PRICE_INDEX] - 1.25, 0.0)
    if condition.interaction:
        v += 0.75 * x[:, :, :, 0] * x[:, :, :, 1]
        v += 0.55 * x[:, :, :, 2] * z[:, None, None, 1]
    if condition.heterogeneity:
        # Unobserved random price sensitivity. This is deliberately distinct
        # from the observed covariates used by the interaction condition.
        random_price = rng.normal(0.0, 0.45, size=n)
        v += random_price[:, None, None] * x[:, :, :, PRICE_INDEX]
    v[:, :, OPT_OUT_INDEX] += 0.15 * z[:, None, 0]
    p = softmax(v.reshape(-1, j)).reshape(n, tasks, j)
    choices = np.array([rng.choice(j, p=pp) for pp in p.reshape(-1, j)]).reshape(n, tasks)
    return x, z, choices, v


def feature_matrix(x, z, structured=False):
    n, t, j, _ = x.shape
    if j != N_ALTERNATIVES:
        raise ValueError(f"expected {N_ALTERNATIVES} alternatives, found {j}")
    flat_x = x.reshape(n * t, j, N_ATTRIBUTES)
    flat_z = np.repeat(z, t, axis=0)
    optout = np.broadcast_to((np.arange(j) == OPT_OUT_INDEX).astype(float)[None, :, None], (n * t, j, 1))
    cols = [flat_x, optout]
    if structured:
        cols += [flat_x[:, :, 0:1] ** 2, flat_x[:, :, PRICE_INDEX:PRICE_INDEX + 1] ** 2]
        cols += [np.maximum(flat_x[:, :, PRICE_INDEX:PRICE_INDEX + 1] - 1.25, 0.0)]
        cols += [flat_x[:, :, 0:1] * flat_x[:, :, 1:2]]
        cols += [flat_x[:, :, 2:3] * flat_z[:, None, 1:2]]
        cols += [flat_x[:, :, PRICE_INDEX:PRICE_INDEX + 1] * flat_z[:, None, 0:1]]
    return np.concatenate(cols, axis=2)


def fit_mnl(X, y, steps=350, lr=0.08, l2=1e-3):
    p = X.shape[2]
    beta = np.zeros(p)
    for _ in range(steps):
        pr = softmax(np.einsum("tjp,p->tj", X, beta))
        target = np.zeros_like(pr)
        target[np.arange(len(y)), y] = 1.0
        grad = np.einsum("tj,tjp->p", pr - target, X) / len(y) + l2 * beta
        beta -= lr * grad
    return beta


def fit_latent_class(X4, y2, steps=100, lr=0.06, l2=1e-3, seed=0):
    """Two-class latent-class MNL fitted by a small EM-gradient routine."""
    rng = np.random.default_rng(seed)
    n, tasks, _, p = X4.shape
    k = 2
    beta = rng.normal(0, 0.02, size=(k, p))
    prior = np.full(k, 1.0 / k)
    for _ in range(steps):
        loglik = np.zeros((n, k))
        probs = np.zeros((n, tasks, k, N_ALTERNATIVES))
        for c in range(k):
            probs[:, :, c, :] = softmax(np.einsum("ntjp,p->ntj", X4, beta[c]).reshape(-1, N_ALTERNATIVES)).reshape(n, tasks, N_ALTERNATIVES)
            loglik[:, c] = np.log(np.clip(probs[:, :, c, :][np.arange(n)[:, None], np.arange(tasks)[None, :], y2], 1e-12, 1)).sum(1)
        z = loglik + np.log(np.clip(prior, 1e-12, 1))[None, :]
        z -= z.max(1, keepdims=True)
        resp = np.exp(z); resp /= resp.sum(1, keepdims=True)
        prior = resp.mean(0)
        for c in range(k):
            target = np.eye(N_ALTERNATIVES)[y2]
            err = probs[:, :, c, :] - target
            weighted = resp[:, c, None, None] * err
            grad = np.einsum("nt,ntj,ntjp->p", resp[:, c, None].repeat(tasks, 1), err, X4) / (resp[:, c].sum() * tasks + 1e-12)
            beta[c] -= lr * (grad + l2 * beta[c])
    return beta, prior


def score_latent_class(X4, y2, v4, beta, prior):
    n, tasks, _, _ = X4.shape
    probs = []
    for c in range(len(prior)):
        probs.append(softmax(np.einsum("ntjp,p->ntj", X4, beta[c]).reshape(-1, N_ALTERNATIVES)).reshape(n, tasks, N_ALTERNATIVES))
    pr = np.tensordot(np.asarray(prior), np.asarray(probs), axes=(0, 0))
    pred = pr.argmax(2)
    yflat = y2.reshape(-1); pflat = pr.reshape(-1, N_ALTERNATIVES); vflat = v4.reshape(-1, N_ALTERNATIVES)
    pred = pflat.argmax(1)
    acc = float((pred == yflat).mean())
    logloss = float(-np.log(np.clip(pflat[np.arange(len(yflat)), yflat], 1e-12, 1)).mean())
    brier = float(((pflat - np.eye(N_ALTERNATIVES)[yflat]) ** 2).sum(1).mean())
    shares = pflat.mean(0); observed = np.bincount(yflat, minlength=N_ALTERNATIVES) / len(yflat)
    share_rmse = float(np.sqrt(np.mean((shares - observed) ** 2)))
    regret = float(np.mean(vflat.max(1) - vflat[np.arange(len(yflat)), pred]))
    return acc, logloss, brier, share_rmse, regret


def score(X, y, v_true, beta):
    pr = softmax(np.einsum("tjp,p->tj", X, beta))
    pred = pr.argmax(1)
    acc = float((pred == y).mean())
    logloss = float(-np.log(np.clip(pr[np.arange(len(y)), y], 1e-12, 1)).mean())
    brier = float(((pr - np.eye(N_ALTERNATIVES)[y]) ** 2).sum(1).mean())
    shares = pr.mean(0)
    observed = np.bincount(y, minlength=N_ALTERNATIVES) / len(y)
    share_rmse = float(np.sqrt(np.mean((shares - observed) ** 2)))
    regret = float(np.mean(v_true.max(1) - v_true[np.arange(len(y)), pred]))
    return acc, logloss, brier, share_rmse, regret


def choose_candidate_terms(xx, choices, train_ids, tasks, seed):
    """Use RF diagnostics, then nested behavioural forward selection.

    The forest only sees the inner respondent split. Candidate terms are ranked
    by grouped alternative-specific feature importance. The behavioural MNL is
    then refit on the inner split and terms are retained only when the inner
    validation loss improves. This keeps discovery and behavioural estimation
    separate while making the selector genuinely ML-assisted.
    """
    if RandomForestClassifier is None:
        raise ImportError("scikit-learn is required for ML-assisted selection") from _SKLEARN_IMPORT_ERROR
    rng = np.random.default_rng(seed)
    shuffled = np.array(train_ids, copy=True)
    rng.shuffle(shuffled)
    cut = max(1, int(0.75 * len(shuffled)))
    inner, valid = shuffled[:cut], shuffled[cut:]
    groups = [[N_ATTRIBUTES + 1 + i] for i in range(6)]
    names = ["quality_sq", "price_sq", "price_hinge", "quality_support", "evidence_digital", "price_income"]
    selected = list(range(N_ATTRIBUTES + 1))
    # Fit the diagnostic learner only on inner respondents and aggregate
    # importance across alternatives for each candidate utility term.
    flat = xx.reshape(xx.shape[0], -1)
    inner_rows = np.concatenate([inner * tasks + q for q in range(tasks)])
    forest = RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=seed,
        n_jobs=1,
    )
    forest.fit(flat[inner_rows], choices[inner].reshape(-1))
    importance = forest.feature_importances_.reshape(N_ALTERNATIVES, xx.shape[2]).sum(axis=0)
    ranked = sorted(range(len(groups)), key=lambda gi: float(importance[groups[gi]].sum()), reverse=True)
    # The diagnostic stage defines a short-list. Behavioural validation then
    # chooses among that shortlist, so the RF ranking has a material role in
    # the final specification instead of serving as unused metadata.
    remaining = ranked[:5]

    def stack(ids, cols):
        mat = xx[:, :, cols]
        tr = np.concatenate([mat[ids * tasks + q] for q in range(tasks)])
        yy = np.concatenate([choices[ids, q] for q in range(tasks)])
        return tr, yy

    tr0, y0 = stack(inner, selected)
    beta0 = fit_mnl(tr0, y0)
    va0, yv = stack(valid, selected)
    best_loss = score(va0, yv, np.zeros((len(yv), N_ALTERNATIVES)), beta0)[1]
    while remaining:
        trials = []
        for gi in remaining:
            cols = selected + groups[gi]
            tr, yy = stack(inner, cols)
            b = fit_mnl(tr, yy, steps=220)
            va, _ = stack(valid, cols)
            loss = score(va, yv, np.zeros((len(yv), N_ALTERNATIVES)), b)[1]
            trials.append((loss, gi))
        loss, gi = min(trials)
        if loss + 1e-4 < best_loss:
            selected += groups[gi]
            remaining.remove(gi)
            best_loss = loss
        else:
            break
    return selected, "+".join(["base"] + [names[i] for i, g in enumerate(groups) if all(c in selected for c in g)])


def run(reps=24, n=400, tasks=12, out="results/simulation_results.csv"):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = []
    for ci, cond in enumerate(CONDITIONS):
        for rep in range(reps):
            seed = 20261004 + ci * 1000 + rep
            x, z, choices, v = make_data(seed, n, tasks, cond)
            idx = np.arange(n)
            rng = np.random.default_rng(seed + 77)
            rng.shuffle(idx)
            cut = int(0.8 * n)
            train_ids, test_ids = idx[:cut], idx[cut:]
            for model, structured in [("MNL", False), ("Structured_MNL", True), ("ML_assisted_spec", True), ("Latent_Class_MNL", False)]:
                xx = feature_matrix(x, z, structured)
                selected = list(range(xx.shape[2]))
                selected_terms = "base+all_candidates" if model == "Structured_MNL" else "base"
                if model == "ML_assisted_spec":
                    selected, selected_terms = choose_candidate_terms(xx, choices, train_ids, tasks, seed + 101)
                xx = xx[:, :, selected]
                if model == "Latent_Class_MNL":
                    n_features = xx.shape[2]
                    x4 = xx.reshape(n, tasks, N_ALTERNATIVES, n_features)
                    beta_lc, prior_lc = fit_latent_class(x4[train_ids], choices[train_ids], seed=seed + 303)
                    vals = score_latent_class(x4[test_ids], choices[test_ids], v[test_ids], beta_lc, prior_lc)
                    rows.append({"condition": cond.name, "replication": rep, "model": model,
                                 "accuracy": vals[0], "logloss": vals[1], "brier": vals[2],
                                 "share_rmse": vals[3], "decision_regret": vals[4],
                                 "selected_terms": "base"})
                    continue
                train = np.concatenate([xx[train_ids * tasks + q] for q in range(tasks)])
                ytrain = np.concatenate([choices[train_ids, q] for q in range(tasks)])
                test = np.concatenate([xx[test_ids * tasks + q] for q in range(tasks)])
                ytest = np.concatenate([choices[test_ids, q] for q in range(tasks)])
                vtest = np.concatenate([v[test_ids, q] for q in range(tasks)])
                beta = fit_mnl(train, ytrain)
                vals = score(test, ytest, vtest, beta)
                rows.append({"condition": cond.name, "replication": rep, "model": model,
                             "accuracy": vals[0], "logloss": vals[1], "brier": vals[2],
                             "share_rmse": vals[3], "decision_regret": vals[4],
                             "selected_terms": selected_terms})
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=24)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--tasks", type=int, default=12)
    ap.add_argument("--out", default="results/simulation_results.csv")
    args = ap.parse_args()
    rows = run(args.reps, args.n, args.tasks, args.out)
    print(f"wrote {len(rows)} model-replication rows to {args.out}")
