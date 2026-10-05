"""Monte Carlo benchmark for the exact randomized model-equivalent test.

Each respondent sees pairs of tasks that preserve the linear candidate's
utility differences while shifting all alternatives along one or more raw
attribute dimensions. The pair labels are assigned by the experiment, and task
order is randomised. A candidate is fit on development respondents and frozen
before held-out pair contrasts are evaluated. This is the intended method
prototype; it is separate from the weaker observational Swissmetro fallback.
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np


def softmax(u):
    z = u - np.max(u, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def design(x, term=None):
    out = np.zeros((len(x), 3, 4), dtype=float)
    out[:, 0, 0] = 1.0
    out[:, 2, 1] = 1.0
    out[:, :, 2:4] = x / 100.0
    if term is None:
        return out
    terms = {
        "nonlinear": (x[:, :, 0:1] / 100.0) ** 2,
        "threshold": np.maximum(x[:, :, 1:2] / 100.0 - 0.85, 0.0),
        "interaction": (x[:, :, 0:1] / 100.0) * (x[:, :, 1:2] / 100.0),
    }
    if term not in terms:
        raise ValueError(f"unknown candidate term: {term}")
    extra = terms[term]
    return np.concatenate([out, extra], axis=2)


def fit_mnl(X, y, steps=80, l2=1e-5):
    """Fit MNL by damped Newton updates with a monotone loss check."""
    beta = np.zeros(X.shape[2])
    target = np.eye(3)[y]

    def loss(b):
        p = softmax(np.einsum("njp,p->nj", X, b))
        return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))) + 0.5 * l2 * (b @ b))

    current = loss(beta)
    for _ in range(steps):
        probs = softmax(np.einsum("njp,p->nj", X, beta))
        grad = np.einsum("nj,njp->p", probs - target, X) / len(y) + l2 * beta
        info = np.eye(X.shape[2]) * l2
        for n in range(len(y)):
            w = np.diag(probs[n]) - np.outer(probs[n], probs[n])
            info += X[n].T @ w @ X[n] / len(y)
        try:
            step = np.linalg.solve(info, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(info) @ grad
        accepted = False
        for damping in (1.0, 0.5, 0.25, 0.1, 0.05):
            candidate = beta - damping * step
            value = loss(candidate)
            if value <= current + 1e-10:
                beta, current = candidate, value
                accepted = True
                break
        if not accepted or np.linalg.norm(step) < 1e-7:
            break
    return beta


def cluster_stat(contrasts, clusters):
    unique = np.unique(clusters)
    sums = np.stack([contrasts[clusters == rid].sum(0) for rid in unique])
    counts = np.asarray([(clusters == rid).sum() for rid in unique], dtype=float)
    means = sums / counts[:, None]
    mean = means.mean(0)
    centered = means - mean
    cov = centered.T @ centered / max(len(unique) * (len(unique) - 1), 1)
    cov += np.eye(3) * 1e-10
    stat = len(unique) * (mean @ np.linalg.pinv(cov) @ mean)
    return float(stat), mean, cov


def multiplier_pvalue(contrasts, clusters, reps, seed):
    observed, _, cov = cluster_stat(contrasts, clusters)
    unique = np.unique(clusters)
    sums = np.stack([contrasts[clusters == rid].sum(0) for rid in unique])
    counts = np.asarray([(clusters == rid).sum() for rid in unique], dtype=float)
    means = sums / counts[:, None]
    centered = means - means.mean(0)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(reps):
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(unique))
        mean = (centered * signs[:, None]).mean(0)
        draws.append(len(unique) * (mean @ np.linalg.pinv(cov) @ mean))
    return float((1.0 + np.sum(np.asarray(draws) >= observed)) / (reps + 1.0))


def simulate(seed, respondents=300, nuisance_tasks=4, condition="additive"):
    rng = np.random.default_rng(seed)
    base_beta = np.array([-0.35, -0.18, -1.15, -0.95])
    # Each shift is common to all alternatives, so the additive candidate's
    # pairwise utility differences remain exactly equal.
    shift_specs = {
        "time_shift": np.array([20.0, 0.0]),
        "cost_shift": np.array([0.0, 60.0]),
        "joint_shift": np.array([20.0, 50.0]),
    }
    xs, ys, ids, pair_ids, pair_side, pair_type = [], [], [], [], [], []
    for rid in range(respondents):
        random_cost = rng.normal(0.0, 1.0) if condition == "random_price" else 0.0
        tasks = []
        for typ, shift in shift_specs.items():
            # A common focal profile keeps the direction of each omitted-term
            # contrast stable across respondents. Small noise changes the
            # tasks without allowing the effect to cancel in the aggregate.
            base = np.array([[50.0, 40.0], [70.0, 50.0], [90.0, 60.0]])
            base = base + rng.normal(0.0, 2.0, size=(3, 2))
            tasks.extend([(typ, 0, base), (typ, 1, base + shift[None, :])])
        for _ in range(nuisance_tasks):
            tasks.append(("nuisance", -1, rng.uniform(40.0, 120.0, size=(3, 2))))
        rng.shuffle(tasks)
        local_pairs = {}
        for xx_type, side, xx in tasks:
            utility = np.array([base_beta[0], 0.0, base_beta[1]]) + base_beta[2:] @ (xx.T / 100.0)
            if condition == "nonlinear":
                utility += 6.0 * (xx[:, 0] / 100.0) ** 2
            elif condition == "threshold":
                utility += -10.0 * np.maximum(xx[:, 1] / 100.0 - 0.85, 0.0)
            elif condition == "interaction":
                utility += 8.0 * (xx[:, 0] / 100.0) * (xx[:, 1] / 100.0)
            elif condition == "random_price":
                utility += random_cost * (xx[:, 1] / 100.0)
            p = softmax(utility[None, :])[0]
            row = len(xs)
            xs.append(xx); ys.append(int(rng.choice(3, p=p))); ids.append(rid)
            if side >= 0:
                local_pairs.setdefault(xx_type, {})[side] = row
            pair_ids.append(-1); pair_side.append(-1); pair_type.append("nuisance")
        # Assign pair metadata after the random order is known.
        for type_index, typ in enumerate(shift_specs):
            left = local_pairs[typ][0]; right = local_pairs[typ][1]
            pid = rid * len(shift_specs) + type_index
            pair_ids[left] = pid; pair_ids[right] = pid
            pair_side[left] = 0; pair_side[right] = 1
            pair_type[left] = typ; pair_type[right] = typ
    return (np.asarray(ids), np.asarray(ys), np.asarray(xs),
            np.asarray(pair_ids), np.asarray(pair_side), np.asarray(pair_type, dtype=object))


def evaluate(ids, y, x, pair_ids, pair_side, pair_type, term, bootstrap, seed):
    unique = np.unique(ids)
    rng = np.random.default_rng(seed)
    shuffled = unique.copy(); rng.shuffle(shuffled)
    folds = np.array_split(shuffled, 2)
    contrasts = []; clusters = []; types = []
    for test_ids in folds:
        test_mask = np.isin(ids, test_ids)
        train_mask = ~test_mask
        beta = fit_mnl(design(x[train_mask], term), y[train_mask])
        probs = softmax(np.einsum("njp,p->nj", design(x[test_mask], term), beta))
        test_rows = np.flatnonzero(test_mask)
        local = {global_row: i for i, global_row in enumerate(test_rows)}
        by_pair = {}
        for global_row in test_rows:
            pid = int(pair_ids[global_row])
            if pid >= 0:
                by_pair.setdefault(pid, {})[int(pair_side[global_row])] = global_row
        for pid, sides in by_pair.items():
            if 0 not in sides or 1 not in sides:
                continue
            left, right = sides[0], sides[1]
            observed = np.eye(3)[y[left]] - np.eye(3)[y[right]]
            pl = probs[local[left]]; pr = probs[local[right]]
            contrasts.append(observed - (pl - pr)); clusters.append(int(ids[left]))
            types.append(pair_type[left])
    contrasts = np.asarray(contrasts); clusters = np.asarray(clusters); types = np.asarray(types)
    pvalue = multiplier_pvalue(contrasts, clusters, bootstrap, seed + 7)
    records = [{"candidate": "oracle_term" if term else "additive",
                "pair_type": "all", "pairs": len(contrasts),
                "pvalue": pvalue, "reject": int(pvalue < 0.05)}]
    for typ in sorted(set(types.tolist())):
        take = types == typ
        if take.sum() < 10:
            continue
        p = multiplier_pvalue(contrasts[take], clusters[take], bootstrap, seed + 17 + len(typ))
        records.append({"candidate": "oracle_term" if term else "additive",
                        "pair_type": typ, "pairs": int(take.sum()),
                        "pvalue": p, "reject": int(p < 0.05)})
    return records


def run(reps=100, bootstrap=199, respondents=300,
        out="results/exact_paired_task_benchmark.csv"):
    rows = []
    conditions = ("additive", "nonlinear", "threshold", "interaction", "random_price")
    for condition in conditions:
        for rep in range(reps):
            data = simulate(20261007 + rep, respondents=respondents, condition=condition)
            oracle_term = {"nonlinear": "nonlinear", "threshold": "threshold",
                           "interaction": "interaction"}.get(condition)
            candidate_terms = [None] if oracle_term is None else [None, oracle_term]
            for term in candidate_terms:
                records = evaluate(*data, term=term, bootstrap=bootstrap,
                                    seed=20400000 + rep)
                for record in records:
                    rows.append({"condition": condition, "rep": rep, **record})
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    for condition in conditions:
        for candidate in ("additive", "oracle_term"):
            subset = [r for r in rows if r["condition"] == condition and r["candidate"] == candidate and r["pair_type"] == "all"]
            if subset:
                print(condition, candidate, "rejection_rate", round(float(np.mean([r["reject"] for r in subset])), 3))
    print(f"wrote {len(rows)} exact paired-task rows to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--bootstrap", type=int, default=199)
    ap.add_argument("--respondents", type=int, default=300)
    ap.add_argument("--out", default="results/exact_paired_task_benchmark.csv")
    args = ap.parse_args()
    run(**vars(args))
