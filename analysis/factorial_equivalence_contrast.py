"""Monte Carlo benchmark for a four-cell mixed-difference extension.

The design keeps an additive candidate's utility differences invariant across
four randomized tasks: base, time-shift, cost-shift, and joint-shift.  The
mixed contrast (joint - time - cost + base) cancels additive main effects and
isolates a cross-attribute interaction.  It is a refinement of the same
model-equivalent-task object, not a second model-selection procedure.
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
    if term == "interaction":
        extra = (x[:, :, 0:1] / 100.0) * (x[:, :, 1:2] / 100.0)
    else:
        raise ValueError(f"unknown candidate term: {term}")
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


def multiplier_pvalue(contrasts, clusters, reps, seed):
    unique = np.unique(clusters)
    sums = np.stack([contrasts[clusters == rid].sum(0) for rid in unique])
    counts = np.asarray([(clusters == rid).sum() for rid in unique], dtype=float)
    means = sums / counts[:, None]
    observed_mean = means.mean(0)
    centered = means - observed_mean
    cov = centered.T @ centered / max(len(unique) * (len(unique) - 1), 1)
    cov += np.eye(3) * 1e-10
    observed = len(unique) * (observed_mean @ np.linalg.pinv(cov) @ observed_mean)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(reps):
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(unique))
        mean = (centered * signs[:, None]).mean(0)
        draws.append(len(unique) * (mean @ np.linalg.pinv(cov) @ mean))
    return float((1.0 + np.sum(np.asarray(draws) >= observed)) / (reps + 1.0))


def simulate(seed, respondents=300, nuisance_tasks=4, condition="additive"):
    rng = np.random.default_rng(seed)
    # Keep the four alternatives in a useful probability range so the mixed
    # difference has power without relying on near-deterministic choices.
    base_beta = np.array([-0.20, -2.00, -0.25, -0.20])
    cells = {
        0: np.array([0.0, 0.0]),
        1: np.array([40.0, 0.0]),
        2: np.array([0.0, 100.0]),
        3: np.array([40.0, 100.0]),
    }
    xs, ys, ids, cell_ids = [], [], [], []
    for rid in range(respondents):
        random_cost = rng.normal(0.0, 1.0) if condition == "random_price" else 0.0
        base = np.array([[50.0, 40.0], [70.0, 50.0], [90.0, 60.0]])
        base = base + rng.normal(0.0, 2.0, size=(3, 2))
        tasks = [(cell, base + shift[None, :]) for cell, shift in cells.items()]
        tasks += [(-1, rng.uniform(40.0, 120.0, size=(3, 2))) for _ in range(nuisance_tasks)]
        rng.shuffle(tasks)
        for cell, xx in tasks:
            utility = np.array([base_beta[0], 0.0, base_beta[1]]) + base_beta[2:] @ (xx.T / 100.0)
            if condition in ("nonlinear", "interaction_nonlinear"):
                utility += 6.0 * (xx[:, 0] / 100.0) ** 2
            elif condition == "threshold":
                utility += -10.0 * np.maximum(xx[:, 1] / 100.0 - 0.85, 0.0)
            if condition in ("interaction", "interaction_nonlinear"):
                utility += 8.0 * (xx[:, 0] / 100.0) * (xx[:, 1] / 100.0)
            elif condition == "random_price":
                utility += random_cost * (xx[:, 1] / 100.0)
            p = softmax(utility[None, :])[0]
            xs.append(xx); ys.append(int(rng.choice(3, p=p))); ids.append(rid); cell_ids.append(cell)
    return np.asarray(ids), np.asarray(ys), np.asarray(xs), np.asarray(cell_ids)


def evaluate(ids, y, x, cell_ids, term, bootstrap, seed):
    unique = np.unique(ids)
    rng = np.random.default_rng(seed)
    shuffled = unique.copy(); rng.shuffle(shuffled)
    contrasts, clusters = [], []
    for test_ids in np.array_split(shuffled, 2):
        test_mask = np.isin(ids, test_ids)
        train_mask = ~test_mask
        beta = fit_mnl(design(x[train_mask], term), y[train_mask])
        probs = softmax(np.einsum("njp,p->nj", design(x[test_mask], term), beta))
        test_rows = np.flatnonzero(test_mask)
        local = {global_row: i for i, global_row in enumerate(test_rows)}
        by_person = {}
        for global_row in test_rows:
            rid = int(ids[global_row])
            cell = int(cell_ids[global_row])
            if cell in (0, 1, 2, 3):
                by_person.setdefault(rid, {})[cell] = global_row
        for rid, cells in by_person.items():
            if set(cells) != {0, 1, 2, 3}:
                continue
            residual = {}
            for cell, row in cells.items():
                residual[cell] = np.eye(3)[y[row]] - probs[local[row]]
            contrasts.append(residual[3] - residual[1] - residual[2] + residual[0])
            clusters.append(rid)
    contrasts = np.asarray(contrasts); clusters = np.asarray(clusters)
    return multiplier_pvalue(contrasts, clusters, bootstrap, seed + 7), len(contrasts)


def run(reps=100, bootstrap=199, respondents=300,
        out="results/factorial_equivalence_contrast.csv"):
    rows = []
    conditions = ("additive", "nonlinear", "threshold", "interaction", "interaction_nonlinear", "random_price")
    for condition in conditions:
        for rep in range(reps):
            data = simulate(20261008 + rep, respondents=respondents, condition=condition)
            oracle_term = "interaction" if condition == "interaction" else None
            candidate_terms = [None] if oracle_term is None else [None, oracle_term]
            for term in candidate_terms:
                pvalue, units = evaluate(*data, term=term, bootstrap=bootstrap, seed=20500000 + rep)
                rows.append({"condition": condition, "rep": rep,
                             "candidate": "oracle_interaction" if term else "additive",
                             "units": units, "pvalue": pvalue, "reject": int(pvalue < 0.05)})
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    for condition in conditions:
        for candidate in ("additive", "oracle_interaction"):
            subset = [r for r in rows if r["condition"] == condition and r["candidate"] == candidate]
            if subset:
                print(condition, candidate, "rejection_rate", round(float(np.mean([r["reject"] for r in subset])), 3))
    print(f"wrote {len(rows)} factorial contrast rows to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--bootstrap", type=int, default=199)
    ap.add_argument("--respondents", type=int, default=300)
    ap.add_argument("--out", default="results/factorial_equivalence_contrast.csv")
    run(**vars(ap.parse_args()))
