"""Compare fibre-aware inference with established lack-of-fit baselines.

The benchmark is deliberately finite and randomized.  It compares the
support-complete score with (i) a fibre-stratified CMH score for a predeclared
binary raw contrast, (ii) the saturated complete-fibre likelihood ratio, and
(iii) a cross-fitted random-forest residual score.  The baselines are
inferential comparators; the support-complete rank remains the pre-outcome
design certificate.
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from support_complete_fibre_benchmark import generate, score_test, patterns
from support_complete_fibre_design import candidate_values, profiles_3x3
from support_complete_vs_saturated_benchmark import saturated_lr_pvalue


def cmh_stat(profile, fibres, y):
    """Mantel-Haenszel score for a predeclared a>b raw contrast."""
    points = profiles_3x3()
    exposure = (points[profile, 0] > points[profile, 1]).astype(int)
    score = 0.0; variance = 0.0
    for g in np.unique(fibres):
        mask = fibres == g
        z = exposure[mask]; yy = y[mask]
        n = len(yy); n1 = int(z.sum()); n0 = n - n1
        m1 = int(yy.sum());
        if n < 2 or n1 == 0 or n0 == 0 or m1 == 0 or m1 == n:
            continue
        a = float(np.sum(yy[z == 1]))
        expected = n1 * m1 / n
        var = n1 * n0 * m1 * (n - m1) / (n * n * (n - 1.0))
        score += a - expected; variance += var
    return float(score * score / variance) if variance > 0 else 0.0


def cmh_pvalue(points, selected_fibre, selected_profile, y, reps=199, seed=0):
    fibres = candidate_values(points)
    observed = cmh_stat(selected_profile, selected_fibre, y)
    rng = np.random.default_rng(seed)
    by_fibre = {int(g): np.flatnonzero(fibres == g) for g in np.unique(fibres)}
    draws = np.empty((reps, len(selected_profile)), dtype=int)
    for g, ids in by_fibre.items():
        mask = selected_fibre == g
        if np.any(mask):
            draws[:, mask] = rng.choice(ids, size=(reps, int(mask.sum())))
    reference = np.asarray([cmh_stat(draws[b], selected_fibre, y) for b in range(reps)])
    return float((1.0 + np.sum(reference >= observed)) / (reps + 1.0))


def ml_residual_pvalue(points, selected_fibre, selected_profile, y,
                       reps=99, seed=0, folds=5):
    """Cross-fitted raw-coordinate residual score with fibre permutations.

    Forests are trained without held-out outcomes.  The statistic is the
    held-out log-score gain over the fibre-only probability.  Permutations
    exchange raw profiles inside each held-out fibre, preserving the complete
    candidate summary.
    """
    rng = np.random.default_rng(seed)
    n = len(y); fibres = candidate_values(points)
    order = rng.permutation(n); fold_ids = np.array_split(order, folds)
    actual_gain = 0.0
    fold_models = []
    for fold, test in enumerate(fold_ids):
        train = np.setdiff1d(order, test, assume_unique=False)
        x_train = np.column_stack((points[selected_profile[train]], selected_fibre[train]))
        x_test = np.column_stack((points[selected_profile[test]], selected_fibre[test]))
        clf = RandomForestClassifier(n_estimators=60, max_depth=4,
                                     min_samples_leaf=8, random_state=seed + fold,
                                     n_jobs=1)
        clf.fit(x_train, y[train])
        prob = clf.predict_proba(x_test)[:, 1]
        p0 = 1.0 / (1.0 + np.exp(-(-0.35 + 0.18 * (selected_fibre[test] - 2))))
        prob = np.clip(prob, 1e-6, 1 - 1e-6); p0 = np.clip(p0, 1e-6, 1 - 1e-6)
        actual_gain += float(np.sum(y[test] * np.log(prob) + (1-y[test]) * np.log(1-prob)
                                  - y[test] * np.log(p0) - (1-y[test]) * np.log(1-p0)))
        fold_models.append((test, clf))
    reference = np.empty(reps)
    for b in range(reps):
        gain = 0.0
        for test, clf in fold_models:
            shuffled = selected_profile[test].copy()
            for g in np.unique(selected_fibre[test]):
                local = np.flatnonzero(selected_fibre[test] == g)
                shuffled[local] = rng.permutation(shuffled[local])
            x_test = np.column_stack((points[shuffled], selected_fibre[test]))
            prob = np.clip(clf.predict_proba(x_test)[:, 1], 1e-6, 1 - 1e-6)
            p0 = np.clip(1.0 / (1.0 + np.exp(-(-0.35 + 0.18 * (selected_fibre[test] - 2)))), 1e-6, 1 - 1e-6)
            gain += float(np.sum(y[test] * np.log(prob) + (1-y[test]) * np.log(1-prob)
                                - y[test] * np.log(p0) - (1-y[test]) * np.log(1-p0)))
        reference[b] = gain
    return float((1.0 + np.sum(reference >= actual_gain)) / (reps + 1.0))


def run(reps=100, respondents=400, randomization_reps=199,
        ml_randomization_reps=99, eta=0.45,
        out="results/fibre_baseline_comparison.csv"):
    points = profiles_3x3()
    conditions = ("null", "aligned", "interaction", "hidden_quadratic", "hidden_orthogonal")
    rows = []
    for condition in conditions:
        for rep in range(reps):
            pts, fibre, profile, y = generate(20261007 + rep,
                respondents=respondents, condition=condition, eta=eta)
            pvals = {
                "support_complete": score_test(pts, fibre, profile, y, "support_complete",
                    randomization_reps=randomization_reps, seed=20262000 + rep),
                "complete_fibre_lr": saturated_lr_pvalue(pts, fibre, profile, y,
                    reps=randomization_reps, seed=20263000 + rep),
                "cmh": cmh_pvalue(pts, fibre, profile, y,
                    reps=randomization_reps, seed=20264000 + rep),
                "ml_residual": ml_residual_pvalue(pts, fibre, profile, y,
                    reps=ml_randomization_reps, seed=20265000 + rep),
            }
            for method, pvalue in pvals.items():
                rows.append({"condition": condition, "method": method, "rep": rep,
                    "respondents": respondents, "eta": eta, "pvalue": pvalue,
                    "reject": int(pvalue < 0.05)})
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    for condition in conditions:
        for method in ("support_complete", "complete_fibre_lr", "cmh", "ml_residual"):
            subset = [r for r in rows if r["condition"] == condition and r["method"] == method]
            print(condition, method, round(float(np.mean([r["reject"] for r in subset])), 3))
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--respondents", type=int, default=400)
    parser.add_argument("--randomization-reps", type=int, default=199)
    parser.add_argument("--ml-randomization-reps", type=int, default=99)
    parser.add_argument("--eta", type=float, default=0.45)
    parser.add_argument("--out", default="results/fibre_baseline_comparison.csv")
    run(**vars(parser.parse_args()))
