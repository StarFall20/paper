"""Cross-fitted observational test for approximate model-equivalent tasks.

The proposed instrument is unavailable in the public Swissmetro file, so this
script implements the weaker fallback explicitly.  A candidate model is fit on
different respondents, held-out tasks are paired within respondent using only
their attribute-derived utility coordinates, and the observed choice contrast
is residualised by the candidate probabilities.  Inference uses respondent
cluster multiplier bootstrap; labels never enter pair construction.

The script also contains a small engineered simulation.  It is a feasibility
and boundary check, not an empirical claim about Swissmetro.
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize


ALT_NAMES = ("train", "sm", "car")
CHOICE_FIELDS = ("TRAIN_AV", "SM_AV", "CAR_AV")


def softmax(u):
    z = u - np.max(u, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def read_swissmetro(path, dce_only=True):
    """Read the public Swissmetro file with the canonical SP preparation.

    The DCE analysis follows the documented Biogeme preparation: retain valid
    choices for purposes 1 and 3, apply the GA discount to train/Swissmetro
    cost, and retain the stated-preference availability flags.  Setting
    ``dce_only=False`` is available for an all-row sensitivity check.
    """
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            choice = int(float(row["CHOICE"])) - 1
            purpose = int(float(row["PURPOSE"]))
            if choice < 0:
                continue
            if dce_only and purpose not in (1, 3):
                continue
            sp = float(row["SP"]) != 0
            ga_free = float(row["GA"]) == 0
            rows.append({
                "id": int(float(row["ID"])),
                "choice": choice,
                "x": np.array([
                    [float(row["TRAIN_TT"]), float(row["TRAIN_CO"]) * ga_free],
                    [float(row["SM_TT"]), float(row["SM_CO"]) * ga_free],
                    [float(row["CAR_TT"]), float(row["CAR_CO"])],
                ], dtype=float),
                "avail": np.array([
                    float(row["TRAIN_AV"]) * sp,
                    float(row["SM_AV"]),
                    float(row["CAR_AV"]) * sp,
                ], dtype=float),
            })
    return rows


def rows_to_arrays(rows):
    ids = np.array([r["id"] for r in rows], dtype=int)
    choices = np.array([r["choice"] for r in rows], dtype=int)
    x = np.stack([r["x"] for r in rows])
    avail = np.stack([r["avail"] for r in rows])
    return ids, choices, x, avail


def design(x):
    """Alternative-specific design with train/car ASCs and time/cost /100."""
    n = len(x)
    out = np.zeros((n, 3, 6), dtype=float)
    out[:, 0, 0] = 1.0  # train ASC
    out[:, 2, 1] = 1.0  # car ASC
    out[:, :, 2:4] = x / 100.0
    return out


def fit_mnl(X, y, avail, l2=1e-8):
    """Fit an availability-aware MNL by converged BFGS maximum likelihood.

    The previous fixed-step update was sensitive to the very wide Swissmetro
    cost range.  The objective is written in log-sum-exp form and returns the
    analytic score, so convergence is checked by the optimizer rather than by
    an arbitrary learning rate and iteration count.
    """
    valid = avail[np.arange(len(y)), y] > 0
    Xv, yv, av = X[valid], y[valid], avail[valid]
    n, _, p = Xv.shape

    def objective(beta, need_grad=True):
        utility = np.einsum("njp,p->nj", Xv, beta)
        utility = np.where(av > 0, utility, -np.inf)
        max_u = np.max(utility, axis=1, keepdims=True)
        exp_shift = np.exp(utility - max_u)
        denom = np.sum(exp_shift, axis=1, keepdims=True)
        logp = utility - max_u - np.log(denom)
        value = -np.sum(logp[np.arange(n), yv]) / n + 0.5 * l2 * np.dot(beta, beta)
        if not need_grad:
            return value
        probs = exp_shift / denom
        target = np.zeros_like(probs)
        target[np.arange(n), yv] = 1.0
        grad = np.einsum("nj,njp->p", probs - target, Xv) / n + l2 * beta
        return value, grad

    result = minimize(lambda b: objective(b), np.zeros(p), jac=True,
                      method="BFGS", options={"gtol": 1e-8, "maxiter": 4000})
    # BFGS can report precision loss when the Hessian is nearly singular even
    # after reaching a numerically stationary point (common for rich
    # alternative-specific designs).  Accept only a finite solution with a
    # small analytic score; otherwise fail loudly.
    if not result.success:
        grad_norm = float(np.linalg.norm(result.jac)) if result.jac is not None else np.inf
        if not np.isfinite(result.fun) or grad_norm > 2e-5:
            raise RuntimeError(f"MNL optimizer did not converge: {result.message}; gradient={grad_norm}")
    return result.x


def predict(X, avail, beta):
    utility = np.einsum("njp,p->nj", X, beta)
    utility = np.where(avail > 0, utility, -1e9)
    return softmax(utility)


def pair_rows(ids, x, avail, beta, tolerance, orientation="time"):
    """Pair rows by frozen model coordinates and availability pattern."""
    X = design(x)
    probs = predict(X, avail, beta)
    utility = np.einsum("njp,p->nj", X, beta)
    pairs = []
    by_id = defaultdict(list)
    for i, rid in enumerate(ids):
        by_id[int(rid)].append(i)
    for rid, indices in by_id.items():
        for left_pos, left_idx in enumerate(indices):
            for right_idx in indices[left_pos + 1:]:
                if not np.array_equal(avail[left_idx], avail[right_idx]):
                    continue
                # Coordinates are utility differences relative to Swissmetro.
                left_coord = np.array([utility[left_idx, 0] - utility[left_idx, 1],
                                       utility[left_idx, 2] - utility[left_idx, 1]])
                right_coord = np.array([utility[right_idx, 0] - utility[right_idx, 1],
                                        utility[right_idx, 2] - utility[right_idx, 1]])
                if np.max(np.abs(left_coord - right_coord)) <= tolerance:
                    # Deterministic covariate orientation avoids using outcomes
                    # to choose the sign.  Observational analyses should report
                    # sensitivity because this convention is not randomized.
                    if orientation == "index":
                        left_key = (left_idx,)
                        right_key = (right_idx,)
                    elif orientation == "cost":
                        left_key = (float(x[left_idx, :, 1].sum()), float(x[left_idx, :, 0].sum()), left_idx)
                        right_key = (float(x[right_idx, :, 1].sum()), float(x[right_idx, :, 0].sum()), right_idx)
                    else:  # time: the declared default
                        left_key = (float(x[left_idx, :, 0].sum()), float(x[left_idx, :, 1].sum()), left_idx)
                        right_key = (float(x[right_idx, :, 0].sum()), float(x[right_idx, :, 1].sum()), right_idx)
                    pair_left, pair_right = left_idx, right_idx
                    if right_key < left_key:
                        pair_left, pair_right = pair_right, pair_left
                    pairs.append((rid, pair_left, pair_right,
                                  probs[pair_left], probs[pair_right]))
    return pairs


def pair_contrasts(pairs, choices):
    if not pairs:
        return np.empty((0, 3)), np.empty(0, dtype=int)
    values = []
    clusters = []
    for rid, left, right, p_left, p_right in pairs:
        observed = np.eye(3)[choices[left]] - np.eye(3)[choices[right]]
        values.append(observed - (p_left - p_right))
        clusters.append(int(rid))
    return np.asarray(values), np.asarray(clusters, dtype=int)


def cluster_stat(contrasts, clusters):
    """Wald-like norm based on respondent-level mean contrasts."""
    if len(contrasts) == 0:
        return np.nan, np.zeros(3), np.nan
    unique = np.unique(clusters)
    sums = np.stack([contrasts[clusters == rid].sum(axis=0) for rid in unique])
    counts = np.asarray([(clusters == rid).sum() for rid in unique], dtype=float)
    # Weight each respondent equally; this prevents respondents with many
    # approximate matches from dominating the test.
    cluster_means = sums / counts[:, None]
    mean = cluster_means.mean(axis=0)
    centered = cluster_means - mean
    cov = centered.T @ centered / max(len(unique) - 1, 1)
    cov += np.eye(3) * 1e-10
    stat = float(len(unique) * (mean @ np.linalg.pinv(cov) @ mean))
    return stat, mean, cov


def multiplier_pvalue(contrasts, clusters, reps=499, seed=20261006):
    if len(contrasts) == 0:
        return np.nan, np.nan
    unique = np.unique(clusters)
    sums = np.stack([contrasts[clusters == rid].sum(axis=0) for rid in unique])
    counts = np.asarray([(clusters == rid).sum() for rid in unique], dtype=float)
    cluster_means = sums / counts[:, None]
    observed, _, cov = cluster_stat(contrasts, clusters)
    rng = np.random.default_rng(seed)
    centered = cluster_means - cluster_means.mean(axis=0)
    bootstrap = np.empty(reps)
    for b in range(reps):
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(unique))
        draw = centered * signs[:, None]
        mean = draw.mean(axis=0)
        # The covariance is fixed under the multiplier null, which gives a
        # fast studentised randomisation of the respondent-level score.
        bootstrap[b] = len(unique) * (mean @ np.linalg.pinv(cov) @ mean)
    pvalue = (1.0 + np.sum(bootstrap >= observed)) / (reps + 1.0)
    return float(pvalue), float(observed)


def cross_fit_test(ids, choices, x, avail, tolerance=0.02, folds=2,
                   bootstrap=499, seed=20261006, orientation="time"):
    unique_ids = np.unique(ids)
    rng = np.random.default_rng(seed)
    shuffled = unique_ids.copy(); rng.shuffle(shuffled)
    fold_ids = np.array_split(shuffled, folds)
    all_contrasts = []
    all_clusters = []
    fold_rows = []
    for fold, test_ids in enumerate(fold_ids):
        test_mask = np.isin(ids, test_ids)
        train_mask = ~test_mask
        beta = fit_mnl(design(x[train_mask]), choices[train_mask], avail[train_mask])
        pairs = pair_rows(ids[test_mask], x[test_mask], avail[test_mask], beta,
                          tolerance, orientation=orientation)
        # Reindex held-out choices to the pair-row arrays.
        test_choices = choices[test_mask]
        contrasts, clusters = pair_contrasts(pairs, test_choices)
        # pair_rows stores local held-out row indices, so the above is valid.
        all_contrasts.append(contrasts); all_clusters.append(clusters)
        fold_rows.append({
            "fold": fold, "train_respondents": int(len(np.unique(ids[train_mask]))),
            "test_respondents": int(len(test_ids)), "pairs": len(pairs),
            "respondents_with_pair": int(len(np.unique(clusters))) if len(clusters) else 0,
            "beta_time": beta[2], "beta_cost": beta[3],
        })
    if not all_contrasts or not any(len(v) for v in all_contrasts):
        return {"pairs": 0, "respondents_with_pair": 0, "pvalue": np.nan,
                "statistic": np.nan, "gap_train": np.nan, "gap_sm": np.nan,
                "gap_car": np.nan, "folds": fold_rows}
    contrasts = np.concatenate(all_contrasts)
    clusters = np.concatenate(all_clusters)
    pvalue, statistic = multiplier_pvalue(contrasts, clusters, reps=bootstrap, seed=seed + 1)
    _, mean, _ = cluster_stat(contrasts, clusters)
    return {"pairs": int(len(contrasts)), "respondents_with_pair": int(len(np.unique(clusters))),
            "pvalue": pvalue, "statistic": statistic,
            "gap_train": float(mean[0]), "gap_sm": float(mean[1]),
            "gap_car": float(mean[2]), "folds": fold_rows}


def simulate(seed=20261006, respondents=400, nuisance_tasks=4,
             condition="additive"):
    """Engineered paired-task data for size/power boundary checks."""
    rng = np.random.default_rng(seed)
    beta = np.array([-0.35, -0.18, -1.15, -0.95])
    focal_left = np.array([[55., 21.], [65., 30.], [75., 39.]])
    focal_right = np.array([[75., 31.], [85., 40.], [95., 49.]])
    rows_x, rows_y, rows_a, rows_id = [], [], [], []
    for rid in range(respondents):
        random_cost = rng.normal(0.0, 1.0) if condition == "random_price" else 0.0
        tasks = [focal_left, focal_right]
        for _ in range(nuisance_tasks):
            base = rng.uniform(35, 130, size=(3, 2))
            tasks.append(base)
        # The paired tasks are randomly interleaved with nuisance tasks.  This
        # removes a fixed first/second position from the simulation; pair
        # orientation is still determined only by row order, never by choice.
        rng.shuffle(tasks)
        for t, xx in enumerate(tasks):
            utility = np.array([beta[0], 0.0, beta[1]]) + beta[2:] @ (xx.T / 100.0)
            if condition in ("nonlinear", "combined"):
                # Deliberately visible misspecification for a power check.
                # The Swissmetro application does not use this simulated
                # coefficient; it only checks whether the test can detect a
                # known omitted basis term.
                utility += 3.5 * (xx[:, 0] / 100.0) ** 2
            if condition in ("interaction", "combined"):
                utility += 4.0 * (xx[:, 0] / 100.0) * (xx[:, 1] / 100.0)
            if condition == "random_price":
                utility += random_cost * (xx[:, 1] / 100.0)
            p = softmax(utility[None, :])[0]
            rows_x.append(xx); rows_a.append(np.ones(3)); rows_id.append(rid)
            rows_y.append(int(rng.choice(3, p=p)))
    return (np.asarray(rows_id), np.asarray(rows_y), np.asarray(rows_x),
            np.asarray(rows_a))


def run_simulation(reps=80, bootstrap=199, tolerance=0.02,
                   out="results/observational_equivalence_simulation.csv",
                   orientation="time"):
    rows = []
    for condition in ("additive", "nonlinear", "interaction", "random_price"):
        for rep in range(reps):
            ids, choices, x, avail = simulate(seed=20261006 + rep, condition=condition)
            result = cross_fit_test(ids, choices, x, avail, tolerance=tolerance,
                                    bootstrap=bootstrap, seed=20300000 + rep,
                                    orientation=orientation)
            rows.append({"condition": condition, "rep": rep, "tolerance": tolerance,
                         "orientation": orientation,
                         "pairs": result["pairs"],
                         "respondents_with_pair": result["respondents_with_pair"],
                         "pvalue": result["pvalue"], "statistic": result["statistic"],
                         "gap_train": result["gap_train"], "gap_sm": result["gap_sm"],
                         "gap_car": result["gap_car"]})
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    for condition in ("additive", "nonlinear", "interaction", "random_price"):
        subset = [r for r in rows if r["condition"] == condition]
        print(condition, "rejection_rate", round(float(np.mean([r["pvalue"] < 0.05 for r in subset])), 4),
              "mean_pairs", round(float(np.mean([r["pairs"] for r in subset])), 1))
    print(f"wrote {len(rows)} observational-equivalence rows to {out}")


def run_swissmetro(path, out="results/observational_equivalence_swissmetro.csv",
                   tolerances=(0.01, 0.02, 0.05, 0.1), bootstrap=499,
                   orientations=("index", "time", "cost")):
    with open(path, newline="") as f:
        raw_rows = sum(1 for _ in csv.DictReader(f, delimiter="\t"))
    rows = read_swissmetro(path, dce_only=True)
    rows = [r for r in rows if r["choice"] >= 0 and r["avail"][r["choice"]] > 0]
    ids, choices, x, avail = rows_to_arrays(rows)
    records = []
    for orientation in orientations:
        for tolerance in tolerances:
            result = cross_fit_test(ids, choices, x, avail, tolerance=tolerance,
                                    bootstrap=bootstrap, orientation=orientation)
            records.append({"orientation": orientation, "tolerance": tolerance,
                            "raw_rows": raw_rows, "rows": len(rows),
                            "dropped_or_non_dce": raw_rows - len(rows),
                            "respondents": len(np.unique(ids)),
                            **{k: v for k, v in result.items() if k != "folds"}})
            print("Swissmetro", "orientation", orientation, "tolerance", tolerance,
                  "pairs", result["pairs"], "respondents_with_pair", result["respondents_with_pair"],
                  "pvalue", result["pvalue"], "gaps", result["gap_train"], result["gap_sm"], result["gap_car"])
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    print(f"wrote {len(records)} Swissmetro observational-equivalence rows to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    sim = sub.add_parser("simulate")
    sim.add_argument("--reps", type=int, default=80)
    sim.add_argument("--bootstrap", type=int, default=199)
    sim.add_argument("--tolerance", type=float, default=0.02)
    sim.add_argument("--orientation", choices=("index", "time", "cost"), default="time")
    sim.add_argument("--out", default="results/observational_equivalence_simulation.csv")
    sm = sub.add_parser("swissmetro")
    sm.add_argument("path")
    sm.add_argument("--bootstrap", type=int, default=499)
    sm.add_argument("--orientations", nargs="+", choices=("index", "time", "cost"),
                    default=("index", "time", "cost"))
    sm.add_argument("--out", default="results/observational_equivalence_swissmetro.csv")
    args = ap.parse_args()
    if args.mode == "simulate":
        run_simulation(reps=args.reps, bootstrap=args.bootstrap,
                       tolerance=args.tolerance, out=args.out,
                       orientation=args.orientation)
    else:
        run_swissmetro(args.path, out=args.out, bootstrap=args.bootstrap,
                       orientations=tuple(args.orientations))
