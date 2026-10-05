"""Tree-model extension with respondent-level holdouts.

The environment does not provide macOS libomp, so the XGBoost package cannot
load its native library. HistGradientBoosting is run as a documented boosted
tree proxy; the script never labels its output as XGBoost evidence.
"""
from __future__ import annotations
import argparse, csv, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from run_simulation import CONDITIONS, N_ALTERNATIVES, make_data

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import log_loss


def metrics(prob, y, v):
    pred = prob.argmax(1)
    y = y.reshape(-1); v = v.reshape(-1, N_ALTERNATIVES)
    acc = float((pred == y).mean())
    ll = float(log_loss(y, prob, labels=list(range(N_ALTERNATIVES))))
    brier = float(((prob - np.eye(N_ALTERNATIVES)[y]) ** 2).sum(1).mean())
    share_rmse = float(np.sqrt(np.mean((prob.mean(0) - np.bincount(y, minlength=N_ALTERNATIVES) / len(y)) ** 2)))
    regret = float(np.mean(v.max(1) - v[np.arange(len(y)), pred]))
    return acc, ll, brier, share_rmse, regret


def run(reps=10, n=400, tasks=12, out="results/tree_extension.csv"):
    rows = []
    for ci, cond in enumerate(CONDITIONS):
        for rep in range(reps):
            seed = 20301005 + ci * 1000 + rep
            x, z, choices, v = make_data(seed, n, tasks, cond)
            features = np.concatenate([x.reshape(n * tasks, -1), np.repeat(z, tasks, axis=0)], axis=1)
            y = choices.reshape(-1)
            vv = v.reshape(-1, N_ALTERNATIVES)
            ids = np.arange(n); np.random.default_rng(seed + 7).shuffle(ids)
            cut = int(.8 * n); train_ids, test_ids = ids[:cut], ids[cut:]
            tr = np.concatenate([np.arange(i * tasks, (i + 1) * tasks) for i in train_ids])
            te = np.concatenate([np.arange(i * tasks, (i + 1) * tasks) for i in test_ids])
            models = [
                ("Random_Forest", RandomForestClassifier(n_estimators=200, min_samples_leaf=5, max_features="sqrt", random_state=seed, n_jobs=1)),
                ("Boosted_Tree_Proxy", HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=15, learning_rate=.05, l2_regularization=1e-3, random_state=seed)),
            ]
            for name, model in models:
                model.fit(features[tr], y[tr])
                pr = model.predict_proba(features[te])
                full = np.zeros((len(te), N_ALTERNATIVES)); full[:, model.classes_.astype(int)] = pr
                vals = metrics(full, y[te], vv[te])
                rows.append({"condition": cond.name, "replication": rep, "model": name,
                             "accuracy": vals[0], "logloss": vals[1], "brier": vals[2],
                             "share_rmse": vals[3], "decision_regret": vals[4]})
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} model-replication rows to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=10); ap.add_argument("--n", type=int, default=400); ap.add_argument("--tasks", type=int, default=12); ap.add_argument("--out", default="results/tree_extension.csv")
    a = ap.parse_args(); run(a.reps, a.n, a.tasks, a.out)
