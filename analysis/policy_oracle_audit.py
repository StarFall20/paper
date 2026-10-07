"""Oracle benchmark for the policy-path audit.

This file is deliberately separate from the earlier disagreement script.  A
model-to-model gap is not an error estimate: both specifications can share an
omitted term.  The benchmark therefore reconstructs the data-generating
utility, evaluates both fitted models on a fixed intervention path, and
reports model-to-oracle error, anchored response error, and the observed-task
fit.  The final witness is an exact finite-support construction in which the
two fitted specifications agree on the observed support and still disagree
with the oracle off support.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(__file__))
from policy_path_disagreement import PATH, counterfactual_x
from recoverability_triage import respondent_split, row_indices
from run_simulation import CONDITIONS, N_ALTERNATIVES, PRICE_INDEX, OPT_OUT_INDEX, make_data


def softmax_rows(u: np.ndarray) -> np.ndarray:
    u = u - u.max(axis=1, keepdims=True)
    e = np.exp(u)
    return e / e.sum(axis=1, keepdims=True)


def oracle_utility(x: np.ndarray, z: np.ndarray, condition, random_price=None) -> np.ndarray:
    """The exact utility used by ``run_simulation.make_data``."""
    v = np.einsum("ntjp,p->ntj", x, np.array([.55, .35, .45, .25, -.85]))
    v[:, :, OPT_OUT_INDEX] -= .35
    if condition.nonlinear:
        v += .55 * x[:, :, :, 0] ** 2 - .30 * x[:, :, :, PRICE_INDEX] ** 2
    if condition.threshold:
        v -= .90 * np.maximum(x[:, :, :, PRICE_INDEX] - 1.25, 0.)
    if condition.interaction:
        v += .75 * x[:, :, :, 0] * x[:, :, :, 1]
        v += .55 * x[:, :, :, 2] * z[:, None, None, 1]
    if condition.heterogeneity:
        if random_price is None:
            raise ValueError("random_price is required for the heterogeneity condition")
        v += random_price[:, None, None] * x[:, :, :, PRICE_INDEX]
    v[:, :, OPT_OUT_INDEX] += .15 * z[:, None, 0]
    return v


def recover_random_price(v, x, z, condition):
    """Recover the latent respondent coefficient from the returned utility."""
    if not condition.heterogeneity:
        return np.zeros(x.shape[0])
    known = oracle_utility(x, z, type("C", (), dict(
        nonlinear=condition.nonlinear, threshold=condition.threshold,
        interaction=condition.interaction, heterogeneity=False))())
    prices = x[:, 0, 0, PRICE_INDEX]
    return (v[:, 0, 0] - known[:, 0, 0]) / prices


def design_matrix(x: np.ndarray, z: np.ndarray, structured: bool = False) -> np.ndarray:
    """Corrected behavioural library, including the opt-out income term."""
    n, tasks, j, _ = x.shape
    fx = x.reshape(n * tasks, j, 5)
    fz = np.repeat(z, tasks, axis=0)
    out = np.broadcast_to((np.arange(j) == OPT_OUT_INDEX).astype(float)[None, :, None],
                          (n * tasks, j, 1))
    cols = [fx, out, out * fz[:, None, 0:1]]
    if structured:
        cols += [fx[:, :, 0:1] ** 2, fx[:, :, PRICE_INDEX:PRICE_INDEX + 1] ** 2]
        cols += [np.maximum(fx[:, :, PRICE_INDEX:PRICE_INDEX + 1] - 1.25, 0.)]
        cols += [fx[:, :, 0:1] * fx[:, :, 1:2]]
        cols += [fx[:, :, 2:3] * fz[:, None, 1:2]]
        cols += [fx[:, :, PRICE_INDEX:PRICE_INDEX + 1] * fz[:, None, 0:1]]
    return np.concatenate(cols, axis=2)


def fit_mnl_lbfgs(X: np.ndarray, y: np.ndarray, l2: float = 1e-3):
    """Converged penalized MNL fit with a reproducible gradient check."""
    rows, j, p = X.shape
    target = np.eye(j)[y]

    def fun(beta):
        pr = softmax_rows(np.einsum("rjp,p->rj", X, beta))
        loss = -np.log(np.clip(pr[np.arange(rows), y], 1e-12, 1.)).mean() + .5 * l2 * np.dot(beta, beta)
        grad = np.einsum("rj,rjp->p", pr - target, X) / rows + l2 * beta
        return float(loss), grad

    result = minimize(lambda b: fun(b)[0], np.zeros(p), jac=lambda b: fun(b)[1],
                      method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-11, "gtol": 1e-7})
    if not result.success:
        raise RuntimeError(f"MNL optimizer did not converge: {result.message}")
    return result.x, float(np.max(np.abs(fun(result.x)[1]))), int(result.nit)


def model_prob(x, z, beta, structured):
    X = design_matrix(x, z, structured).reshape(-1, N_ALTERNATIVES, len(beta))
    return softmax_rows(np.einsum("rjp,p->rj", X, beta)).reshape(x.shape[0], x.shape[1], N_ALTERNATIVES)


def true_prob(x, z, condition, v, random_price):
    # Both fitted models predict population probabilities conditional on the
    # observed attributes, not probabilities conditional on an unobserved
    # person's random coefficient. Integrate that coefficient in the oracle.
    if condition.heterogeneity:
        nodes, weights = np.polynomial.hermite.hermgauss(31)
        prob = np.zeros(x.shape[:3])
        for node, weight in zip(nodes, weights / np.sqrt(np.pi)):
            coefficient = np.full(len(x), .45 * np.sqrt(2.) * node)
            utility = oracle_utility(x, z, condition, coefficient)
            prob += weight * softmax_rows(utility.reshape(-1, N_ALTERNATIVES)).reshape(x.shape[:3])
        return prob
    return softmax_rows(oracle_utility(x, z, condition, random_price).reshape(-1, N_ALTERNATIVES)).reshape(x.shape[0], x.shape[1], N_ALTERNATIVES)


def path_metrics(x, z, condition, v, random_price, beta0, beta1):
    gaps, e0, e1, r0, r1 = [], [], [], [], []
    for q, p in PATH:
        xx = counterfactual_x(x, q, p)
        truth = true_prob(xx, z, condition, v, random_price)
        pb = model_prob(xx, z, beta0, False)
        ps = model_prob(xx, z, beta1, True)
        gaps.append(float(np.mean(np.abs(pb - ps))))  # mean absolute probability gap
        e0.append(float(np.mean(np.abs(pb - truth))))
        e1.append(float(np.mean(np.abs(ps - truth))))
        # The anchor is the (q,p)=(0,0) point; response errors compare changes,
        # so a common level error cancels before the policy path is judged.
        if q == 0.0 and p == 0.0:
            anchor_truth, anchor0, anchor1 = truth, pb, ps
    for q, p in PATH:
        xx = counterfactual_x(x, q, p)
        truth = true_prob(xx, z, condition, v, random_price)
        pb = model_prob(xx, z, beta0, False)
        ps = model_prob(xx, z, beta1, True)
        r0.append(float(np.mean(np.abs((pb - anchor0) - (truth - anchor_truth)))))
        r1.append(float(np.mean(np.abs((ps - anchor1) - (truth - anchor_truth)))))
    return {
        "model_gap": float(np.mean(gaps)),
        "base_oracle_error": float(np.mean(e0)),
        "structured_oracle_error": float(np.mean(e1)),
        "base_response_error": float(np.mean(r0)),
        "structured_response_error": float(np.mean(r1)),
        "base_oracle_max": float(np.max(e0)),
        "structured_oracle_max": float(np.max(e1)),
    }


def exact_common_mode_witness():
    """Finite-support impossibility witness, independent of estimation noise."""
    # On the observed support q in {-1,+1}, h(q)=q^2-1 is identically zero.
    # At the declared intervention q=0 it equals -1.  The two fitted models
    # can agree exactly on observed utilities while the oracle changes.
    q_obs = np.array([-1., 1.])
    q_path = np.array([-1., -.5, 0., .5, 1.])
    rows = []
    for q in q_path:
        model_utility = np.array([[.55*q, .3, -.35]])
        oracle = model_utility.copy()
        oracle[0, 0] += q*q - 1.
        pm = softmax_rows(model_utility)
        pt = softmax_rows(oracle)
        rows.append({"q": float(q), "observed_support": int(q in q_obs),
                     "model_gap": 0.0, "oracle_omitted_term": float(q*q - 1.),
                     "both_models_oracle_error": float(np.mean(np.abs(pm - pt)))})
    return rows


def run(reps=12, n=400, tasks=12, out="results/policy_oracle_audit.csv", witness_out="results/common_mode_witness.csv"):
    rows = []
    for ci, condition in enumerate(CONDITIONS):
        for rep in range(reps):
            seed = 29100000 + ci * 1000 + rep
            x, z, choices, v = make_data(seed, n, tasks, condition)
            random_price = recover_random_price(v, x, z, condition)
            tr, va, te = respondent_split(n, seed + 77)
            tr_rows = row_indices(tr, tasks)
            ytr = choices[tr].reshape(-1)
            b0, g0, it0 = fit_mnl_lbfgs(design_matrix(x, z, False)[tr_rows], ytr)
            b1, g1, it1 = fit_mnl_lbfgs(design_matrix(x, z, True)[tr_rows], ytr)
            metrics = path_metrics(x[va], z[va], condition, v[va], random_price[va], b0, b1)
            X0 = design_matrix(x, z, False)[row_indices(va, tasks)]
            X1 = design_matrix(x, z, True)[row_indices(va, tasks)]
            yva = choices[va].reshape(-1)
            p0 = softmax_rows(np.einsum("rjp,p->rj", X0, b0))
            p1 = softmax_rows(np.einsum("rjp,p->rj", X1, b1))
            rows.append({"condition": condition.name, "replication": rep,
                         **metrics, "observed_model_gap": float(np.mean(np.abs(p0 - p1))),
                         "observed_base_logloss": float(-np.log(np.clip(p0[np.arange(len(yva)), yva], 1e-12, 1.)).mean()),
                         "observed_structured_logloss": float(-np.log(np.clip(p1[np.arange(len(yva)), yva], 1e-12, 1.)).mean()),
                         "base_converged_grad": g0, "structured_converged_grad": g1,
                         "base_iterations": it0, "structured_iterations": it1})
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(witness_out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(exact_common_mode_witness()[0])); w.writeheader(); w.writerows(exact_common_mode_witness())
    for c in [c.name for c in CONDITIONS]:
        sub = [r for r in rows if r["condition"] == c]
        print(c, "gap", round(float(np.mean([r["model_gap"] for r in sub])), 5),
              "base_err", round(float(np.mean([r["base_oracle_error"] for r in sub])), 5),
              "struct_err", round(float(np.mean([r["structured_oracle_error"] for r in sub])), 5),
              "resp0", round(float(np.mean([r["base_response_error"] for r in sub])), 5),
              "resp1", round(float(np.mean([r["structured_response_error"] for r in sub])), 5))
    print("wrote", out, witness_out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--tasks", type=int, default=12)
    ap.add_argument("--out", default="results/policy_oracle_audit.csv")
    ap.add_argument("--witness-out", default="results/common_mode_witness.csv")
    run(**vars(ap.parse_args()))
