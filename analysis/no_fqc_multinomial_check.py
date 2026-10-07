"""Multinomial full-menu implementation check for NO-FQC.

The design keeps every candidate menu vector fixed across the reflected
orientation.  The check then uses independent multinomial contrasts and a
declared nuisance projection before comparing rejection rates with an
unprojected score.
"""
from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass

import numpy as np


def softmax(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    z = v - np.max(v)
    e = np.exp(z)
    return e / e.sum()


def pairwise_phi(b: np.ndarray) -> np.ndarray:
    """Return the complete J-menu vector of pairwise candidate gaps."""
    return np.array([b[i] - b[j]
                     for i in range(len(b)) for j in range(i + 1, len(b))],
                    dtype=float)


@dataclass(frozen=True)
class Arm:
    arm_id: int
    b: np.ndarray
    z: np.ndarray
    frame: np.ndarray
    scale_load: float


ARMS = (
    Arm(1, np.array([100.0, 90.0, 80.0]), np.array([2.0, 0.0, -2.0]),
        np.array([1.0, -1.0, 0.0]), 0.08),
    Arm(2, np.array([94.0, 86.0, 76.0]), np.array([1.5, -0.5, -2.5]),
        np.array([0.0, 1.0, -1.0]), 0.10),
    Arm(3, np.array([108.0, 91.0, 73.0]), np.array([2.5, 0.5, -1.5]),
        np.array([-1.0, 0.0, 1.0]), 0.12),
    Arm(4, np.array([88.0, 82.0, 68.0]), np.array([1.0, -1.0, -3.0]),
        np.array([1.0, 0.0, -1.0]), 0.14),
    Arm(5, np.array([115.0, 97.0, 79.0]), np.array([3.0, 0.0, -1.0]),
        np.array([0.0, -1.0, 1.0]), 0.16),
    Arm(6, np.array([102.0, 84.0, 71.0]), np.array([2.0, -1.5, -2.0]),
        np.array([-1.0, 1.0, 0.0]), 0.18),
    Arm(7, np.array([97.0, 87.0, 69.0]), np.array([0.5, -0.5, -2.5]),
        np.array([1.0, -1.0, 0.0]), 0.20),
    Arm(8, np.array([111.0, 93.0, 77.0]), np.array([2.8, 0.2, -1.8]),
        np.array([0.0, 1.0, -1.0]), 0.22),
)


def utilities(arm: Arm, t: float, eta: np.ndarray,
              nuisance: np.ndarray, unlisted: float = 0.0) -> np.ndarray:
    """Construct a three-alternative utility vector for one orientation."""
    # The candidate basis is b.  Raw z changes along the fibre, while b does
    # not.  The three local departure directions are deliberately distinct.
    z = arm.z + t
    h = np.array([
        z ** 2,
        z * np.array([1.0, -0.5, 0.75]),
        (z ** 3) / 6.0,
    ])
    v0 = 0.03 * arm.b
    v = v0 + eta @ h

    # A signed scale tangent and a signed framing tangent are declared before
    # outcomes.  They change the reflected menu while leaving b unchanged.
    scale, framing = nuisance
    v = v * (1.0 + scale * t * arm.scale_load)
    v = v + framing * t * arm.frame

    # This mechanism is intentionally outside the declared nuisance library.
    # It is used only as a boundary sensitivity condition.
    v = v + unlisted * t * (z ** 2) * np.array([0.4, -0.2, 0.3])
    return v


def delta_vector(eta: np.ndarray, nuisance: np.ndarray,
                 unlisted: float = 0.0) -> np.ndarray:
    """Stack two independent full-menu contrasts in the score coordinates."""
    rows = []
    cmat = np.array([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0]])
    for arm in ARMS:
        p_plus = softmax(utilities(arm, 1.0, eta, nuisance, unlisted))
        p_minus = softmax(utilities(arm, -1.0, eta, nuisance, unlisted))
        rows.append(cmat @ (p_plus - p_minus))
    return np.concatenate(rows)


def local_matrices(step: float = 1e-5):
    zero_eta = np.zeros(3)
    zero_nuisance = np.zeros(2)
    d0 = delta_vector(zero_eta, zero_nuisance)
    D = np.zeros((len(d0), 3))
    for j in range(3):
        e = np.zeros(3); e[j] = step
        D[:, j] = (delta_vector(e, zero_nuisance) -
                   delta_vector(-e, zero_nuisance)) / (2.0 * step)
    G = np.zeros((len(d0), 2))
    for j in range(2):
        n = np.zeros(2); n[j] = step
        G[:, j] = (delta_vector(zero_eta, n) -
                   delta_vector(zero_eta, -n)) / (2.0 * step)

    # C maps a three-category one-hot vector to two independent contrasts
    # relative to the third category.  This retains the full multinomial menu
    # information without using a singular J-dimensional covariance.
    cmat = np.array([[1.0, 0.0, -1.0],
                     [0.0, 1.0, -1.0]])
    blocks = []
    for arm in ARMS:
        p = softmax(utilities(arm, 0.0, zero_eta, zero_nuisance))
        # Z=2*T*C*Y for a fair orientation T. At the candidate null its mean
        # is zero. Var(Z)=4*C*diag(p)*C', including orientation variance.
        # This differs from the centered within-task multinomial covariance
        # C*(diag(p)-p*p')*C'; random orientation adds the p*p' term back.
        blocks.append(4.0 * cmat @ np.diag(p) @ cmat.T)
    Omega = np.zeros((2 * len(ARMS), 2 * len(ARMS)))
    for k, block in enumerate(blocks):
        sl = slice(2 * k, 2 * k + 2)
        Omega[sl, sl] = block
    W = np.linalg.inv(Omega)

    gram = G.T @ W @ G
    P = G @ np.linalg.inv(gram) @ G.T @ W
    M = np.eye(len(d0)) - P
    K = D.T @ W @ M @ D
    K = 0.5 * (K + K.T)
    eigenvalues = np.linalg.eigvalsh(K)
    return D, G, Omega, W, M, K, eigenvalues


def randomization_pvalue(scores: np.ndarray, reps: int,
                         rng: np.random.Generator) -> float:
    observed = abs(float(np.mean(scores)))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(reps, scores.shape[0]))
    draws = (signs @ scores) / scores.shape[0]
    return float((1.0 + np.sum(np.abs(draws) >= observed)) / (reps + 1.0))


def choice_draw(p: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.choice(3, p=p))


def simulate_scores(condition: str, respondents: int, w: np.ndarray,
                    rng: np.random.Generator) -> np.ndarray:
    eta = np.zeros(3)
    nuisance = np.zeros(2)
    unlisted = 0.0
    if condition in ("omitted", "combined"):
        eta[0] = 0.15
    if condition in ("declared_scale", "combined"):
        nuisance[0] = 0.65
    if condition in ("declared_framing", "combined"):
        nuisance[1] = 0.20
    if condition == "strong_declared_framing":
        nuisance[1] = 0.65
    if condition == "unlisted_process":
        unlisted = 0.10

    scores = np.zeros(respondents)
    cmat = np.array([[1.0, 0.0, -1.0],
                     [0.0, 1.0, -1.0]])
    for k, arm in enumerate(ARMS):
        orientation = rng.choice(np.array([-1.0, 1.0]), size=respondents)
        p_plus = softmax(utilities(arm, 1.0, eta, nuisance, unlisted))
        p_minus = softmax(utilities(arm, -1.0, eta, nuisance, unlisted))
        probabilities = np.where(orientation[:, None] > 0,
                                 p_plus[None, :], p_minus[None, :])
        u = rng.random(respondents)
        selected = np.sum(u[:, None] > np.cumsum(probabilities, axis=1), axis=1)
        selected = np.minimum(selected, 2)
        contrast = cmat[:, selected].T
        scores += 2.0 * orientation * (contrast @ w[2*k:2*k+2])
    return scores


def evaluate(condition: str, respondents: int, permutation_reps: int,
             seed: int, w_no_fqc: np.ndarray, w_raw: np.ndarray):
    rng = np.random.default_rng(seed)
    scores_no = simulate_scores(condition, respondents, w_no_fqc, rng)
    # Use a separate stream so the two diagnostics see identical design-level
    # randomness without sharing a mutable generator state.
    scores_raw = simulate_scores(condition, respondents, w_raw,
                                 np.random.default_rng(seed))
    p_no = randomization_pvalue(scores_no, permutation_reps,
                                np.random.default_rng(seed + 1001))
    p_raw = randomization_pvalue(scores_raw, permutation_reps,
                                 np.random.default_rng(seed + 2001))
    return int(p_no < 0.05), p_no, int(p_raw < 0.05), p_raw


def write_design(path: str, phi_deviation: float, rank: int,
                 eigenvalues: np.ndarray, orthogonality: float):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = ["arm", "b_1", "b_2", "b_3", "z_1", "z_2", "z_3",
              "phi_12", "phi_13", "phi_23", "full_phi_max_deviation",
              "residual_rank", "eigenvalue_1", "eigenvalue_2",
              "eigenvalue_3", "max_Gt_w"]
    rows = []
    for arm in ARMS:
        phi = pairwise_phi(arm.b)
        rows.append({
            "arm": arm.arm_id,
            "b_1": arm.b[0], "b_2": arm.b[1], "b_3": arm.b[2],
            "z_1": arm.z[0], "z_2": arm.z[1], "z_3": arm.z[2],
            "phi_12": phi[0], "phi_13": phi[1], "phi_23": phi[2],
            "full_phi_max_deviation": phi_deviation,
            "residual_rank": rank,
            "eigenvalue_1": eigenvalues[0],
            "eigenvalue_2": eigenvalues[1],
            "eigenvalue_3": eigenvalues[2],
            "max_Gt_w": orthogonality,
        })
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def run(reps: int = 200, respondents: int = 600, permutation_reps: int = 199,
        summary_out: str = "results/no_fqc_multinomial_summary.csv",
        design_out: str = "results/no_fqc_multinomial_design.csv"):
    D, G, _Omega, W, M, K, eigenvalues = local_matrices()
    g = D[:, 0]
    residual_signal = float(g.T @ W @ M @ g)
    w_no = W @ M @ g / np.sqrt(residual_signal)
    raw_signal = float(g.T @ W @ g)
    w_raw = W @ g / np.sqrt(raw_signal)
    phi_deviation = 0.0
    for arm in ARMS:
        plus = pairwise_phi(arm.b)
        minus = pairwise_phi(arm.b.copy())
        phi_deviation = max(phi_deviation, float(np.max(np.abs(plus - minus))))
    orthogonality = float(np.max(np.abs(G.T @ w_no)))
    rank = int(np.linalg.matrix_rank(M @ D, tol=1e-8))
    write_design(design_out, phi_deviation, rank, eigenvalues, orthogonality)

    conditions = ("null", "omitted", "declared_scale", "declared_framing",
                  "strong_declared_framing", "combined", "unlisted_process")
    rows = []
    for condition in conditions:
        no_rejections = []; raw_rejections = []; no_p = []; raw_p = []
        for rep in range(reps):
            a, p_a, b, p_b = evaluate(
                condition, respondents, permutation_reps,
                73000000 + rep, w_no, w_raw)
            no_rejections.append(a); raw_rejections.append(b)
            no_p.append(p_a); raw_p.append(p_b)
        rows.append({
            "condition": condition,
            "replications": reps,
            "respondents": respondents,
            "permutation_reps": permutation_reps,
            "no_fqc_rejection_rate": float(np.mean(no_rejections)),
            "raw_rejection_rate": float(np.mean(raw_rejections)),
            "no_fqc_mean_pvalue": float(np.mean(no_p)),
            "raw_mean_pvalue": float(np.mean(raw_p)),
            "full_phi_max_deviation": phi_deviation,
            "residual_rank": rank,
            "residual_eigenvalue_min": float(np.min(eigenvalues)),
            "max_Gt_w": orthogonality,
        })
    os.makedirs(os.path.dirname(summary_out) or ".", exist_ok=True)
    with open(summary_out, "w", newline="") as handle:
        fields = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    print("full-menu phi max deviation", phi_deviation)
    print("NO-FQC rank", rank, "eigenvalues", np.round(eigenvalues, 6).tolist())
    print("max |G' w_no_fqc|", orthogonality)
    for row in rows:
        print(row["condition"], "NO-FQC", round(row["no_fqc_rejection_rate"], 3),
              "raw", round(row["raw_rejection_rate"], 3))
    print(f"wrote {len(rows)} summary rows to {summary_out}")
    print(f"wrote {len(ARMS)} design rows to {design_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--respondents", type=int, default=600)
    parser.add_argument("--permutation-reps", type=int, default=199)
    parser.add_argument("--summary-out", default="results/no_fqc_multinomial_summary.csv")
    parser.add_argument("--design-out", default="results/no_fqc_multinomial_design.csv")
    run(**vars(parser.parse_args()))
