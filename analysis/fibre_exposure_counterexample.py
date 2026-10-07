"""Finite-support counterexample: candidate precision can coexist with zero fibre exposure."""
from __future__ import annotations

import csv
import os
from collections import defaultdict

import numpy as np


def candidate_precision(points, weights):
    points = np.asarray(points, dtype=float)
    weights = np.asarray(weights, dtype=float)
    phi = points.sum(axis=1)
    return float(np.sum(weights * (phi - np.sum(weights * phi)) ** 2))


def fibre_exposure(points, weights):
    points = np.asarray(points, dtype=float)
    weights = np.asarray(weights, dtype=float)
    phi = points.sum(axis=1).astype(int)
    h = points[:, 0] - points[:, 1]
    total = 0.0
    for level in sorted(set(phi)):
        ids = np.where(phi == level)[0]
        mass = float(np.sum(weights[ids]))
        if mass == 0.0:
            continue
        conditional = weights[ids] / mass
        mean = float(np.sum(conditional * h[ids]))
        total += mass * float(np.sum(conditional * (h[ids] - mean) ** 2))
    return total


def exposure_matrix(points, weights):
    points = np.asarray(points, dtype=float)
    weights = np.asarray(weights, dtype=float)
    phi = points.sum(axis=1).astype(int)
    features = np.column_stack((points[:, 0] - points[:, 1],
                                points[:, 0] * points[:, 1]))
    matrix = np.zeros((features.shape[1], features.shape[1]))
    for level in sorted(set(phi)):
        ids = np.where(phi == level)[0]
        mass = float(np.sum(weights[ids]))
        if mass == 0.0:
            continue
        conditional = weights[ids] / mass
        mean = np.sum(conditional[:, None] * features[ids], axis=0)
        centered = features[ids] - mean
        matrix += mass * centered.T @ (conditional[:, None] * centered)
    return matrix


def run(out="results/fibre_exposure_counterexample.csv"):
    designs = {
        "candidate_D_optimal_endpoints": (
            np.array([[0, 0], [1, 1]]), np.array([0.5, 0.5])),
        "full_factorial_orthogonal": (
            np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            np.full(4, 0.25)),
        "fibre_balanced_middle": (
            np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            np.array([0.125, 0.375, 0.375, 0.125])),
    }
    rows = []
    for name, (points, weights) in designs.items():
        eigenvalues = np.linalg.eigvalsh(exposure_matrix(points, weights))
        rows.append({
            "design": name,
            "candidate_precision_var_phi": candidate_precision(points, weights),
            "fibre_exposure_var_h_given_phi": fibre_exposure(points, weights),
            "singleton_fibre_share": float(np.sum(
                weights[np.isin(points.sum(axis=1), [0, 2])])),
            "exposure_rank": int(np.linalg.matrix_rank(
                exposure_matrix(points, weights), tol=1e-10)),
            "exposure_min_eigenvalue": float(np.min(eigenvalues)),
        })
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]),
                                lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    for row in rows:
        print(row)
    print("wrote", len(rows), "rows to", out)


if __name__ == "__main__":
    run()
