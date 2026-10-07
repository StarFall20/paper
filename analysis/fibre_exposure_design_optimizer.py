"""Finite-support design search for fibre-conditional sufficiency audits.

The search compares an ordinary candidate-index precision design with a
fibre-aware design.  It uses only a finite profile universe, so every result
is exact for the declared support and residual dictionary.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class DesignSummary:
    name: str
    candidate_variance: float
    fibre_logdet: float
    fibre_min_eigenvalue: float
    fibre_rank: int
    non_singleton_mass: float
    weights: np.ndarray


def profiles_3x3() -> np.ndarray:
    return np.asarray([(a, b) for a in range(3) for b in range(3)], dtype=float)


def candidate_values(points: np.ndarray) -> np.ndarray:
    return points.sum(axis=1)


def residual_dictionary(points: np.ndarray) -> np.ndarray:
    a, b = points[:, 0], points[:, 1]
    return np.column_stack((a - b, a * b))


def candidate_variance(points: np.ndarray, weights: np.ndarray) -> float:
    phi = candidate_values(points)
    mean = float(weights @ phi)
    return float(weights @ (phi - mean) ** 2)


def exposure_matrix(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    phi = candidate_values(points).astype(int)
    features = residual_dictionary(points)
    matrix = np.zeros((features.shape[1], features.shape[1]))
    for level in sorted(set(phi)):
        ids = np.flatnonzero(phi == level)
        mass = float(weights[ids].sum())
        if mass <= 0.0 or len(ids) < 2:
            continue
        conditional = weights[ids] / mass
        mean = conditional @ features[ids]
        centered = features[ids] - mean
        matrix += mass * centered.T @ (conditional[:, None] * centered)
    return matrix


def fibre_logdet(matrix: np.ndarray, ridge: float = 1e-8) -> float:
    sign, value = np.linalg.slogdet(matrix + ridge * np.eye(matrix.shape[0]))
    return float(value) if sign > 0 else float("-inf")


def summaries(points: np.ndarray, weights: np.ndarray, name: str) -> DesignSummary:
    matrix = exposure_matrix(points, weights)
    eigenvalues = np.linalg.eigvalsh(matrix)
    phi = candidate_values(points).astype(int)
    non_singleton = np.isin(phi, [1, 2, 3])
    return DesignSummary(
        name=name,
        candidate_variance=candidate_variance(points, weights),
        fibre_logdet=fibre_logdet(matrix),
        fibre_min_eigenvalue=float(eigenvalues[0]),
        fibre_rank=int(np.linalg.matrix_rank(matrix, tol=1e-10)),
        non_singleton_mass=float(weights[non_singleton].sum()),
        weights=weights.copy(),
    )


def allowed_move(counts: np.ndarray, points: np.ndarray, source: int, target: int,
                 min_fibre_mass: float, total: int) -> bool:
    if source == target or counts[source] <= 0:
        return False
    trial = counts.copy()
    trial[source] -= 1
    trial[target] += 1
    if min_fibre_mass <= 0:
        return True
    phi = candidate_values(points).astype(int)
    for level in (1, 2, 3):
        if trial[phi == level].sum() / total < min_fibre_mass - 1e-12:
            return False
    return True


def coordinate_exchange(points: np.ndarray, total: int, score: Callable[[np.ndarray], float],
                         min_fibre_mass: float = 0.0, max_rounds: int = 500) -> np.ndarray:
    # Start with one observation at every profile and distribute the rest evenly.
    counts = np.ones(len(points), dtype=int)
    counts += (total - len(points)) // len(points)
    for idx in range((total - counts.sum())):
        counts[idx % len(points)] += 1
    for _ in range(max_rounds):
        current = score(counts / total)
        best = current
        best_move: tuple[int, int] | None = None
        for source in range(len(points)):
            for target in range(len(points)):
                if not allowed_move(counts, points, source, target,
                                    min_fibre_mass, total):
                    continue
                trial = counts.copy()
                trial[source] -= 1
                trial[target] += 1
                value = score(trial / total)
                if value > best + 1e-12:
                    best = value
                    best_move = (source, target)
        if best_move is None:
            break
        source, target = best_move
        counts[source] -= 1
        counts[target] += 1
    return counts / total


def write_rows(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(frontier_out: str = "results/fibre_exposure_design_frontier.csv",
        summary_out: str = "results/fibre_exposure_design_summary.csv") -> None:
    points = profiles_3x3()
    total = 120

    # Ordinary candidate-index precision: no support constraint.  The optimum
    # concentrates on the extreme candidate-summary values and is fibre blind.
    d_weights = coordinate_exchange(
        points, total,
        lambda w: candidate_variance(points, w),
        min_fibre_mass=0.0,
    )
    d_summary = summaries(points, d_weights, "candidate_D_optimal")

    # Fibre-aware search: retain a minimum mass in every non-singleton fibre,
    # then maximize the log determinant of the residual exposure matrix.
    fibre_weights = coordinate_exchange(
        points, total,
        lambda w: fibre_logdet(exposure_matrix(points, w)),
        min_fibre_mass=0.10,
    )
    fibre_summary = summaries(points, fibre_weights, "fibre_logdet_optimal")

    # A small Pareto frontier makes the precision/exposure trade-off visible.
    frontier = []
    for lam in (0.0, 0.05, 0.10, 0.25, 0.50, 1.0):
        def score(w: np.ndarray, lam: float = lam) -> float:
            return fibre_logdet(exposure_matrix(points, w)) + lam * candidate_variance(points, w)
        weights = coordinate_exchange(points, total, score, min_fibre_mass=0.10)
        summary = summaries(points, weights, f"frontier_lambda_{lam:g}")
        frontier.append({
            "design": summary.name,
            "lambda_candidate_precision": lam,
            "candidate_variance_phi": summary.candidate_variance,
            "fibre_logdet": summary.fibre_logdet,
            "fibre_min_eigenvalue": summary.fibre_min_eigenvalue,
            "fibre_rank": summary.fibre_rank,
            "non_singleton_fibre_mass": summary.non_singleton_mass,
            "weights_by_profile": ";".join(
                f"({int(a)},{int(b)}):{weight:.5f}"
                for (a, b), weight in zip(points, summary.weights)
                if weight > 0
            ),
        })

    named = [d_summary, fibre_summary]
    summary_rows = []
    for summary in named:
        summary_rows.append({
            "design": summary.name,
            "candidate_variance_phi": summary.candidate_variance,
            "fibre_logdet": summary.fibre_logdet,
            "fibre_min_eigenvalue": summary.fibre_min_eigenvalue,
            "fibre_rank": summary.fibre_rank,
            "non_singleton_fibre_mass": summary.non_singleton_mass,
            "weights_by_profile": ";".join(
                f"({int(a)},{int(b)}):{weight:.5f}"
                for (a, b), weight in zip(points, summary.weights)
                if weight > 0
            ),
        })
    write_rows(frontier_out, frontier)
    write_rows(summary_out, summary_rows)
    for row in summary_rows:
        print(row)
    print("wrote", len(frontier), "frontier rows to", frontier_out)
    print("wrote", len(summary_rows), "summary rows to", summary_out)


if __name__ == "__main__":
    run()
