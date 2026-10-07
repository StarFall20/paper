"""Support-complete fibre design for a finite candidate summary.

The usual exposure calculation uses a declared residual dictionary.  This
module removes that choice on a finite discrete support: within every
non-singleton fibre it spans the entire contrast space orthogonal to the
constant vector.  The resulting spectrum is a certificate for all
non-constant response departures on the declared support, not only for a few
hand-picked residual terms.
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np


def profiles_3x3() -> np.ndarray:
    return np.asarray([(a, b) for a in range(3) for b in range(3)], dtype=float)


def candidate_values(points: np.ndarray) -> np.ndarray:
    return points.sum(axis=1).astype(int)


def fibre_contrast_basis(size: int) -> np.ndarray:
    """Orthonormal basis for vectors whose entries sum to zero."""
    if size < 2:
        return np.zeros((size, 0))
    centering = np.eye(size) - np.ones((size, size)) / size
    q, _, _ = np.linalg.svd(centering)
    return q[:, : size - 1]


def support_complete_exposure(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Exposure on the direct sum of all within-fibre contrast spaces."""
    fibres = candidate_values(points)
    blocks = []
    for value in sorted(set(fibres)):
        ids = np.flatnonzero(fibres == value)
        if len(ids) < 2:
            continue
        mass = float(weights[ids].sum())
        if mass <= 0:
            blocks.append(np.zeros((len(ids) - 1, len(ids) - 1)))
            continue
        conditional = weights[ids] / mass
        covariance = np.diag(conditional) - np.outer(conditional, conditional)
        q = fibre_contrast_basis(len(ids))
        blocks.append(mass * (q.T @ covariance @ q))
    if not blocks:
        return np.zeros((0, 0))
    return np.block([[block if i == j else np.zeros((block.shape[0], blocks[j].shape[0]))
                      for j, block_j in enumerate(blocks)]
                     for i, block in enumerate(blocks)])


def candidate_variance(points: np.ndarray, weights: np.ndarray) -> float:
    phi = candidate_values(points).astype(float)
    mean = float(weights @ phi)
    return float(weights @ (phi - mean) ** 2)


def rank_and_spectrum(matrix: np.ndarray) -> tuple[int, float, float]:
    if matrix.size == 0:
        return 0, 0.0, 0.0
    eigenvalues = np.linalg.eigvalsh(matrix)
    rank = int(np.sum(eigenvalues > 1e-10))
    positive = eigenvalues[eigenvalues > 1e-10]
    return rank, float(positive.min()) if len(positive) else 0.0, float(eigenvalues.sum())


def allowed_move(counts: np.ndarray, fibres: np.ndarray, source: int,
                 target: int, min_profile: int) -> bool:
    if source == target or counts[source] <= min_profile:
        return False
    trial = counts.copy()
    trial[source] -= 1
    trial[target] += 1
    # A support-complete certificate needs every profile in each non-singleton
    # fibre to retain positive mass.
    for value in sorted(set(fibres)):
        ids = np.flatnonzero(fibres == value)
        if len(ids) >= 2 and np.any(trial[ids] <= min_profile):
            return False
    return True


def coordinate_exchange(points: np.ndarray, total: int, score, min_profile: int = 1,
                        max_rounds: int = 500) -> np.ndarray:
    counts = np.ones(len(points), dtype=int) * min_profile
    counts += (total - counts.sum()) // len(points)
    for idx in range(total - counts.sum()):
        counts[idx % len(points)] += 1
    fibres = candidate_values(points)
    for _ in range(max_rounds):
        current = score(counts / total)
        best = current
        move = None
        for source in range(len(points)):
            for target in range(len(points)):
                if not allowed_move(counts, fibres, source, target, min_profile):
                    continue
                trial = counts.copy(); trial[source] -= 1; trial[target] += 1
                value = score(trial / total)
                if value > best + 1e-12:
                    best = value; move = (source, target)
        if move is None:
            break
        source, target = move
        counts[source] -= 1; counts[target] += 1
    return counts / total


def summary(points: np.ndarray, weights: np.ndarray, name: str) -> dict:
    matrix = support_complete_exposure(points, weights)
    rank, minimum, trace = rank_and_spectrum(matrix)
    return {
        "design": name,
        "candidate_variance_phi": candidate_variance(points, weights),
        "support_complete_rank": rank,
        "support_complete_dimension": matrix.shape[0],
        "support_complete_min_eigenvalue": minimum,
        "support_complete_trace": trace,
        "weights_by_profile": ";".join(
            f"({int(a)},{int(b)}):{weight:.5f}"
            for (a, b), weight in zip(points, weights) if weight > 0
        ),
    }


def run(out: str = "results/support_complete_fibre_design.csv") -> None:
    points = profiles_3x3(); total = 180
    # The exact candidate-precision optimum puts mass on the two extreme
    # candidate-summary values.  It is deliberately fibre blind.
    ordinary = np.zeros(len(points), dtype=float)
    ordinary[0] = ordinary[-1] = 0.5
    complete = coordinate_exchange(
        points, total,
        lambda w: np.linalg.slogdet(
            support_complete_exposure(points, w) + 1e-10 * np.eye(4)
        )[1],
        min_profile=1,
    )
    rows = [summary(points, ordinary, "candidate_precision"),
            summary(points, complete, "support_complete")] 
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    for row in rows:
        print(row)
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/support_complete_fibre_design.csv")
    run(**vars(parser.parse_args()))
