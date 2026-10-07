"""Sensitivity gate for a policy-weighted fibre exposure criterion.

The policy weight is a declared prior over residual directions induced by a
policy-relevant profile distribution.  The script tests whether it changes
the selected support relative to structural log-determinant exposure.  A
change is useful only if the policy distribution is fixed before outcomes and
has a substantive interpretation; otherwise the extension stays secondary.
"""
from __future__ import annotations

import csv
import os

import numpy as np

from fibre_exposure_design_optimizer import (
    candidate_variance,
    coordinate_exchange,
    exposure_matrix,
    fibre_logdet,
    profiles_3x3,
    residual_dictionary,
    summaries,
)


def policy_matrix(points: np.ndarray, policy_weights: np.ndarray) -> np.ndarray:
    features = residual_dictionary(points)
    return (features.T * policy_weights) @ features


def policy_trace(points: np.ndarray, design_weights: np.ndarray,
                 policy_weights: np.ndarray) -> float:
    return float(np.trace(policy_matrix(points, policy_weights)
                          @ exposure_matrix(points, design_weights)))


def support_string(points: np.ndarray, weights: np.ndarray) -> str:
    return ";".join(
        f"({int(a)},{int(b)}):{weight:.5f}"
        for (a, b), weight in zip(points, weights)
        if weight > 0
    )


def run(out: str = "results/fibre_policy_weighted_design.csv") -> None:
    points = profiles_3x3()
    total = 120
    uniform = np.full(len(points), 1.0 / len(points))
    interaction = np.zeros(len(points))
    interaction[[4, 2, 6]] = [0.70, 0.15, 0.15]  # (1,1), (0,2), (2,0)
    decomposition = np.zeros(len(points))
    decomposition[[2, 6]] = [0.50, 0.50]  # (0,2), (2,0)
    policies = {
        "uniform_profile_policy": uniform,
        "interaction_policy": interaction,
        "decomposition_policy": decomposition,
    }

    rows = []
    for name, policy in policies.items():
        matrix = policy_matrix(points, policy)
        weights = coordinate_exchange(
            points,
            total,
            lambda w, p=policy: policy_trace(points, w, p),
            min_fibre_mass=0.10,
        )
        summary = summaries(points, weights, f"policy_{name}")
        rows.append({
            "policy": name,
            "policy_matrix_rank": int(np.linalg.matrix_rank(matrix, tol=1e-10)),
            "candidate_variance_phi": summary.candidate_variance,
            "structural_fibre_logdet": summary.fibre_logdet,
            "structural_fibre_rank": summary.fibre_rank,
            "structural_min_eigenvalue": summary.fibre_min_eigenvalue,
            "policy_weighted_exposure": policy_trace(points, weights, policy),
            "weights_by_profile": support_string(points, weights),
        })

    # Structural reference, using the same 10% non-singleton-fibre floor.
    structural = coordinate_exchange(
        points,
        total,
        lambda w: fibre_logdet(exposure_matrix(points, w)),
        min_fibre_mass=0.10,
    )
    structural_summary = summaries(points, structural, "structural_reference")
    rows.append({
        "policy": "structural_logdet_reference",
        "policy_matrix_rank": "",
        "candidate_variance_phi": structural_summary.candidate_variance,
        "structural_fibre_logdet": structural_summary.fibre_logdet,
        "structural_fibre_rank": structural_summary.fibre_rank,
        "structural_min_eigenvalue": structural_summary.fibre_min_eigenvalue,
        "policy_weighted_exposure": "",
        "weights_by_profile": support_string(points, structural),
    })

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print("wrote", len(rows), "rows to", out)


if __name__ == "__main__":
    run()
