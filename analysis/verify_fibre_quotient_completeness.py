"""Reproducibility checks for the fibre quotient completeness certificate."""
from __future__ import annotations

import sys

import numpy as np

import structural_fibre_design as design


def main() -> None:
    pool, raw, matrix, _scales = design.prepare_pool()
    maximin = (201, 59, 557, 997, 764)
    algebraic = (39, 7, 112)

    for name, indices in (("maximin", maximin), ("algebraic", algebraic)):
        assert len(set(indices)) == len(indices)
        for index in indices:
            row = pool[index]
            assert tuple(row["basis_A"]) == tuple(design.candidate_basis(row["a_plus"]))
            assert tuple(row["basis_A"]) == tuple(design.candidate_basis(row["a_minus"]))
            assert tuple(row["basis_B"]) == tuple(design.candidate_basis(row["b_plus"]))
            assert tuple(row["basis_B"]) == tuple(design.candidate_basis(row["b_minus"]))
            assert design.raw_signature(row["a_plus"], row["b_plus"]) == \
                design.raw_signature(row["a_minus"], row["b_minus"])
        rank = int(np.linalg.matrix_rank(matrix[list(indices)]))
        assert rank == len(design.FEATURE_NAMES)
        eig = np.linalg.eigvalsh(matrix[list(indices)].T @ matrix[list(indices)])
        print(name, "rank", rank, "eigenvalues", [round(float(x), 6) for x in eig])

    full_rank = int(np.linalg.matrix_rank(matrix))
    assert full_rank == len(design.FEATURE_NAMES)
    print("pool", len(pool), "full_rank", full_rank, "status", "PASS")


if __name__ == "__main__":
    main()
