"""Small algebra check for the candidate-preserving fibre proposition."""
from __future__ import annotations

import numpy as np


def pairwise_differences(basis: np.ndarray) -> np.ndarray:
    """Return all alternative-pair basis differences for shape (J, P)."""
    j = basis.shape[0]
    return np.stack(
        [basis[a] - basis[b] for a in range(j) for b in range(a + 1, j)],
        axis=0,
    )


def softmax(u: np.ndarray) -> np.ndarray:
    z = u - np.max(u)
    e = np.exp(z)
    return e / e.sum()


def main() -> None:
    rng = np.random.default_rng(20261006)
    base = rng.normal(size=(3, 4))
    common_shift = rng.normal(size=4)
    transformed = base + common_shift
    beta = rng.normal(size=4)

    assert np.allclose(pairwise_differences(base), pairwise_differences(transformed))
    assert np.allclose(softmax(base @ beta), softmax(transformed @ beta))

    changed = transformed.copy()
    changed[0, 2] += 0.4
    assert not np.allclose(pairwise_differences(base), pairwise_differences(changed))
    assert not np.allclose(softmax(base @ beta), softmax(changed @ beta))
    print("UFIT algebra checks passed")


if __name__ == "__main__":
    main()
