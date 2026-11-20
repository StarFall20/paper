"""Compute the unit-power existence boundary for two Gaussian tasks per person."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import erfcx, ndtr


def tail_square(a):
    """Expected (G-a)_+ squared for standard normal G and a >= 0."""
    density = np.exp(-a * a / 2) / np.sqrt(2 * np.pi)
    return (1 + a * a) * ndtr(-a) - a * density


def block_root(rho):
    mean = np.sqrt(1 + rho)
    difference = np.sqrt(1 - rho)
    return 0.5 * np.array([[mean + difference, mean - difference],
                           [mean - difference, mean + difference]])


def radial_third(b):
    """Integral of r^3 exp(-r^2/2-br) over the positive ray."""
    return 2 + b * b - (3 * b + b**3) * np.sqrt(np.pi / 2) * erfcx(b / np.sqrt(2))


def expected_distance_squared(scale, rho):
    if scale == 0:
        return 2.0
    if rho == 0:
        return 4 * tail_square(scale)
    if rho == 1:
        return 1 + 2 * tail_square(2 * scale)
    points = scale * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) @ block_root(rho)
    right = np.roll(points, -1, axis=0)
    edges = right - points
    tangents = edges / np.linalg.norm(edges, axis=1)[:, None]
    normals = np.column_stack((tangents[:, 1], -tangents[:, 0]))
    angles = np.arctan2(normals[:, 1], normals[:, 0])
    gaps = np.mod(angles - np.roll(angles, 1), 2 * np.pi)
    assert np.all((0 < gaps) & (gaps < np.pi))
    left_coordinate = np.sum(points * tangents, axis=1)
    right_coordinate = np.sum(right * tangents, axis=1)
    height = np.sum(points * normals, axis=1)
    distance = np.sum(tail_square(height) * (ndtr(right_coordinate) - ndtr(left_coordinate)))
    for vertex, angle, gap in zip(points, np.roll(angles, 1), gaps, strict=True):
        def along_direction(rotation):
            unit = np.array([np.cos(angle + rotation), np.sin(angle + rotation)])
            return radial_third(vertex @ unit)

        integral, error = quad(along_direction, 0, gap, epsabs=1e-11, epsrel=1e-11)
        assert error < 1e-9
        distance += np.exp(-0.5 * (vertex @ vertex)) * integral / (2 * np.pi)
    return float(distance)


def boundary(rho, auxiliary_ratio):
    def negative_value(scale):
        return 2 * (auxiliary_ratio * scale)**2 - auxiliary_ratio * (
            2 - expected_distance_squared(scale, rho)
        )

    result = minimize_scalar(
        negative_value, bounds=(0, auxiliary_ratio**-0.5),
        method="bounded", options={"xatol": 1e-13},
    )
    assert result.success and result.fun < 0
    return {"rho": rho, "auxiliary_ratio": auxiliary_ratio,
            "critical_slopes_per_labelled_person": -float(result.fun),
            "critical_slopes_per_labelled_task": -float(result.fun) / 2,
            "maximizing_scale": float(result.x)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    grid = [boundary(rho, ratio)
            for ratio in (0.5, 1.0, 2.0, 4.0)
            for rho in (0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 0.999, 1.0)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(grid, indent=2) + "\n")
    print(json.dumps([row for row in grid if row["auxiliary_ratio"] == 1], indent=2))


if __name__ == "__main__":
    main()
