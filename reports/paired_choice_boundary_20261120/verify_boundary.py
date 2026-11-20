"""Check the critical contrast with analytic bounds and an independent integral."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import qmc


def projected_distance(absolute_normals, scale, rho):
    x, y = absolute_normals.T
    if rho == 1:
        return np.mean(np.maximum(x - 2 * scale, 0)**2 + y**2)
    a = scale * np.sqrt(2 * (1 + rho))
    b = scale * np.sqrt(2 * (1 - rho))
    # In the first quadrant the only exposed oblique edge joins (a,0) and (0,b).
    fraction = np.clip((a * (a - x) + b * y) / (a * a + b * b), 0, 1)
    squared = (x - a * (1 - fraction))**2 + (y - b * fraction)**2
    return np.mean(np.where(x / a + y / b > 1, squared, 0))


def analytic(a):
    pdf = np.exp(-a * a / 2) / np.sqrt(2 * np.pi)
    upper = ndtr(-a)
    function = 1 - 2 * ((1 + a**2) * upper - a * pdf)
    derivative = 4 * (pdf - a * upper)
    return function, derivative


def midpoint_bounds():
    lower = 2 * analytic(0.4)[0] - 2 * 0.4**2
    linear = analytic(0)[1] * np.sqrt(0.02)
    u = 1 / 3
    f, derivative = analytic(2 * u)
    value = -2 * u * u + f + linear * u
    slope = -4 * u + 2 * derivative + linear
    upper = value + slope**2 / 8
    assert upper < 0.585 < lower
    return {"rho_0_lower_bound": lower, "rho_099_upper_bound": upper, "test_ratio": 0.585}


def verify(rows):
    assert len(rows) == 32
    points = np.abs(ndtri(qmc.Sobol(2, scramble=True, seed=14).random_base2(19)))
    errors = []
    for row in rows:
        ratio = row["auxiliary_ratio"]
        scale = row["maximizing_scale"]
        rho = row["rho"]
        d2 = projected_distance(points, scale, rho)
        expected = ratio * (2 - d2) - 2 * (ratio * scale)**2
        errors.append(abs(expected - row["critical_slopes_per_labelled_person"]))
        assert row["critical_slopes_per_labelled_task"] * 2 == row["critical_slopes_per_labelled_person"]
    assert max(errors) < 0.00015
    return {
        "polygons_checked": len(rows),
        "independent_gaussian_points_per_polygon": len(points),
        "maximum_value_difference": max(errors),
        "finite_correlation_contrast": midpoint_bounds(),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--boundaries", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    result = verify(json.loads(a.boundaries.read_text(encoding="utf-8")))
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
