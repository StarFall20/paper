"""Projection expectations for a fixed positive correction weight in task blocks."""
import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import ndtr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "paired_choice_boundary_20261120"))
from compute_boundary import block_root, boundary as unit_boundary, tail_square


def distance(half_width, center, root):
    """E squared distance from standard Gaussian to a translated parallelogram."""
    if half_width == 0:
        return 2 + center @ center
    vertices = half_width * np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]]) @ root + center
    vertices_next = np.roll(vertices, -1, axis=0)
    edges = vertices_next - vertices
    tangents = edges / np.linalg.norm(edges, axis=1)[:, None]
    normals = np.column_stack((tangents[:, 1], -tangents[:, 0]))
    angles = np.arctan2(normals[:, 1], normals[:, 0])
    arcs = np.mod(angles - np.roll(angles, 1), 2*np.pi)

    start = np.sum(vertices * tangents, axis=1)
    end = np.sum(vertices_next * tangents, axis=1)
    height = np.sum(vertices * normals, axis=1)
    total = float(tail_square(height) @ (ndtr(end)-ndtr(start)))
    for vertex, angle, arc in zip(vertices, np.roll(angles, 1), arcs, strict=True):
        norms = vertex @ vertex

        def integrand(rotation):
            b = vertex @ np.array([np.cos(angle + rotation), np.sin(angle + rotation)])
            # Combine the radial Gaussian integral with its vertex density.
            # This stays finite for a vertex far from the origin.
            scaled = np.exp(-max(0., norms-b*b)/2)
            return (2+b*b)*np.exp(-norms/2) - (3*b+b**3)*np.sqrt(2*np.pi)*scaled*ndtr(-b)

        area, error = quad(integrand, 0., arc, epsabs=1e-10, epsrel=1e-11)
        assert error < 1e-9*max(1.,abs(area))
        total += area/(2*np.pi)
    return total


def score(u, rho, lam, tau):
    root = block_root(rho)
    widths = u*(1-lam), u*lam/tau
    shift = np.array([[1., 1.], [1., -1.]]) @ root
    # Negating a box center preserves its expected distance by normal symmetry.
    labelled = np.mean([distance(widths[0], u*row, root) for row in shift])
    auxiliary = distance(widths[1], np.zeros(2), root)
    return float(2-labelled+tau*(2-auxiliary))


def independent_score(u, lam, tau):
    # With R=I, labelled intervals have endpoints at u*lam and u*(2-lam).
    return float(2*(1 - tail_square(-u*lam) - tail_square(u*(2-lam)))
                 + 2*tau*(1-2*tail_square(u*lam/tau)))


def boundary(rho, lam, tau):
    assert 0 <= rho < 1 and 0 < lam <= 1 and tau > 0
    max_scale = np.sqrt(4*(1+tau)/(lam**2*(1-rho)))
    fit = minimize_scalar(
        lambda u: -score(u,rho,lam,tau), method="bounded",
        bounds=(0., max_scale), options={"xatol": 1e-11},
    )
    assert fit.success and fit.fun < 0
    if rho == 0:
        for u in np.r_[np.linspace(0, 4, 9), fit.x]:
            assert abs(score(u,rho,lam,tau)-independent_score(u,lam,tau)) < 1e-9
    if lam == 1:
        assert abs(-fit.fun-unit_boundary(rho,tau)["critical_slopes_per_labelled_person"]) < 1e-9
    return dict(correlation=rho, power=lam, auxiliary_ratio=tau,
                critical_slopes_per_labelled_person=-float(fit.fun), maximizing_scale=float(fit.x))


def run(out):
    grid = json.loads((HERE/"plan.json").read_text())["boundary_grid"]
    values = [boundary(rho, lam, grid["auxiliary_ratio"])
              for lam, rho in itertools.product(grid["power"],grid["correlation"])]
    out.write_text(json.dumps(values,indent=2)+"\n")
    for r in values:
        print(f"{r['power']:.2f} {r['correlation']:.2f} {r['critical_slopes_per_labelled_person']:.8f}")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    run(parser.parse_args().out)
