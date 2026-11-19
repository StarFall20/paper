"""Verify the pinned PPI++ replay against the frozen probe plan and prior release."""

import argparse
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(records_path, prior_path):
    records = load(records_path)
    prior = load(prior_path)
    index = {(r["case"], r["api"], r["requested_power"], r["one_step"]): r for r in records}
    assert len(records) == len(index) == 136
    for row in prior:
        new = index[(row["case"], "point", row["requested_power"], False)]
        assert new["result"] == row.get("beta")
        assert new["exception"] == row["exception"]
        assert new["returned_finite"] == row["return_finite_array"]
    assert len(prior) == 68
    groups = {}
    for state in ("outside", "interior"):
        names = [r["case"] for r in prior if r["kind"] == "public"
                 and r["requested_power"] == 1 and r["support"]["state"] == state]
        groups[state] = names
        assert len(names) == len(set(names)) == 8
        for name in names:
            fixed = index[name, "point", 1, False]
            assert fixed["returned_finite"] and fixed["support"] == state
            assert fixed["optimizer_trace"][-1]["success"] == (state == "interior")
            if state == "outside":
                keys = [("point", None, False), ("point", None, True),
                        ("ci", 1, False), ("ci", None, False), ("ci", None, True)]
                for key in keys:
                    r = index[(name, *key)]
                    assert r["exception"]["type"] == "LinAlgError" and not r["power_trace"]
            else:
                p = index[name, "point", None, True]
                ci = index[name, "ci", None, True]
                assert p["returned_finite"] and ci["returned_finite"]
                assert len(p["optimizer_trace"]) == 1 and len(ci["optimizer_trace"]) == 3
                assert p["power_trace"][-1]["clip"] and not ci["power_trace"][-1]["clip"]
                assert ci["interval_center_minus_final_optimizer_max"] < 1e-12
    controls = groups["interior"]
    point_steps = [index[name, "point", None, True] for name in controls]
    ci_steps = [index[name, "ci", None, True] for name in controls]
    ci_weights = [r["effective_power"] for r in ci_steps]
    assert sum(x < 0 for x in ci_weights) == 6
    exact_boundary = index["matched_reference_0.55", "point", 0.55, False]
    exact_boundary_ci = index["matched_reference_0.55", "ci", 0.55, False]
    assert exact_boundary["support"] == "boundary_or_near_boundary"
    assert exact_boundary["optimizer_trace"][-1]["success"] and exact_boundary["score_max"] < 6e-15
    assert exact_boundary_ci["returned_finite"]
    lo, hi = (row[1] for row in exact_boundary_ci["result"])
    return {
        "api_calls": len(records),
        "matched_default_calls": len(prior),
        "preselected_unit_exterior_cases": 8,
        "preselected_unit_interior_controls": 8,
        "unit_exterior_finite_point_arrays": 8,
        "unit_exterior_successful_optimizations": 0,
        "unit_exterior_automatic_and_one_step_point_singular_each": 8,
        "unit_exterior_fixed_automatic_and_one_step_ci_singular_each": 8,
        "independently_reconstructed_one_step_points": len(point_steps),
        "maximum_one_step_reconstruction_error": max(r["independent_one_step_error"] for r in point_steps),
        "one_step_score_range": [min(r["score_max"] for r in point_steps), max(r["score_max"] for r in point_steps)],
        "one_step_ci_with_full_final_optimization": len(ci_steps),
        "one_step_ci_negative_power_controls": sum(x < 0 for x in ci_weights),
        "one_step_ci_power_range": [min(ci_weights), max(ci_weights)],
        "maximum_one_step_ci_center_error": max(r["interval_center_minus_final_optimizer_max"] for r in ci_steps),
        "analytic_boundary": {"exact_power": "11/20", "reported_slope": exact_boundary["result"][1],
                              "reported_score_max": exact_boundary["score_max"],
                              "slope_ci": [lo, hi], "slope_ci_width": hi - lo,
                              "rational_objective_infimum_attained": False},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--expected", type=Path)
    args = parser.parse_args()
    summary = verify(args.records, args.prior)
    if args.expected is not None:
        assert summary == load(args.expected)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
