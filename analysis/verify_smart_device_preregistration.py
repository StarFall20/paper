"""Verify the frozen smart-device parity design invariants."""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(__file__)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import smart_device_fibre_candidate_grid as grid
import smart_device_fibre_design as base
import smart_device_nuisance_balanced_design as balanced


def main():
    candidates = grid.enumerate_candidates()
    pure = [row for row in candidates
            if abs(row["odd_gap_plus"] + row["odd_gap_minus"]) < 1e-9]
    balanced_pool = balanced.balanced_pool(candidates)
    tasks = balanced.choose_balanced(balanced_pool)
    assert len(tasks) == 5
    expected = [0.32, 0.40, 0.57, 0.80, 0.98]
    for task, target in zip(tasks, expected):
        assert abs(task["target_gap"] - target) < 1e-12
        assert abs(task["odd_gap_plus"] + task["odd_gap_minus"]) < 1e-9
        assert abs(task["gap"] - target) <= 0.06
        assert abs(task["distance"] - base.squared_distance(
            task["a_minus"], task["b_minus"])) < 1e-12
        assert balanced.is_balanced(task)
    gaps = [round(task["gap"], 2) for task in tasks]
    signals = [round(task["signal"], 2) for task in tasks]
    print("candidate_pool", len(candidates))
    print("pure_odd_pool", len(pure))
    print("nuisance_balanced_pool", len(balanced_pool))
    print("selected_gaps", gaps)
    print("selected_signals", signals)
    print("preregistration_invariants=PASS")


if __name__ == "__main__":
    main()
