"""Summarise candidate-term recovery for the mechanism benchmark."""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict


TERM_NAMES = (
    "quality_sq",
    "price_sq",
    "price_hinge",
    "quality_support",
    "evidence_digital",
    "price_income",
)

TRUE_TERMS = {
    "additive": set(),
    "nonlinear": {"quality_sq", "price_sq"},
    "threshold": {"price_hinge"},
    "interaction": {"quality_support", "evidence_digital"},
    "heterogeneity": set(),
    "nonlinear_threshold": {"quality_sq", "price_sq", "price_hinge"},
    "nonlinear_interaction": {"quality_sq", "price_sq", "quality_support", "evidence_digital"},
    "combined": {"quality_sq", "price_sq", "price_hinge", "quality_support", "evidence_digital"},
}


def parse_terms(value: str) -> set[str]:
    return {x for x in value.split("+") if x in TERM_NAMES}


def run(inp: str, out: str) -> None:
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    with open(inp, newline="") as f:
        for row in csv.DictReader(f):
            if row["model"] != "ML_assisted_spec":
                continue
            selected = parse_terms(row["selected_terms"])
            truth = TRUE_TERMS[row["condition"]]
            tp = len(selected & truth)
            fp = len(selected - truth)
            fn = len(truth - selected)
            grouped[row["condition"]].append({
                "selected_count": len(selected),
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "precision": tp / len(selected) if selected else (1.0 if not truth else 0.0),
                "recall": tp / len(truth) if truth else 1.0,
            })
    fields = ["condition", "replications", "true_term_count", "selected_count_mean",
              "true_positive_mean", "false_positive_mean", "false_negative_mean",
              "precision_mean", "recall_mean"]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for condition in sorted(grouped):
            rows = grouped[condition]
            mean = lambda key: sum(r[key] for r in rows) / len(rows)
            w.writerow({
                "condition": condition,
                "replications": len(rows),
                "true_term_count": len(TRUE_TERMS[condition]),
                "selected_count_mean": mean("selected_count"),
                "true_positive_mean": mean("true_positive"),
                "false_positive_mean": mean("false_positive"),
                "false_negative_mean": mean("false_negative"),
                "precision_mean": mean("precision"),
                "recall_mean": mean("recall"),
            })
    print(f"wrote specification recovery summary to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/assisted_spec_results_corrected_30rep.csv")
    ap.add_argument("--out", default="results/assisted_spec_recovery_corrected_30rep.csv")
    args = ap.parse_args()
    run(args.input, args.out)
