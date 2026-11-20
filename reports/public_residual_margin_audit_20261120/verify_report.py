"""Independently validate the policy ledger and its published summaries."""

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
import math
from pathlib import Path


def read_table(path):
    reader = gzip.open if path.suffix == ".gz" else open
    with reader(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def mean(values):
    return math.fsum(values) / len(values)


def quantile(values, q):
    ordered = sorted(values)
    index = q * (len(values) - 1)
    lo = math.floor(index)
    hi = math.ceil(index)
    return ordered[lo] + (index - lo) * (ordered[hi] - ordered[lo])


def verify(root, scheme):
    keys = ("trial", "label_rows", "prediction")
    metrics = ("primal_dual_gap", "dual_error", "primal_error", "witness_error", "max_direct_score")
    configs = read_table(root / "frontier_checks.csv.gz")
    support = {tuple(row[k] for k in keys): row for row in configs}
    assert len(configs) == len(support) == 2000
    assert all(abs(float(row[k])) < 1e-8 for row in configs for k in metrics)
    ledger = read_table(root / "policy_rows.csv.gz")
    assert len(ledger) == 12000
    assert all(
        n == 1 for n in Counter(tuple(row[k] for k in (*keys, "method")) for row in ledger).values()
    )
    base = {tuple(row[k] for k in keys): row for row in ledger if row["method"] == "all_soft"}
    group = defaultdict(list)
    for row in ledger:
        key = tuple(row[k] for k in keys)
        end = float(support[key]["endpoint"])
        method = row["method"]
        assert float(row["endpoint"]) == end
        multiplier = {
            "all_soft": 0.0, "human_and_soft": int(key[1]) / 1200.0,
            "fixed_quarter": .25, "fixed_half": .5, "full": 1.0,
            "protected_half_margin": min(1.0, .5 * end),
        }[method]
        assert abs(float(row["multiplier"]) - multiplier) < 1e-12
        loss, brier, ridge = (float(row[k]) for k in ("logloss", "brier", "ridge_logloss"))
        soft = float(base[key]["logloss"])
        assert math.isfinite(loss) and 0 <= brier <= 1
        if row["status"] == "outside_all_soft_fallback":
            assert multiplier > end and row["logloss"] == base[key]["logloss"]
            assert row["brier"] == base[key]["brier"]
        else:
            assert row["status"] == "finite_root" and multiplier < end
        assert abs(float(row["diff_soft"]) - (loss - soft)) < 1e-12
        assert abs(float(row["diff_ridge"]) - (loss - ridge)) < 1e-12
        group[(key[1], key[2], method)].append((loss, brier, loss - soft, loss - ridge,
                                               row["status"], multiplier))
    summary = read_table(root / "summary.csv")
    assert len(summary) == len(group) == 24
    for item in summary:
        values = group[(item["label_rows"], item["prediction"], item["method"])]
        loss, brier, relative, relative_ridge, statuses, multipliers = zip(*values)
        assert len(values) == 500
        expected = dict(
            cases=len(values), exterior=sum(x != "finite_root" for x in statuses),
            mean_loss=mean(loss), median_loss=quantile(loss, .5), p90_loss=quantile(loss, .9),
            mean_brier=mean(brier), mean_vs_soft=mean(relative), mean_vs_ridge=mean(relative_ridge),
            better_than_soft=sum(x < -1e-10 for x in relative),
            reduced_multiplier=sum(x < 1 for x in multipliers),
        )
        assert all(abs(float(item[key]) - value) < 1e-12 for key, value in expected.items())
    return dict(
        scheme=scheme, unique_configurations=len(ledger), checked_frontiers=len(configs),
        verified_summary_groups=len(summary),
        protected_fits=sum(row["method"] == "protected_half_margin" for row in ledger),
        fallback_configurations=sum(row["status"] == "outside_all_soft_fallback" for row in ledger),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps([verify(args.root / scheme, scheme) for scheme in ("row", "block")], indent=2))
