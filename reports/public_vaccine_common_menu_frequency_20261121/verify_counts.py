"""Recount the stored states without relying on the fitting or summary code."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path


def verify(folder):
    summary = json.loads((folder / "results.json").read_text())
    with (folder / "fit_states.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    groups = {}
    for item in rows:
        key = (item["forecast"], int(item["budget"]), float(item["weight"]))
        record = groups.setdefault(key, {}).setdefault(int(item["allocation"]), {})
        assert item["scenario"] not in record
        record[item["scenario"]] = (item["ordinary"], item["state"])
    assert len(rows) == summary["evaluated_objectives"] == 2 * 3 * 4 * 96 * 7
    assert len(groups) == len(summary["groups"]) == 24
    for specification in summary["groups"]:
        key = (specification["forecast"], specification["budget"], specification["weight"])
        scenario_counts = {s: Counter() for s in specification["states"]}
        eligible, crossings, pairs, comparisons_to_original = 0, 0, 0, 0
        for record in groups[key].values():
            assert set(record) == set(scenario_counts)
            ordinary, = set(item[0] for item in record.values())
            if ordinary != "interior":
                continue
            eligible += 1
            count = Counter()
            for name, (_, state) in record.items():
                scenario_counts[name][state] += 1
                count[state] += 1
            opposite = count["interior"] * count["outside_mean_support"]
            crossings += opposite > 0
            pairs += opposite
            original = record["original"][1]
            if original == "interior":
                comparisons_to_original += count["outside_mean_support"]
            if original == "outside_mean_support":
                comparisons_to_original += count["interior"]
        assert scenario_counts == specification["states"]
        assert (eligible, crossings, pairs, comparisons_to_original) == (
            specification["eligible"], specification["opposite_groups"],
            specification["opposite_pairs"], specification["pairs_with_original"])
    certificates = Counter(r["certificate"] for r in rows if "certificate" in r)
    return {"objectives": len(rows), "specifications": len(groups),
            "witness_pairs": len(summary["witnesses"]), "certificates": dict(certificates)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_folder", type=Path)
    print(json.dumps(verify(parser.parse_args().result_folder)))
