"""Check complete Web of Science exports and produce a bibliographic screening record."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent
PRIORITIES = {
    "000514816000002": ("read", "Ordinary unpenalized binary logistic phase transition"),
    "000587719400043": ("read", "Ordinary binary-response GLM boundary with elliptical predictors"),
    "001500112200001": ("read", "Multinomial existence and positive class probabilities as regularization"),
    "001427598200003": ("read", "Outcome dependence on networks with a penalized pseudolikelihood"),
    "001715176900001": ("read", "Gaussian conic kinematics for explicitly constrained logistic MLE"),
    "001757410700001": ("read", "Positive prior correction and high-dimensional logistic inference"),
    "001760036100001": ("read", "LLM augmentation for conjoint preference estimators"),
    "001740302300001": ("read", "Real and GPT food choice tasks and preference comparison"),
    "001871091700001": ("read", "Synthetic respondents and non-market economic valuation"),
    "001734791100001": ("context", "LLM preference understanding in an apparel application"),
    "001793601000001": ("context", "LLM preference identification and recommendation in apparel"),
    "001917263100021": ("read", "Exact upstream correction objective and variance tuning"),
    "001138596500031": ("read", "Original prediction-powered inference"),
    "001802990800001": ("context", "Review of predictions, designs and finite-sample inference"),
    "001472745800001": ("read", "Human and LLM mixed-subject design with power planning"),
    "001207175200001": ("context", "Cross-fitting of prediction-assisted inference"),
    "001929891000001": ("context", "Semisupervised efficiency rather than optimizer existence"),
    "001683723600017": ("read", "Prediction-powered model fit testing with prediction comparisons"),
    "001898001900001": ("read", "Precision at K for LLM ranking, with human calibration"),
    "001781718900001": ("context", "Unbiased model-assisted point estimation"),
    "001609561600048": ("context", "Calibration of risk with predictions and labels"),
    "000755619400001": ("context", "High-dimensional binary linear classification risk"),
}


def fields(text):
    output = {}
    name = None
    for line in text.splitlines():
        if re.match(r"^[A-Z0-9]{2} ", line):
            name = line[:2]
            output.setdefault(name, []).append(line[3:])
        elif name is not None and line.startswith("   "):
            output[name].append(line.strip())
    return {key: " ".join(values).strip() for key, values in output.items()}


def screen(downloads, out):
    log = json.loads((ROOT / "search_log.json").read_text(encoding="utf-8"))
    records = []
    provenance = []
    for query in log["webOfScience"]:
        filename = query["download"]
        if filename is None:
            continue
        source = downloads / filename
        data = source.read_bytes()
        entries = [fields(part) for part in re.split(r"^ER\s*$", data.decode("utf-8-sig"), flags=re.M)]
        entries = [part for part in entries if "UT" in part]
        assert len(entries) == query["total"], (filename, len(entries))
        provenance.append(dict(query=query["id"], filename=filename, count=len(entries),
                               sha256=hashlib.sha256(data).hexdigest()))
        for number, part in enumerate(entries, 1):
            uid = part["UT"].removeprefix("WOS:")
            decision, rationale = PRIORITIES.get(uid, ("exclude", {
                "W03": "Different model class, regularity problem or unrelated use of phase language",
                "W05": "Application context or choice terminology without corrected-fit existence",
                "W07": "Different PPI target, application or inferential objective",
            }[query["id"]]))
            records.append(dict(query=query["id"], number=number, wos_id=part["UT"], title=part["TI"],
                                journal=part.get("SO", ""), year=part.get("PY", ""), doi=part.get("DI", ""),
                                abstract_available="AB" in part, screening=decision, rationale=rationale))
    assert len(records) == len({r["wos_id"] for r in records}) == 71
    assert len({r["doi"] for r in records if r["doi"]}) == sum(bool(r["doi"]) for r in records)
    assert set(PRIORITIES) <= {r["wos_id"].removeprefix("WOS:") for r in records}
    out.mkdir(parents=True, exist_ok=True)
    with (out / "screened_wos_records.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    overview = {
        "institutional_topic_queries": len(log["webOfScience"]),
        "records_in_three_complete_exports": len(records),
        "additional_screened_single_hit": 1,
        "records_with_abstract_in_export": sum(r["abstract_available"] for r in records),
        "screening": {state: sum(r["screening"] == state for r in records)
                      for state in ("read", "context", "exclude")},
        "export_provenance": provenance,
        "scope": "Full title lists and selected abstracts or primary results. Read marks priority, with reading level reported separately.",
    }
    (out / "screening_summary.json").write_text(json.dumps(overview, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(overview))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    screen(args.download_dir, args.out)
