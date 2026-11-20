"""Build a manuscript note from the verified numerical results."""

import argparse
from pathlib import Path
import json

from run import HERE, OUTSIDE


def write(args):
    results = json.loads((args.results / "results.json").read_text())
    fronts = json.loads((args.results / "frontiers.json").read_text())
    matches = json.loads((args.results / "matched_derivatives.json").read_text())
    rows = results["groups"]
    assert fronts["path_count"] == matches["witness_allocations"] == len(results["witnesses"])
    assert len(rows) == 24 and results["evaluated_objectives"] == 16128
    description = (
        "### Frequency changes in released vaccine choices\n\n"
        "The public vaccine choice archive of Lu et al. (2026) supplies a second design library. "
        f"The 1,200 auxiliary rows contain {results['exact_ordered_full_menus']:,} distinct ordered menus. "
        "All menus retain positive integer frequencies in each of the seven designs, and the auxiliary total "
        "is always 4,800. The original design repeats each row four times. Two designs allocate additional rows "
        "by multinomial draws. Four others upweight the top or bottom fifth of rows by forecast probability "
        "at total variation distances 0.20 and 0.60 from the original row frequencies. The fits use 11 utility "
        "contrasts. Both ridge logit predictors use a separate 1,200-row training block, one with human "
        "choices and the other with archived AI choices as the training outcome. The same fitted probability function "
        "is used for both sample roles and is held fixed across the seven designs.\n\n"
        "The following counts use 96 nested allocations from a disjoint label pool. The denominator retains "
        "allocations with full-rank finite ordinary Logit. An allocation crosses if some pair of its seven "
        "designs includes a finite estimate and an objective unbounded below. The label counts refer to choice "
        "records. The archive has no confirmed respondent identifiers.\n\n"
    )
    header = ("Table J1  Allocations that cross the finite-estimation boundary by correction weight\n\n"
              "| Predictor training outcome | Label records | Eligible | 0.75 | 0.90 | 0.95 | 1.00 |\n"
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    body = []
    for name, title in (("human_ridge_logit", "Human choices"), ("archived_AI_ridge_logit", "Archived AI choices")):
        for budget in (25, 50, 150):
            group = [r for r in rows if r["forecast"] == name and r["budget"] == budget]
            assert [item["weight"] for item in group] == [.75, .9, .95, 1]
            assert len({item["eligible"] for item in group}) == 1
            cells = [title, str(budget), str(group[0]["eligible"])] + [str(r["opposite_groups"]) for r in group]
            body.append("| " + " | ".join(cells) + " |")

    proof = (HERE / "frequency_path.md").read_text().rstrip()
    extremes = [r for r in fronts["paths"] if abs(r["weight"] - .75) < 1e-10 and r["budget"] == 50]
    if not extremes:
        extremes = fronts["paths"]
    example = min(extremes, key=lambda item: item["total_variation_from_finite_at_frontier"])
    example_title = ("human-trained" if example["forecast"] == "human_ridge_logit" else "AI-trained")
    descriptions = {"original": "original frequencies", "low_strong": "strongly upweighted lowest quintile",
                    "high_strong": "strongly upweighted highest quintile",
                    "low_moderate": "moderately upweighted lowest quintile",
                    "high_moderate": "moderately upweighted highest quintile",
                    "multinomial_1": "first multinomial frequencies",
                    "multinomial_2": "second multinomial frequencies"}
    example_from = descriptions[example["from_scenario"]]
    example_to = descriptions[example["to_scenario"]]
    show_front = (
        f"An example uses {example['budget']} labelled records, the {example_title} predictor and a correction "
        f"weight of {example['weight']:.2f}. Moving from the {example_from} toward the {example_to} "
        f"gives a last feasible transfer of {example['maximum_feasible_transfer']:.3f}, at total variation "
        f"distance {example['total_variation_from_finite_at_frontier']:.3f} from the finite design. "
        f"Expanded-row calculations and two probability formulations verify all {fronts['path_count']} "
        "selected endpoint pairs. The frequency program also verifies an interior and an exterior "
        "point on either side of each computed boundary."
    )
    successes = [r for r in matches["rows"] if r["final_state"] == OUTSIDE]
    within = sum(r["total_variation_bound"] == .2 for r in successes)
    focal = [r for r in successes if r["budget"] == 50 and r["weight"] == .75
             and r["total_variation_bound"] == .2]
    assert len({r["forecast"] for r in focal}) == 2
    mantissa, exponent = f"{matches['max_derivative_error']:.2e}".split("e")
    local = (
        "A separate exploratory calculation tests whether local derivatives can be held fixed on these same "
        "released menus. Starting from each selected finite design, choose positive fractional frequencies that "
        "match the objective value, gradient and Hessian at zero. The constraints preserve the score and the "
        "second moment of the utility contrasts. The search minimizes the recession cost in the direction "
        "recovered from the paired exterior design. It tests two total variation budgets, 0.20 and 0.60. "
        f"Of {matches['audited_scenarios']} candidates, {len(successes)} are certified exterior, including "
        f"{within} at budget 0.20. The latter cases include both predictors at 50 label records and "
        "correction weight 0.75. The largest numerical discrepancy among the matched derivatives is "
        f"${mantissa}\\times10^{{{int(exponent)}}}$. These frequencies are designed from the sample and describe "
        "what can happen within this menu library. The crossing counts describe the stated allocation and "
        "frequency rules for this archive."
    )
    note = description + header + "\n".join(body) + "\n\n" + proof + "\n\n" + show_front + "\n\n" + local + "\n"
    args.out.write_text(note)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    write(parser.parse_args())
