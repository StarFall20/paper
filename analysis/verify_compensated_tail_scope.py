"""Exact scope check for a qualitative row-type reduction of compensated logit.

This is a classical one-parameter counterexample used for attribution, not a
new theorem, estimator, phase experiment or criticism of an inapplicable prior
theorem. Fraction arithmetic supplies the reference; the existing support LP
is checked against it. No external data or downloaded research code is used.
"""
from pathlib import Path
from fractions import Fraction as F
import argparse
import csv
import hashlib
import json
import math


def softplus(u):
    return max(u, 0.0) + math.log1p(math.exp(-abs(u)))


def run(out):
    import numpy as np
    from power_tuned_choice_audit import objective, support

    out.mkdir(parents=True, exist_ok=False)
    levels = [F(1, 8), F(1, 4), F(1, 2), F(3, 4), F(7, 8)]
    powers = levels
    expected_lp = {
        'finite': 'interior',
        'boundary': 'boundary_or_near_boundary',
        'exterior': 'outside_mean_support',
    }
    rows = []
    max_value_error = max_score_error = max_recession_error = 0.0
    logits = [-100., -10., -2., -.7, 0., .7, 2., 10., 100.]
    for y in [0, 1]:
        for power in powers:
            for gl in levels:
                for gu in levels:
                    moment = F(y) + power * (gu - gl)
                    state = ('finite' if 0 < moment < 1 else
                             'boundary' if moment in [0, 1] else 'exterior')
                    # Exact signs specify identical row types within each y.
                    labelled_slopes = [power * gl - y,
                                       1 - power + power * gl - y]
                    auxiliary_slopes = [-power * gu, power * (1 - gu)]
                    assert all(s > 0 for s in labelled_slopes) if y == 0 else all(s < 0 for s in labelled_slopes)
                    assert auxiliary_slopes[0] < 0 < auxiliary_slopes[1]
                    x = np.ones((2, 1))
                    yy = np.array([float(y), 0.])
                    gg = np.array([float(gl), float(gu)])
                    loss = objective(x, yy, gg, 1, float(power))
                    lp = support(loss)
                    assert lp['status'] == expected_lp[state], (y, power, gl, gu, lp)
                    for u in logits:
                        direct = ((1 - float(power)) * softplus(u)
                                  - (y - float(power * gl)) * u
                                  + float(power) * (softplus(u) - float(gu) * u))
                        closed = softplus(u) - float(moment) * u
                        implemented = loss.value(np.array([u]))[0]
                        max_value_error = max(max_value_error, abs(direct - closed),
                                              abs(implemented - closed))
                    root = score = information = None
                    if state == 'finite':
                        root = math.log(float(moment / (1 - moment)))
                        score = float(loss.value(np.array([root]))[1][0])
                        information = float(loss.hessian(np.array([root]))[0, 0])
                        assert information > 0
                        max_score_error = max(max_score_error, abs(score))
                    # Recession costs for +/-1 are exact fractions.
                    tail_left, tail_right = moment, 1 - moment
                    if state == 'exterior':
                        d = -1 if moment < 0 else 1
                        reference_cost = tail_left if d < 0 else tail_right
                        actual_cost = (float(loss.weights @ np.maximum(x[:, 0] * d, 0))
                                       - float(loss.moment[0]) * d)
                        assert reference_cost < 0
                        max_recession_error = max(max_recession_error,
                                                  abs(actual_cost - float(reference_cost)))
                    rows.append(dict(
                        human_y=y, power=str(power), labelled_prediction=str(gl),
                        auxiliary_prediction=str(gu), corrected_moment=str(moment),
                        labelled_row_type='increasing' if y == 0 else 'decreasing',
                        auxiliary_row_type='coercive', exact_state=state,
                        lp_state=lp['status'], root=root, root_score=score,
                        root_information=information,
                        recession_minus_one=str(tail_left),
                        recession_plus_one=str(tail_right)))
    assert len(rows) == 250
    counts = {s: sum(r['exact_state'] == s for r in rows)
              for s in expected_lp}
    assert counts == {'finite': 100, 'boundary': 50, 'exterior': 100}
    assert max_value_error < 1e-12 and max_score_error < 1e-14
    assert max_recession_error < 1e-14
    with (out / 'exact_scope_cases.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    base = Path(__file__).resolve().parent
    sources = {name: hashlib.sha256((base / name).read_bytes()).hexdigest()
               for name in [Path(__file__).name, 'power_tuned_choice_audit.py']}
    report = dict(
        date='2026-11-15', state='passed', exact_fraction_configurations=len(rows),
        exact_states=counts, support_lp_matches=len(rows),
        maximum_objective_identity_error=max_value_error,
        maximum_finite_root_score=max_score_error,
        maximum_exterior_recession_error=max_recession_error,
        source_hashes=sources, new_observations=0, random_model_draws=0,
        prior_theorem_refuted=False, new_originality_proof=False,
        meaning=('Qualitative row types do not determine aggregate attainment '
                 'when labelled row infima are minus infinity. The existing '
                 'theorem explicitly assumes finite row infima.'))
    (out / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
