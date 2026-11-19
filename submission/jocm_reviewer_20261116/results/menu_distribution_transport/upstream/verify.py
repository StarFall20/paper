"""Recalculate each recorded score and held-out loss from the pinned sources."""
import argparse
import json
from collections import Counter
from itertools import groupby
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from source_data import cases, dependencies, read, sha


COMPARATORS = ('labelled_MLE', 'labelled_ridge', 'prompt_matched_MLE',
               'prompt_matched_ridge', 'raw_LLM', 'fixed_unit', 'released_rule')
METHODS = set(COMPARATORS) | {'safe_half'}
POLICIES = {'safe_half', 'fixed_unit', 'released_rule'}


def probabilities(x, beta):
    utility = x @ beta
    utility -= utility.max(axis=1, keepdims=True)
    logp = utility - np.log(np.exp(utility).sum(axis=1, keepdims=True))
    return np.exp(logp), logp


def moment(x, values):
    return (x * values[:, :, None]).sum(axis=1).mean(axis=0)


def score(row, case, beta):
    l, y, gl, _ = case.label
    u, gu, _ = case.aux
    method, state = row['method'], row['state']
    if method.startswith('prompt_matched'):
        l = np.concatenate((l, case.prompt[0]))
        y = np.concatenate((y, case.prompt[1]))
    pl = probabilities(l, beta)[0]
    gradient = moment(l, pl-y)
    if method in POLICIES and state == 'native':
        pu = probabilities(u, beta)[0]
        gradient += row['power'] * (moment(u, pu-gu) - moment(l, pl-gl))
    if method.endswith('ridge') or state.endswith('_ridge_fallback'):
        gradient += .01 * beta
    return float(np.linalg.norm(gradient, ord=np.inf))


def verify_case(case, data, plan):
    assert set(data) == METHODS
    x, y, raw, people = case.eval
    for method, row in data.items():
        assert row['prompt_people'] == case.prompt_people and row['prompt_menus'] == case.prompt_menus
        state = row['state']
        if state == 'unavailable':
            assert method == 'labelled_MLE' and all(row[k] is None for k in ('beta', 'loss', 'brier', 'score'))
            continue
        if method == 'raw_LLM':
            assert state == 'frozen' and row['beta'] is None
            p, lp = raw, np.log(raw)
        else:
            beta = np.asarray(row['beta'])
            assert np.isfinite(beta).all()
            if state.endswith('_ridge_fallback'):
                key = 'labelled_ridge' if state == 'labelled_ridge_fallback' else 'prompt_matched_ridge'
                assert row['beta'] == data[key]['beta'] and row['reason']
            else:
                assert state == 'native' and row['reason'] is None
            error = score(row, case, beta)
            assert error <= plan['tolerance']['score'] and abs(error - row['score']) < 1e-12
            p, lp = probabilities(x, beta)
        losses = -lp[y.astype(bool)]
        brier = np.square(p-y).sum(axis=1)
        for key, metric in [('loss', losses), ('brier', brier)]:
            respondent_loss = [metric[people == person].sum() / 6 for person in np.unique(people)]
            assert abs(np.mean(respondent_loss) - row[key]) < 1e-11


def describe(rows, tolerance):
    result = {'cases': len(rows), 'states': {}, 'power': {}, 'paired': {}}
    for name in sorted(METHODS):
        result['states'][name] = dict(Counter(r[name]['state'] for r in rows))
    for name in sorted(POLICIES):
        result['power'][name] = spread([r[name]['power'] for r in rows])
    result['prompt_menus'] = spread([r['safe_half']['prompt_menus'] for r in rows])
    result['prompt_people'] = spread([r['safe_half']['prompt_people'] for r in rows])
    for name in COMPARATORS:
        for metric in ('loss', 'brier'):
            delta = np.asarray([r['safe_half'][metric] - r[name][metric] for r in rows if r[name][metric] is not None])
            result['paired'].setdefault(name, {})[metric] = dict(
                n=len(delta), wins=int((delta < -tolerance).sum()), ties=int((abs(delta) <= tolerance).sum()),
                mean=float(delta.mean()), median=float(np.median(delta)))
    result['unit_unavailable_safe_native'] = sum(r['fixed_unit']['state'] != 'native' and r['safe_half']['state'] == 'native' for r in rows)
    result['released_unavailable_safe_native'] = sum(r['released_rule']['state'] != 'native' and r['safe_half']['state'] == 'native' for r in rows)
    result['fallback_reasons'] = {
        method: dict(Counter(r[method]['reason'] for r in rows if r[method]['reason'])) for method in sorted(METHODS)
    }
    return result


def spread(values):
    return [float(min(values)), float(np.median(values)), float(max(values))]


def main(args):
    plan, backend = dependencies(args.dependencies)
    fit_file = args.out / 'fits.json'
    rows = read(fit_file)
    columns = ('context', 'rep', 'coverage', 'label_people', 'label_fold', 'eval_fold')
    outcome = {}
    for key, entries in groupby(rows, key=lambda r: tuple(r[n] for n in columns)):
        original = list(entries)
        assert key not in outcome and len(original) == len(METHODS)
        outcome[key] = {r['method']: r for r in original}
    verified = {}
    for case in cases(args, plan, backend):
        assert case.key not in verified
        data = outcome[case.key]
        verify_case(case, data, plan)
        verified[case.key] = data
    assert set(verified) == set(outcome) and len(verified) == 480
    summaries = {}
    for budget, count in [(15, 120), (5, 360)]:
        rows = [v for k, v in verified.items() if k[3] == budget]
        assert len(rows) == count
        summaries[str(budget)] = describe(rows, plan['tolerance']['tie'])
    receipt = {'status': 'passed', 'scores_and_predictions': 3840, 'cases': 480,
               'fits_sha256': sha(fit_file), 'summaries': summaries}
    (args.out / 'summary.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 'dependencies', 'out'):
        cli.add_argument('--' + name, type=Path, required=True)
    with threadpool_limits(limits=1):
        main(cli.parse_args())
