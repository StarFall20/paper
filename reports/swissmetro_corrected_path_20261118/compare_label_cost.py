"""Compare direct estimation using exactly the human labels consumed by correction."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

from audit import center_features, fit_likelihood, load_data, logloss, sha

ROOT = Path(__file__).resolve().parent


def read_rows(p):
    with p.open(newline='') as f:
        return list(csv.DictReader(f))


def compare(args):
    sys.path.insert(0, str(args.numerics.parent))
    from audit_path import objective
    plan = json.loads((ROOT/'plan.json').read_text())
    original = json.loads((args.results/'summary.json').read_text())
    assert sha(args.input) == original['source_sha256'] == plan['data_sha256']
    assert sha(args.numerics) == original['numerical_helper_sha256']
    data = read_rows(args.results/'allocations.csv')
    paths = read_rows(args.results/'paths.csv')
    for n, h in original['output_sha256'].items():
        assert sha(args.results/n) == h
    ids, y, av, designs = load_data(args.input)
    split = np.random.default_rng(plan['split_seed']).permutation(np.unique(ids))
    teacher, evaluation, pool = np.split(split, [600,840])
    teach = np.isin(ids,teacher)
    evaluate = np.isin(ids,evaluation)
    held_choices, held_av = y[evaluate], av[evaluate]
    orders = [np.random.default_rng(plan['allocation_seed'] + i).permutation(pool) for i in range(plan['allocations'])]
    kept = {(int(row['record']),float(row['fraction'])): float(row['holdout_loss']) for row in paths if row['fit']=='score_verified'}
    results = []
    for name, design in designs.items():
        features = center_features(design,av)
        test_features = features[evaluate]
        for allocation in (r for r in data if r['specification']==name):
            n, budget, trial = (int(allocation[k]) for k in ('record','labelled_people','allocation'))
            labels = teach | np.isin(ids,orders[trial][:budget])
            assert len(np.unique(ids[labels])) == 600+budget and not np.any(labels & evaluate)
            beta = fit_likelihood(objective,features[labels],av[labels],y[labels],0)
            risk = logloss(test_features,held_av,held_choices,beta)
            for fraction in (.5,.9):
                predicted_loss=kept.get((n,fraction))
                used_loss = float(allocation['ridge_loss']) if predicted_loss is None else predicted_loss
                results.append(dict(record=n,specification=name,budget=budget,individual_labels=600+budget,
                                    label_rows=int(labels.sum()),fraction=fraction,matched_direct_loss=risk,
                                    corrected_available=predicted_loss is not None, corrected_or_ridge_loss=used_loss,
                                    difference=used_loss-risk))
    assert len(results)==1152
    by_cell=defaultdict(list)
    for row in results:
        by_cell[(row['specification'],row['budget'],row['fraction'])].append(row)
    aggregate=[]
    for (name,budget,fraction), cell in by_cell.items():
        assert len(cell)==96
        analytic=[r for r in cell if r['corrected_available']]
        aggregate.append(dict(specification=name,budget=budget,fraction=fraction,allocations=len(cell),
                              available=len(analytic),mean_corrected_fallback_minus_matched_direct=float(np.mean([r['difference'] for r in cell])),
                              fallback_lower_loss_count=sum(r['difference']<0 for r in cell),
                              available_mean_difference=float(np.mean([r['difference'] for r in analytic])) if analytic else None))
    args.out.mkdir(parents=True)
    with (args.out/'records.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    state=dict(state='passed',fits=576,path_comparisons=1152,results=aggregate,path_input_sha256=original['output_sha256'],
               plan_sha256=sha(ROOT/'label_cost_plan.json'),code_sha256=sha(Path(__file__)),records_sha256=sha(args.out/'records.csv'))
    (args.out/'summary.json').write_text(json.dumps(state,allow_nan=False,indent=2)+'\n')
    print(json.dumps(dict(state='passed',fits=576,path_comparisons=1152)))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('input','results','numerics','out'):
        p.add_argument('--'+flag,type=Path,required=True)
    with threadpool_limits(limits=1):
        compare(p.parse_args())
