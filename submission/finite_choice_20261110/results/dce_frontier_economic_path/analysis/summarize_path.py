"""Summarize all planned budgets, powers and paired premium decisions."""
import argparse
from collections import defaultdict
import csv
import json
import pathlib

import numpy as np

def q(values):
    return np.quantile(values, [0.05, 0.5, 0.95]).tolist() if len(values) else None

def should_accept(value):
    return {'True': True, 'False': False, '': None}[value]

def summarize(results):
    with (results/'paths.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    cases = {c['frontier_record']:c for c in json.loads((results/'cases.json').read_text())}
    half = {int(r['frontier_record']):r for r in rows if float(r['relative_distance'])==0.5}
    cells = defaultdict(list)
    for row in rows:
        cells[row['group'], int(row['labelled_people']), float(row['relative_distance'])].append(row)
    summaries = []
    for (group, people, dist), part in sorted(cells.items(), key=lambda row: (row[0][0],row[0][1],-row[0][2])):
        resolved = [r for r in part if r['fit_state']=='verified_numerical_fit']
        wtp = [float(r['negative_price_Bere_GBP']) for r in resolved if r['negative_price_Bere_GBP']]
        reference_paths = [
            (r, cases[int(r['frontier_record'])], half[int(r['frontier_record'])])
            for r in resolved
        ]
        pp = dict(group=group, labelled_people=people, relative_distance=dist, denominator=len(part), finite=len(resolved),
                  negative_cost=len(wtp), positive_cost=sum(r['price_state']=='positive' for r in resolved),
                  numerical_zero_cost=sum(r['price_state']=='numerical_zero' for r in resolved),
                  Bere_GBP_5_50_95=q(wtp), holdout_loss_5_50_95=q([float(r['heldout_logloss']) for r in resolved]),
                  paired_holdout_change_from_half_5_50_95=q([float(r['heldout_logloss'])-float(h['heldout_logloss']) for r,_,h in reference_paths if h['heldout_logloss']]),
                  paired_Bere_change_from_half_5_50_95=q([float(r['negative_price_Bere_GBP'])-float(h['negative_price_Bere_GBP']) for r,_,h in reference_paths if r['negative_price_Bere_GBP'] and h['negative_price_Bere_GBP']]),
                  premium={})
        for amount in [0.0, 2.5, 5.0]:
            key = f'accept_GBP_{amount:g}'
            base = dict()
            target = [should_accept(r[key]) for r in resolved]
            for method in ('half_frontier_or_half_unit','labelled_ridge','all_human_MLE'):
                values = []
                for r,c,h in reference_paths:
                    reference = should_accept(h[key]) if method=='half_frontier_or_half_unit' else c[method][key]
                    values.append((reference,should_accept(r[key])))
                common = [(a,b) for a,b in values if a is not None and b is not None]
                base[method] = dict(shared=len(common), changed=sum(a!=b for a,b in common),
                                    accepts_lost=sum(a and not b for a,b in common), accepts_gained=sum(not a and b for a,b in common),
                                    missing_pair=len(part)-len(common))
            pp['premium'][str(amount)] = dict(accept=sum(x is True for x in target),
                                              undefined=len(part)-sum(x is not None for x in target), paired=base)
        summaries.append(pp)
    facets = [c for c in cases.values() if c['facet_state']=='transverse_facet']
    nearest = [r for r in rows if float(r['relative_distance'])==1e-7]
    by_pair = {(int(r['frontier_record']), float(r['relative_distance'])):r for r in rows}
    uncapped_nearest = [r for r in nearest if r['group']=='uncapped_positive']
    near_errors = [abs(float(r['normal_prediction_error'])) for r in uncapped_nearest]
    tang_errors = [float(r['tangent_error']) for r in uncapped_nearest]
    empirical = []
    for c in facets:
        rid = c['frontier_record']
        r1,r2 = (by_pair[rid,dist] for dist in [1e-6,1e-7])
        empirical.append((float(r2['heldout_logloss'])-float(r1['heldout_logloss']))/np.log(10))
    predictions = [c['heldout_slope_per_log_distance'] for c in facets]
    notes = dict(rows=len(rows), cases=len(cases), finite_fits=sum(row['fit_state']=='verified_numerical_fit' for row in rows),
                 qualified_facets=len(facets), strictly_positive_forecast_slope=sum(p>0 for p in predictions),
                 near_endpoint_normal_error_5_50_95=q(near_errors), near_endpoint_tangent_error_5_50_95=q(tang_errors),
                 forecast_log_slope_5_50_95=q(predictions), observed_vs_predicted_forecast_slope_abs_err_5_50_95=q(abs(np.array(empirical)-predictions)),
                 negative_normal_cost=sum(c['exact_integer_normal'][10]<0 for c in facets),
                 zero_normal_cost=sum(c['exact_integer_normal'][10]==0 for c in facets),
                 positive_normal_cost=sum(c['exact_integer_normal'][10]>0 for c in facets),
                 finite_negative_price_face_ratio_5_50_95=q([c['face_ratio_limit_GBP'] for c in facets
                    if c['ratio_limit_state']=='negative_price_normal' or
                    (c['ratio_limit_state']=='both_normal_components_zero' and c['tangent_coefficients'][10]<-1e-9)]),
                 capped_beta_last_to_one_median_max=[float(np.median([c['unit_last_beta_difference'] for c in cases.values() if c['group']=='capped_low_budget'])),
                         max(c['unit_last_beta_difference'] for c in cases.values() if c['group']=='capped_low_budget')])
    return dict(summaries=summaries, geometry=notes)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=pathlib.Path, required=True)
    parser.add_argument('--out', type=pathlib.Path, required=True)
    args=parser.parse_args()
    assert not args.out.exists()
    args.out.write_text(json.dumps(summarize(args.results),allow_nan=False,indent=2)+'\n')
