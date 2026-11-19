"""Assemble the fixed design results and recheck ambiguous ordinary boundaries."""
import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from run_audit import prepare, ordinary_boundary_witness, summaries

HERE = Path(__file__).resolve().parent


def run(args):
    plan = json.loads((HERE/'plan.json').read_text())
    source = prepare(args.input,plan)
    budget_keys = {(spec,i,n) for spec in plan['specifications'] for i in range(plan['allocations'])
                   for n in plan['nested_label_budgets']}
    fitted = {}
    records = {}
    for directory in args.parts:
        for line in (directory/'coefficients.jsonl').read_text().splitlines():
            payload = json.loads(line)
            key = tuple(payload[k] for k in ('model','allocation','label_budget'))
            assert key not in fitted, key
            fitted[key] = payload
        with (directory/'records.csv').open(newline='') as f:
            for row in csv.DictReader(f):
                key = (row['specification'],int(row['allocation']),int(row['label_budget']),row['method'])
                assert key not in records,key
                fields = {}
                for name, value in row.items():
                    if value == '':
                        continue
                    if name in ('allocation','label_budget','rank'):
                        fields[name] = int(value)
                    elif name in ('direct_loss','native_loss','plugin_cross','plugin_variance','policy_loss',
                                  'recession','ridge_loss','score','support_lp_margin','weight','witness_min_probability'):
                        fields[name] = float(value)
                    elif name in ('direction','exact_boundary_direction'):
                        fields[name] = json.loads(value)
                    else:
                        fields[name] = value
                records[key] = fields
    assert set(fitted) == budget_keys
    assert len(records) == 4*len(budget_keys)
    corrections = []
    for key, row in records.items():
        if row['state'] != 'unresolved' or row['weight'] != 0:
            continue
        spec, allocation, count, method = key
        picked = source['orders'][allocation][:count]
        rows = np.isin(source['ids'],picked)
        assert np.all(source['pool_rows'][rows])
        d = ordinary_boundary_witness(source['designs'][spec][rows],source['mask'][rows],source['choice'][rows])
        if d is not None:
            row.update(state='boundary',exact_boundary_direction=d,original_state='unresolved')
            corrections.append(dict(model=spec,allocation=allocation,budget=count,method=method))
    args.out.mkdir(parents=True,exist_ok=False)
    with (args.out/'coefficients.jsonl').open('w') as f:
        for key in sorted(fitted):
            f.write(json.dumps(fitted[key],allow_nan=False)+'\n')
    with (args.out/'records.csv').open('w',newline='') as f:
        fields = sorted({k for row in records.values() for k in row})
        writer = csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(records[k] for k in sorted(records))
    result = summaries(records.values())
    result.update(records=len(records),model_allocation_budgets=len(fitted),
                  teacher_beta=source['teacher_beta'],boundary_refinements=corrections,
                  all_states=Counter(r['state'] for r in records.values()),source_sha256=plan['source_sha256'],
                  plan_sha256=sha256((HERE/'plan.json').read_bytes()).hexdigest())
    (args.out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print({k:result[k] for k in ('records','model_allocation_budgets','all_states')},len(corrections))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--parts',type=Path,nargs='+',required=True)
    parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args())
