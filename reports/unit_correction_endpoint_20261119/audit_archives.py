"""Count saved unit-weight states. This does not rerun the empirical fits."""
import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path


def run(root):
    if (root/'submission/finite_choice_20261110').is_dir():
        swiss = root/'reports/swissmetro_label_budget_20261119'
        data = root/'submission/finite_choice_20261110/results'
    else:
        swiss = root/'results/swissmetro_label_budget'
        data = root/'results'
    paths = {
        'Swissmetro': swiss/'results/summary.json',
        'RADIANT_food': data/'public_panel_dce/results/method_records.csv',
        'RADIANT_released_predictions': data/'menu_distribution_transport/results.json',
    }
    swissmetro = Counter()
    for c in json.loads(paths['Swissmetro'].read_text())['cells']:
        if c['method']=='unit':
            swissmetro.update(c['states'])
    with (paths['RADIANT_food']).open(newline='') as f:
        food = Counter(r['support'] for r in csv.DictReader(f) if r['method']=='corrected_unit')
    external = Counter(r['unit']['state'] for r in json.loads(paths['RADIANT_released_predictions'].read_text()))
    data = {}
    for k,counts,n in zip(paths,(swissmetro,food,external),(1344,2000,480)):
        counts['exterior'] += counts.pop('outside',0)
        assert set(counts)=={'interior','exterior'}, (k,counts)
        assert sum(counts.values())==n
        data[k]=dict(unit_cases=n,states=counts,source=str(paths[k].relative_to(root)),
                     source_sha256=sha256(paths[k].read_bytes()).hexdigest())
    return data


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    result=run(args.repo)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print({k:v['states'] for k,v in result.items()})
