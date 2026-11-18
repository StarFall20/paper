"""Build exact grouped binary choice designs with one power and predictor per case."""
import argparse
from collections import Counter
from fractions import Fraction as F
from hashlib import sha256
import json
from math import lcm
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def symmetric(magnitude_mass):
    weights = Counter()
    for magnitude, value in magnitude_mass:
        x, w = F(magnitude), F(value)
        weights[x] += w / 2
        weights[-x] += w / 2
    assert sum(weights.values()) == 1
    return weights


def group_profiles(distribution, share):
    profiles = Counter({(F(0), -1): share / 2, (F(0), 1): share / 2})
    for x, mass in distribution.items():
        profiles[x, 0] += (1 - share) * mass
    return profiles


def counts(weights, size):
    rows = []
    for key, weight in sorted(weights.items()):
        if weight:
            count = weight * size
            assert count.denominator == 1 and count > 0
            rows.append([str(key[0]), *key[1:], int(count)])
    return rows


def build(family, reference, power, plan):
    high = symmetric(reference['labelled_magnitude_mass'])
    low = symmetric(reference['auxiliary_magnitude_mass'])
    m_high, m_low = [sum(abs(x) * w for x, w in group.items()) for group in (high, low)]
    tilt = 1 - power * (m_high - m_low) / (2 * m_high)
    assert 0 < tilt < 1
    share = F(plan['informative_share'])
    offset = F(plan['prediction_offset'])
    labelled = Counter()
    for (x, z), mass in group_profiles(high, share).items():
        p = F(1, 2) + (offset * z if z else tilt * ((x > 0) - (x < 0)) / 2)
        labelled[x, z, 1] += mass * p
        labelled[x, z, 0] += mass * (1 - p)
    groups = {}
    for state, value in plan['auxiliary_mixes'].items():
        fraction = F(value)
        combined = Counter()
        for distribution, factor in ((high, 1 - fraction), (low, fraction)):
            for x, mass in distribution.items():
                combined[x] += mass * factor
        groups[state] = group_profiles(combined, share)
    n = lcm(*(w.denominator for group in groups.values() for w in group.values()))
    m = lcm(*(w.denominator for w in labelled.values()))
    return dict(family=family, K=reference['K'], matched_degree=2*reference['K']+1, power=str(power),
                labelled_count=m, auxiliary_count=n, labelled=counts(labelled, m),
                auxiliary={key: counts(value, n) for key, value in groups.items()})


def main(source_path, out):
    plan_path = ROOT/'plan.json'
    plan = json.loads(plan_path.read_text())
    assert sha(source_path) == plan['source_sha256']
    source = json.loads(source_path.read_text())
    cases = [build(family, record, F(power), plan)
             for family in plan['source_families']
             for record in source[family]
             for power in plan['powers']]
    result = dict(status='built', source_sha256=sha(source_path), plan_sha256=sha(plan_path),
                  constructor_sha256=sha(Path(__file__)), cases=cases)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print({'cases': len(cases), 'largest_order': max(c['matched_degree'] for c in cases)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    main(args.source, args.out)
