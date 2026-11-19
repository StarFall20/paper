"""Check constructed directions analytically and verify empirical scope from saved audits."""
import argparse
import csv
from fractions import Fraction
from hashlib import sha256
import json
from math import cosh, log, atanh
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCES = {
    'radiant_endpoints.json': '16eaf3f6a5d2188b8b9896f8920c8030e9030b5e42dab1d4240dacdc1c18e1a9',
    'swissmetro_allocations.csv': 'e82230099f98f67cb0cd2347b207c343f308da38f7d5b89c3eb55ce8896e3340',
}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('results', 'archive', 'out'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads((ROOT/'plan.json').read_text())
    results = json.loads(args.results.read_text())
    assert results['status']=='passed' and len(results['cases'])==8 and results['recorded_fits']==72
    for case in plan['cases']:
        rate = list(map(Fraction,case['rate']))
        a = case['gaps']
        lengths = [sum(r*z for r,z in zip(rate,g,strict=True)) for g in a]
        assert min(lengths)==1
        top_factor = 2*cosh(atanh(1/8))
        if case['name']=='two_normal_unique':
            constants = [-log(sum(g[j] for g in a)*top_factor/4) for j in range(2)]
        else:
            constants = [-log(sum(g[0] for g in a)*top_factor/4)*float(v) for v in rate]
        expected = constants+[atanh(1/8)]
        for row in [r for r in results['cases'] if r['case']==case['name']]:
            path = row['path']
            for k, fit in path.items():
                assert fit['score_error'] < 10**(-int(k)-26)
                if 'nonunique' in case['name'] or 'levels' in case['name']:
                    # Zero projected sample score pairs equal outside probabilities in each secondary direction.
                    actual = fit['scaled_coefficients']
                    assert abs(actual[1]-actual[0]/2) < 1e-12
                    if len(actual)==3:
                        assert abs(actual[2]-actual[0]) < 1e-12
            near = path[str(max(plan['fit_decimals']))]
            assert max(abs(x-y) for x,y in zip(near['bounded_remainder'],expected,strict=True)) < 2e-12

    for name, want in SOURCES.items():
        assert digest(args.archive/name)==want
    food = json.loads((args.archive/'radiant_endpoints.json').read_text())
    assert len(food)==len({r['frontier_record'] for r in food})==107
    assert all(0<r['uncompressed_frontier']<1 and r['min_tied_probability']>0 and r['minimum_gap']>0 for r in food)
    with (args.archive/'swissmetro_allocations.csv').open(newline='') as f:
        transit = [r for r in csv.DictReader(f) if r['endpoint']=='positive_uncapped']
    assert len(transit)==len({r['record'] for r in transit})==93
    for row in transit:
        assert float(row['face_margin'])>0 and row['face_support']=='interior'
        assert int(row['facet_rank'])+1 == {'canonical_4':4, 'mode_specific_8':8}[row['specification']]
    verification = dict(status='passed', closed_form_remainder_checks=8, exact_structure_paths=36,
                        archived_positive_endpoints={'RADIANT':len(food),'Swissmetro':len(transit)},
                        empirical_scope='Both retained endpoint audits report single facets. The multiface paths are constructed.',
                        results_sha256=digest(args.results),prior_inputs=SOURCES)
    args.out.write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps(verification))


if __name__=='__main__':
    main()
