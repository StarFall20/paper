"""Check archived Swissmetro fits directly on source rows and re-solve sampled caps."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.special import softmax
from threadpoolctl import threadpool_limits


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rows(p):
    with p.open(newline='') as f:
        return list(csv.DictReader(f))


def raw_data(source):
    raw = np.genfromtxt(source, delimiter='\t', names=True, dtype=int)
    raw = raw[raw['CHOICE'] > 0]
    n = len(raw)
    base = np.zeros((n, 3, 4))
    flexible = np.zeros((n, 3, 8))
    eligible = np.zeros((n, 3), dtype=bool)
    for j, mode in enumerate(['TRAIN', 'SM', 'CAR']):
        cost = raw[mode+'_CO'] / 100
        time = raw[mode+'_TT'] / 100
        eligible[:, j] = raw[mode+'_AV'] == 1
        if j != 1:
            eligible[:, j] &= raw['SP'] != 0
            base[:, j, int(j == 2)] = 1
            flexible[:, j, int(j == 2)] = 1
        if j < 2:
            cost *= raw['GA'] == 0
            flexible[:, j, j+6] = raw[mode+'_HE'] / 100
        base[:, j, 2] = time
        base[:, j, 3] = cost
        flexible[:, j, j+2] = time
        flexible[:, j, 5] = cost
    return raw['ID'], raw['CHOICE']-1, eligible, {'canonical_4': base, 'mode_specific_8': flexible}


def summary_check(data, path_table, summary, counters):
    specification, budget = summary['specification'], summary['budget']
    cells = [a for a in data if a['specification'] == specification and int(a['labelled_people']) == budget]
    assert len(cells) == summary['n']
    assert {key: sum(a['endpoint'] == key for a in cells) for key in set(a['endpoint'] for a in cells)} == summary['endpoint_counts']
    for group, check in summary['path_groups'].items():
        curated = [a for a in cells if a['endpoint'] == group]
        assert len(curated) == check['n']
        slopes = [float(a['holdout_log_slope']) for a in curated if a.get('holdout_log_slope')]
        assert len(slopes) == check['single_facet_cases']
        assert sum(s>0 for s in slopes) == check['positive_slope_cases']
        positive = []
        diffs = []
        for distance, rec in check['paths'].items():
            candidates = [path_table.get((int(a['record']), float(distance)), {}) for a in curated]
            available = [r for r in candidates if r.get('fit') == 'score_verified']
            vals = [float(r['holdout_loss']) for r in available]
            assert len(vals) == rec['n']
            assert sum(float(r['cost_beta']) < -1e-8 for r in available) == rec['negative_cost_count']
            if vals:
                assert abs(np.median(vals)-rec['loss_median']) < 1e-12
            else:
                assert rec['loss_median'] is None
        for rec in curated:
            a = path_table.get((int(rec['record']), .5), {})
            b = path_table.get((int(rec['record']), .9), {})
            if a.get('holdout_loss') and b.get('holdout_loss'):
                diffs.append(float(b['holdout_loss'])-float(a['holdout_loss']))
                positive.append(diffs[-1] > 0)
        assert len(diffs) == check['paired_n'] and sum(positive) == check['paired_positive']
        assert check['paired_loss_median'] is None if not diffs else abs(np.median(diffs)-check['paired_loss_median']) < 1e-12
    counters['summary_cells'] += 1


def checked_cap(features, available, weights0, weights1, b0, b1):
    # Independent variable layout includes all entries with bound zero for unavailable options.
    n, j, p = features.shape
    eq_rows = [np.repeat(np.arange(n), j)]
    eq_cols = [np.arange(n*j)]
    eq_values = [np.ones(n*j)]
    for k in range(p):
        eq_rows.append(np.full(n*j, n+k))
        eq_cols.append(np.arange(n*j))
        eq_values.append(features.reshape(-1, p)[:, k])
    eq_rows.append(np.arange(n+p))
    eq_cols.append(np.full(n+p, n*j))
    eq_values.append(np.r_[weights0-weights1, b0-b1])
    matrix = coo_matrix((np.concatenate(eq_values), (np.concatenate(eq_rows), np.concatenate(eq_cols))),
                        shape=(n+p, n*j+1)).tocsc()
    bound = [(0, None) if valid else (0, 0) for valid in available.ravel()] + [(0, 1)]
    out = linprog(np.r_[np.zeros(n*j), -1], A_eq=matrix, b_eq=np.r_[weights0, b0], bounds=bound,
                  method='highs')
    assert out.success, out.message
    return float(out.x[-1])


def evaluate(utility, observed, valid):
    from scipy.special import logsumexp
    masked = np.where(valid, utility, -np.inf)
    return float(np.mean(logsumexp(masked, axis=1)-masked[np.arange(len(masked)), observed]))


def verify(args):
    plan = json.loads((Path(__file__).parent/'plan.json').read_text())
    result = json.loads((args.results/'summary.json').read_text())
    assert sha(args.input) == plan['data_sha256'] == result['source_sha256']
    assert sha(Path(__file__).parent/'plan.json') == result['plan_sha256']
    assert sha(Path(__file__).parent/'audit.py') == result['script_sha256']
    assert all(sha(args.results / n) == h for n, h in result['output_sha256'].items())
    ids, y, mask, designs = raw_data(args.input)
    split = np.random.default_rng(plan['split_seed']).permutation(np.unique(ids))
    teach, test, pool = np.split(split, [600, 840])
    train = np.isin(ids, teach)
    test_rows = np.isin(ids, test)
    pool_rows = np.isin(ids, pool)
    sample = ids[pool_rows]
    canon = designs['canonical_4']
    teacher = np.asarray(result['input_summary']['teacher_coefs'])
    train_mask = mask[train]
    teacher_p = softmax(np.where(train_mask, canon[train] @ teacher, -np.inf), axis=1)
    label = np.eye(3)[y[train]]
    teacher_score = np.einsum('ij,ijp->p', teacher_p-label, canon[train])/len(teacher_p)
    assert max(abs(teacher_score)) < 1e-8
    teacher_loss = evaluate(canon[test_rows] @ teacher, y[test_rows], mask[test_rows])
    assert abs(teacher_loss-result['input_summary']['teacher_evaluation_logloss']) < 1e-12
    fmask, emask = mask[pool_rows], mask[test_rows]
    yy, ey = y[pool_rows], y[test_rows]
    predict = softmax(np.where(fmask, canon[pool_rows] @ teacher, -np.inf), axis=1)
    chosen = np.eye(3)[yy]
    all_alloc = rows(args.results/'allocations.csv')
    all_path = rows(args.results/'paths.csv')
    coefficients = {(r['record'], r['method']): np.array(r['beta']) for r in json.loads((args.results/'coefficients.json').read_text())}
    path_table = {(int(r['record']), float(r['fraction'])): r for r in all_path}
    assert len(all_alloc) == result['allocation_records'] == 576
    assert len(path_table) == len(all_path) == result['path_fits']
    assert [int(r['record']) for r in all_alloc] == list(range(576))
    rng_rows = {i: np.random.default_rng(plan['allocation_seed']+i).permutation(pool) for i in range(plan['allocations'])}
    counts = {'scores': 0, 'ridge_scores': 0, 'cap_programs': 0, 'uncapped_dual_checks': 0, 'face_primal_checks': 0, 'summary_cells': 0}
    max_errors = {'score': 0., 'holdout_loss': 0., 'cap': 0.}

    for name, design in designs.items():
        f, ev = design[pool_rows], design[test_rows]
        direct = np.asarray(result['input_summary'][f'full_human_coefs_{name}'])
        pq = softmax(np.where(fmask, f @ direct, -np.inf), axis=1)
        full_score = np.einsum('ij,ijp->p', pq-chosen, f)/len(yy)
        assert max(abs(full_score)) < 1e-8
        full_eval = evaluate(ev @ direct, ey, emask)
        for row in (r for r in all_alloc if r['specification'] == name):
            number, m, trial = (int(row[k]) for k in ['record','labelled_people','allocation'])
            inf = np.isin(sample, rng_rows[trial][:m])
            aux = ~inf
            labelled_count = int(row['labelled_tasks'])
            assert labelled_count == inf.sum()
            assert abs(full_eval - float(row['full_human_loss'])) < 1e-12
            leach = 1/labelled_count
            ueach = 1/(len(yy)-labelled_count)
            b0 = np.einsum('ij,ijp->p', chosen[inf], f[inf])*leach
            dp = np.einsum('ij,ijp->p', predict[aux], f[aux])*ueach - np.einsum('ij,ijp->p', predict[inf], f[inf])*leach
            cap = float(row['cap'])
            if trial % 16 == 0:
                other = checked_cap(f,fmask,inf*leach,aux*ueach,b0,b0+dp)
                assert abs(other-cap) < 1e-7
                max_errors['cap'] = max(max_errors['cap'], abs(other-cap))
                counts['cap_programs'] += 1
            if row['endpoint'] == 'positive_uncapped':
                direction = np.array(json.loads(row['normal']))
                v = np.max(np.where(fmask, f @ direction, -np.inf), axis=1)
                intercept = v[inf].mean() - b0@direction
                slope = v[inf].mean()-v[aux].mean()+dp@direction
                assert slope > 0 and abs(intercept/slope-cap)<1e-7
                if row.get('holdout_log_slope'):
                    eu = ev @ direction
                    gap = np.max(np.where(emask,eu,-np.inf),axis=1)-eu[np.arange(len(ey)),ey]
                    assert abs(gap.mean()/float(row['gap_min'])-float(row['holdout_log_slope']))<1e-8
                    tied = fmask & ((v[:, None] - f@direction) <= 1e-8)
                    shift = f[np.arange(len(f)), tied.argmax(axis=1)]
                    rank = np.linalg.matrix_rank((f-shift[:, None, :])[tied], tol=1e-8)
                    assert rank == f.shape[-1]-1 == int(row['facet_rank'])
                    weights = (1-cap)*inf*leach+cap*aux*ueach
                    target = b0+cap*dp
                    uniform = np.sum((tied/tied.sum(axis=1, keepdims=True))[:, :, None]*f*weights[:, None, None], axis=(0,1))
                    margin = checked_cap(f,tied,weights,np.zeros(len(f)),target,target-uniform)
                    assert margin > 1e-8 and abs(margin-float(row['face_margin'])) < 1e-6, (number, margin, row['face_margin'])
                    counts['face_primal_checks'] += 1
                counts['uncapped_dual_checks'] += 1
            ridge = coefficients[(number,'labelled_ridge')]
            rfit = softmax(np.where(fmask[inf], f[inf] @ ridge, -np.inf),axis=1)
            score = np.einsum('ij,ijp->p', rfit-chosen[inf], f[inf])*leach + plan['ridge_penalty']*ridge
            assert max(abs(score)) < 1e-8
            assert abs(evaluate(ev@ridge,ey,emask)-float(row['ridge_loss'])) < 1e-12
            counts['ridge_scores'] += 1
            for fraction in plan['paths']['fractions_of_upper_limit']:
                rec = path_table.get((number, fraction))
                if rec is None or rec['fit'] != 'score_verified':
                    assert (number, f'path_{fraction:g}') not in coefficients
                    continue
                beta = coefficients[(number, f'path_{fraction:g}')]
                power = float(rec['power'])
                assert 0 < power < cap and abs(power-cap*fraction)<1e-12
                p = softmax(np.where(fmask, f@beta, -np.inf),axis=1)
                score = (np.einsum('ij,ijp->p', (1-power)*p[inf]-chosen[inf]+power*predict[inf], f[inf])*leach
                         +power*np.einsum('ij,ijp->p', p[aux]-predict[aux],f[aux])*ueach)
                loss = evaluate(ev@beta,ey,emask)
                raw_scores = [abs(float(rec['holdout_loss'])-loss), max(abs(score))]
                assert raw_scores[0]<1e-10 and raw_scores[1]<1e-8
                max_errors['score'] = max(max_errors['score'], float(raw_scores[1]))
                max_errors['holdout_loss'] = max(max_errors['holdout_loss'], raw_scores[0])
                cost, time = (3,2) if name == 'canonical_4' else (5,3)
                assert float(rec['cost_beta']) == beta[cost] and float(rec['swissmetro_time_beta']) == beta[time]
                if beta[cost] < -1e-8:
                    assert abs(float(rec['swissmetro_time_value_per_hour'])-60*beta[time]/beta[cost]) < 1e-10
                else:
                    assert not rec['swissmetro_time_value_per_hour']
                counts['scores'] += 1
    for summary in result['summaries']:
        summary_check(all_alloc, path_table, summary, counts)
    assert counts['scores'] + counts['ridge_scores'] == len(coefficients)
    assert counts['ridge_scores'] == 576 and counts['cap_programs'] == 36
    verification = dict(state='passed', verified=counts, maximum_absolute_errors=max_errors,
                        summary_sha256=sha(args.results/'summary.json'), verifier_sha256=sha(Path(__file__)),
                        original_available_alternatives_reconstructed=True)
    args.out.write_text(json.dumps(verification, indent=2)+'\n')
    print(json.dumps(verification))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('input', 'results', 'out'):
        parser.add_argument('--' + arg, type=Path, required=True)
    with threadpool_limits(limits=1):
        verify(parser.parse_args())
