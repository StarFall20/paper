"""Compare certified availability and forecasting over nested label budgets."""
import argparse
from collections import defaultdict
import csv
from functools import partial
from fractions import Fraction
from math import lcm
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import qr
from scipy.optimize import linprog, root
from scipy.sparse import coo_matrix, csc_matrix, hstack
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
source_tree = HERE.parent
swissmetro_path = source_tree / 'swissmetro_corrected_path_20261118'
frontier_path = source_tree / 'dce_frontier_economic_path_20261118'
if not swissmetro_path.is_dir():
    swissmetro_path = source_tree / 'swissmetro_corrected_path'
    frontier_path = source_tree / 'dce_frontier_economic_path'
sys.path.insert(0, str(swissmetro_path))
sys.path.insert(0, str(frontier_path / 'analysis'))
from audit import (DesignSupport, center_features, chosen_moment, fit_likelihood,
                   load_data, logloss, predicted_moment, probability, sha)
from audit_path import objective, solve

CERTIFICATE_TOL = 1e-8


def prepare(input_path, plan):
    assert sha(input_path) == plan['source_sha256']
    ids, choice, mask, designs = load_data(input_path)
    people = np.random.default_rng(plan['split_seed']).permutation(np.unique(ids))
    teacher, evaluation, pool = np.split(people, [
        plan['teacher_respondents'], plan['teacher_respondents'] + plan['evaluation_respondents']])
    teacher_rows, eval_rows, pool_rows = [np.isin(ids, p) for p in (teacher, evaluation, pool)]
    assert len(choice) == 10719 and len(pool) == 351
    assert all(np.all(np.unique(ids[r], return_counts=True)[1] == 9) for r in (teacher_rows, eval_rows, pool_rows))
    designs = {name: center_features(f, mask) for name, f in designs.items()}
    base = designs['canonical_4']
    teacher_beta = fit_likelihood(objective, base[teacher_rows], mask[teacher_rows], choice[teacher_rows], 0)
    proxy = probability(base[pool_rows], mask[pool_rows], teacher_beta)
    orders = [np.random.default_rng(plan['allocation_seed'] + i).permutation(pool)
              for i in range(plan['allocations'])]
    return dict(ids=ids, choice=choice, mask=mask, designs=designs, teacher_rows=teacher_rows,
                evaluation_rows=eval_rows, pool_rows=pool_rows, proxy=proxy, orders=orders,
                teacher_beta=teacher_beta.tolist())


def grouped_scores(row_scores, inverse, counts):
    means = np.zeros((len(counts), row_scores.shape[1]))
    np.add.at(means, inverse, row_scores)
    return means / counts[:, None]


def objective_blas(features, weights, target, beta, mask):
    utility = np.where(mask, features @ beta, -np.inf)
    normalizer = logsumexp(utility, axis=1)
    prob = np.exp(utility-normalizer[:,None])
    means = np.sum(prob[:,:,None]*features, axis=1)
    centered = (features-means[:,None,:]).reshape(-1,len(beta))
    mass = (prob*weights[:,None]).ravel()
    return weights @ normalizer - target @ beta, weights @ means - target, centered.T @ (mass[:,None]*centered)


def solve_blas(features, weights, target, start, mask, penalty=0.0):
    def terms(beta):
        loss, gradient, hessian = objective_blas(features, weights, target, beta, mask)
        return (loss + penalty*(beta@beta)/2, gradient+penalty*beta,
                hessian + penalty*np.eye(len(beta)))

    beta = start.copy()
    for _ in range(36):
        value, score, hessian = terms(beta)
        if max(abs(score)) < 3e-13:
            return beta
        try:
            step = np.linalg.solve(hessian, score)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(step)) or max(abs(step)) > 1500:
            break
        for k in range(16):
            factor = 2.0**-k
            proposal = beta-factor*step
            trial = terms(proposal)[0]
            if trial <= value - 1e-4*factor*(step@score) + 2e-14:
                beta = proposal
                break
        else:
            break
    return beta


def fit_human_blas(features, mask, choice, penalty):
    weights = np.full(len(features),1/len(features))
    target = chosen_moment(features,choice)
    beta = solve_blas(features,weights,target,np.zeros(features.shape[-1]),mask,penalty)
    def score(x):
        return objective_blas(features,weights,target,x,mask)[1]+penalty*x
    if max(abs(score(beta))) > 1e-9:
        beta = root(score,beta,method='hybr',options={'xtol':1e-11}).x
    assert max(abs(score(beta))) < 1e-8
    return beta


def select_weight(features, mask, choice, proxy, indices, by_person, candidates, pilot):
    n = len(features)
    current_p = probability(features, mask, pilot)
    model_mean = np.einsum('ij,ijp->ip', current_p, features)
    r_proxy = model_mean - np.einsum('ij,ijp->ip', proxy, features)
    pooled = np.full(n, 1/n)
    _, _, hessian = objective(features, pooled, np.zeros(features.shape[-1]), pilot, mask)
    whitened_proxy = np.linalg.solve(hessian, r_proxy.T).T
    g = grouped_scores(whitened_proxy, *by_person)
    is_labelled = np.isin(candidates, indices)
    inverse, counts = by_person
    label_rows = np.flatnonzero(is_labelled[inverse])
    r_human = model_mean[label_rows] - features[label_rows, choice[label_rows]]
    whitened_human = np.linalg.solve(hessian, r_human.T).T
    label_inverse = np.searchsorted(np.flatnonzero(is_labelled), inverse[label_rows])
    label_y = grouped_scores(whitened_human, label_inverse, counts[is_labelled])
    label_g, aux_g = g[is_labelled], g[~is_labelled]
    label_g = label_g - label_g.mean(axis=0)
    label_y = label_y - label_y.mean(axis=0)
    aux_g = aux_g - aux_g.mean(axis=0)
    m, a = len(label_y), len(aux_g)
    cross = np.sum(label_g * label_y) / (m - 1)
    variance = np.sum(label_g**2) / (m - 1) + (m/a) * np.sum(aux_g**2) / (a-1)
    weight = float(np.clip(cross/variance, 0, 1)) if variance > 1e-14 else 0.0
    return weight, float(cross), float(variance)


class ActiveSupport:
    def __init__(self, features, mask, weights):
        self.features = features
        self.mask = mask
        self.weights = weights
        self.menu, self.alternative = np.nonzero(mask)
        self.reference = mask.argmax(axis=1)
        self.p = features.shape[-1]
        self.contrasts = features[self.menu, self.alternative] - features[self.menu, self.reference[self.menu]]
        _, triangular, order = qr(self.contrasts.T, mode='economic', pivoting=True)
        self.rank = int(np.sum(np.abs(np.diag(triangular)) > 1e-9))
        self.basis = self.contrasts[order[:self.p]].T
        self.basis_menu = self.menu[order[:self.p]]
        self.basis_alternative = self.alternative[order[:self.p]]

    def positive_witness(self, p, target):
        error = target - predicted_moment_weighted(self.features, p, self.weights)
        change = np.linalg.solve(self.basis, error) / self.weights[self.basis_menu]
        adjusted = p.copy()
        np.add.at(adjusted, (self.basis_menu, self.basis_alternative), change)
        np.add.at(adjusted, (self.basis_menu, self.reference[self.basis_menu]), -change)
        margin = float(np.min(adjusted[self.mask]))
        residual = float(np.max(np.abs(predicted_moment_weighted(self.features, adjusted, self.weights) - target)))
        if margin > 1e-11 and residual < 1e-11 and np.max(np.abs(adjusted.sum(axis=1)-1)) < 1e-11:
            return margin
        return None

    def negative_direction(self, target):
        rows = np.arange(len(self.menu))
        dimension = len(self.features)
        row_max = coo_matrix((-np.ones(len(rows)), (rows, self.menu)), shape=(len(rows), dimension))
        matrix = hstack((csc_matrix(self.features[self.menu, self.alternative]), row_max), format='csc')
        opt = linprog(np.r_[-target, self.weights], A_ub=matrix,
                      b_ub=np.zeros(len(rows)), bounds=[(-1, 1)] * self.p + [(None, None)] * dimension,
                      method='highs-ipm')
        if opt.success:
            direction = opt.x[:self.p]
            slack = recession(self.features, self.mask, self.weights, target, direction)
            if slack < -CERTIFICATE_TOL:
                return direction.tolist(), slack
        return None, None


def predicted_moment_weighted(f, probability_rows, weights):
    return weights @ np.einsum('ij,ijp->ip', probability_rows, f)


def recession(f, mask, weights, target, direction):
    largest = np.where(mask, f @ direction, -np.inf).max(axis=1)
    return float(weights @ largest - target @ direction)


def project_direction_to_ties(grid, direction):
    tight = np.unique(grid[np.abs(grid @ direction) < 1e-8],axis=0)
    matrix = [[Fraction(int(value)) for value in row] for row in tight]
    pivots = []
    for j in range(len(direction)):
        i0 = len(pivots)
        i = next((i for i in range(i0,len(matrix)) if matrix[i][j]),None)
        if i is None:
            continue
        matrix[i0], matrix[i] = matrix[i],matrix[i0]
        row = matrix[i0]
        matrix[i0] = [u/row[j] for u in row]
        for k in range(len(matrix)):
            if k != i0 and matrix[k][j]:
                scale = matrix[k][j]
                matrix[k] = [a-scale*b for a,b in zip(matrix[k],matrix[i0])]
        pivots.append(j)
    out = [Fraction(float(x)).limit_denominator(10**6) for x in direction]
    free = [j for j in range(len(out)) if j not in pivots]
    for i,j in enumerate(pivots):
        out[j] = -sum(matrix[i][k]*out[k] for k in free)
    return out


def ordinary_boundary_witness(f, mask, choice):
    differences = (f - f[np.arange(len(f)), choice][:, None, :])[mask]
    scaled = 100 * differences
    grid = np.rint(scaled).astype(np.int64)
    if np.max(np.abs(grid - scaled)) > 1e-9:
        return None
    check = linprog(np.zeros(f.shape[-1]), A_ub=np.vstack((grid, grid.sum(axis=0))),
                    b_ub=np.r_[np.zeros(len(grid)), -1], bounds=(None, None), method='highs')
    if not check.success:
        return None
    rounded = [Fraction(float(x)).limit_denominator(10**6) for x in check.x]
    for fractions in (rounded,project_direction_to_ties(grid,check.x)):
        multiplier = lcm(*(x.denominator for x in fractions))
        integer = [int(multiplier*x) for x in fractions]
        exact = grid.astype(object) @ integer
        if exact.min() < 0 and exact.max() <= 0:
            return integer
    return None


def classify(f, mask, weights, target, pilot, score_tolerance, solver, ordinary_choice=None):
    active = weights > 0
    f, mask, weights = f[active], mask[active], weights[active]
    checker = ActiveSupport(f, mask, weights)
    if checker.rank < checker.p:
        direction, slack = checker.negative_direction(target)
        if direction is not None:
            return 'exterior', None, dict(rank=checker.rank, direction=direction, recession=slack)
        return 'rank_deficient', None, dict(rank=checker.rank)
    beta = solver(f, weights, target, pilot, mask)
    if np.all(np.isfinite(beta)) and max(abs(beta)) < 10000:
        pr = probability(f, mask, beta)
        w = checker.positive_witness(pr, target)
        if w is not None:
            score = np.max(abs(predicted_moment_weighted(f, pr, weights) - target))
            if score < score_tolerance:
                return 'interior', beta, dict(rank=checker.rank, witness_min_probability=w, score=float(score))
    if ordinary_choice is not None:
        separating = ordinary_boundary_witness(f, mask, ordinary_choice)
        if separating is not None:
            return 'boundary', None, dict(rank=checker.rank, exact_boundary_direction=separating)
    direction, slack = checker.negative_direction(target)
    if direction is not None:
        return 'exterior', None, dict(rank=checker.rank, direction=direction, recession=slack)
    state, margin = DesignSupport(f, mask).interior(weights, target)
    if state == 'interior':
        beta = solve(f, weights, target, pilot, mask)
        assert np.all(np.isfinite(beta))
        score = np.max(abs(predicted_moment_weighted(f, probability(f, mask, beta), weights) - target))
        if score < score_tolerance:
            return 'interior', beta, dict(rank=checker.rank, support_lp_margin=margin, score=float(score))
        return 'interior_fit_failed', None, dict(rank=checker.rank, support_lp_margin=margin, score=float(score))
    return 'unresolved', None, dict(rank=checker.rank, primal_state=state, support_lp_margin=margin)


def evaluate_cell(data, plan, name, allocation, budget, context):
    f, mask, choice, ids = context['pool']
    e_f, e_mask, e_y = context['evaluation']
    train_f, train_mask, train_y = context['teacher']
    proxy = data['proxy']
    picked = data['orders'][allocation][:budget]
    use = np.isin(ids, picked)
    rest = ~use
    f_l, a_l, y_l = f[use], mask[use], choice[use]
    fit_human = context['fit_human']
    ridge = fit_human(f_l, a_l, y_l, plan['same_label_ridge'])
    pooled_human = fit_human(np.concatenate((train_f, f_l)),
                                  np.concatenate((train_mask, a_l)), np.concatenate((train_y, y_l)), 0)
    eval_loss = partial(logloss, e_f, e_mask, e_y)
    ridge_loss, direct_loss = eval_loss(ridge), eval_loss(pooled_human)
    tune, cross, variance = select_weight(f, mask, choice, proxy, picked,
                                         context['by_person'], context['people'], ridge)
    b0 = chosen_moment(f_l, y_l)
    shift = predicted_moment(f[rest], proxy[rest]) - predicted_moment(f_l, proxy[use])
    w0, w1 = use.astype(float) / use.sum(), rest.astype(float) / rest.sum()
    methods = []
    coefficients = dict(ridge=ridge.tolist(), total_label_direct=pooled_human.tolist())
    solutions = {}
    for method, power in [('zero', 0.0), ('half', 0.5), ('unit', 1.0), ('cluster_plugin', tune)]:
        if power not in solutions:
            solutions[power] = classify(f, mask, (1-power)*w0+power*w1, b0+power*shift,
                                        ridge, score_tolerance=1e-8, solver=context['solver'], ordinary_choice=y_l if power == 0 else None)
        state, beta, certificate = solutions[power]
        loss = eval_loss(beta) if beta is not None else None
        methods.append(dict(specification=name, allocation=allocation, label_budget=budget, method=method,
                            weight=power, state=state, ridge_loss=ridge_loss, direct_loss=direct_loss,
                            native_loss=loss, policy_loss=loss if loss is not None else ridge_loss,
                            plugin_cross=cross, plugin_variance=variance, **certificate))
        if beta is not None:
            coefficients[method] = beta.tolist()
    return methods, coefficients


def summaries(records):
    cells, paths = defaultdict(list), defaultdict(dict)
    for row in records:
        cells[(row['specification'], row['label_budget'], row['method'])].append(row)
        paths[(row['specification'], row['allocation'], row['method'])][row['label_budget']] = row['state']
    compact = []
    for (model, budget, method), rows in cells.items():
        loss = np.array([r['policy_loss'] for r in rows])
        ridge = np.array([r['ridge_loss'] for r in rows])
        direct = np.array([r['direct_loss'] for r in rows])
        p = np.array([r['weight'] for r in rows])
        states = {state: sum(r['state'] == state for r in rows) for state in sorted({r['state'] for r in rows})}
        compact.append(dict(model=model, label_budget=budget, method=method, cases=len(rows), states=states,
                            weight_zero=int(sum(p == 0)), weight_one=int(sum(p == 1)),
                            weight_median=float(np.median(p)), mean_policy_loss=float(loss.mean()),
                            mean_difference_to_ridge=float((loss-ridge).mean()), wins_vs_ridge=int(sum(loss+1e-10 < ridge)),
                            mean_difference_to_direct=float((loss-direct).mean()), wins_vs_direct=int(sum(loss+1e-10 < direct))))
    changes = defaultdict(lambda: dict(cases=0, lost_finite=0, gained_finite=0, newly_exterior=0))
    for (model, _, method), observed in paths.items():
        sizes = sorted(observed)
        for left, right in zip(sizes, sizes[1:]):
            a, b = observed[left], observed[right]
            cur = changes[(model, method, left, right)]
            cur['cases'] += 1
            cur['lost_finite'] += (a == 'interior' and b != 'interior')
            cur['gained_finite'] += (a != 'interior' and b == 'interior')
            cur['newly_exterior'] += (a != 'exterior' and b == 'exterior')
    return dict(cells=compact,
                transitions=[dict(model=m, method=k, previous_labels=a, next_labels=b, **v)
                             for (m,k,a,b),v in changes.items()])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--specification')
    parser.add_argument('--start-allocation',type=int,default=0)
    parser.add_argument('--fast',action='store_true')
    args = parser.parse_args()
    plan = json.loads((HERE/'plan.json').read_text())
    data = prepare(args.input, plan)
    args.out.mkdir(parents=True, exist_ok=False)
    records = []
    with (args.out/'coefficients.jsonl').open('w') as result:
        for name, feature in data['designs'].items():
            if args.specification and name != args.specification:
                continue
            context = {group: tuple(x[data[group+'_rows']] for x in
                                    (feature, data['mask'], data['choice'], data['ids']))
                       for group in ('pool','evaluation','teacher')}
            context['evaluation'] = context['evaluation'][:3]
            context['teacher'] = context['teacher'][:3]
            people, inverse, counts = np.unique(context['pool'][3], return_inverse=True, return_counts=True)
            context.update(by_person=(inverse,counts), people=people,
                           fit_human=fit_human_blas if args.fast else partial(fit_likelihood, objective),
                           solver=solve_blas if args.fast else solve)
            stop = min(plan['allocations'], args.start_allocation + (args.limit or plan['allocations']))
            for trial in range(args.start_allocation,stop):
                for budget in plan['nested_label_budgets']:
                    rows, coefficients = evaluate_cell(data, plan, name, trial, budget, context)
                    records.extend(rows)
                    result.write(json.dumps(dict(model=name, allocation=trial, label_budget=budget, **coefficients))+'\n')
                if trial % 16 == 15 or trial == 0:
                    print(f'{name} completed {trial+1} nested paths', flush=True)
    keys = sorted({k for r in records for k in r})
    with (args.out/'records.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)
    s = summaries(records)
    s.update(total_method_records=len(records), teacher_beta=data['teacher_beta'],
             unresolved=sum(r['state'] in ('unresolved','interior_fit_failed') for r in records))
    (args.out/'summary.json').write_text(json.dumps(s,indent=2)+'\n')
    print(json.dumps(dict(records=len(records), unresolved=s['unresolved'])), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
