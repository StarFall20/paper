"""Replicate correction-power diagnostics on published Swissmetro choice data."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.linalg import null_space
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix, csc_matrix, hstack, vstack
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
TOL = {'primal_feasibility_tolerance': 1e-9, 'dual_feasibility_tolerance': 1e-9}
CAP_TOL = 1e-8


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False, indent=2) + '\n')


def save_csv(path, rows):
    with path.open('w', newline='') as stream:
        keys = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_data(path):
    frame = pd.read_csv(path, sep='\t')
    frame = frame.loc[frame.CHOICE != 0].reset_index(drop=True)
    sample = frame.ID.to_numpy()
    choice = frame.CHOICE.to_numpy() - 1
    availability = frame[['TRAIN_AV', 'SM_AV', 'CAR_AV']].to_numpy().astype(bool)
    availability[:, [0, 2]] &= frame.SP.to_numpy()[:, None] != 0
    assert np.all(availability[np.arange(len(choice)), choice])
    assert np.all(availability.sum(axis=1) >= 1)
    fares = frame[['TRAIN_CO', 'SM_CO', 'CAR_CO']].to_numpy(dtype=float)
    fares[:, :2] *= frame.GA.to_numpy()[:, None] == 0
    times = frame[['TRAIN_TT', 'SM_TT', 'CAR_TT']].to_numpy(dtype=float) / 100
    constants = np.broadcast_to(np.eye(3)[:, [0, 2]], (len(frame), 3, 2))
    common = np.concatenate([constants, times[:, :, None], fares[:, :, None] / 100], axis=2)
    mode_times = times[:, :, None] * np.eye(3)
    headway = np.zeros((len(frame), 3, 2))
    headway[:, 0, 0] = frame.TRAIN_HE / 100
    headway[:, 1, 1] = frame.SM_HE / 100
    expanded = np.concatenate([constants, mode_times, fares[:, :, None] / 100, headway], axis=2)
    return sample, choice, availability, {'canonical_4': common, 'mode_specific_8': expanded}


def center_features(features, available):
    anchor = features[np.arange(len(features)), available.argmax(axis=1)]
    return features - anchor[:, None, :]


def probability(features, available, beta):
    utils = np.where(available, features @ beta, -np.inf)
    return np.exp(utils - logsumexp(utils, axis=1, keepdims=True))


def logloss(features, available, chosen, beta):
    utils = np.where(available, features @ beta, -np.inf)
    return float(np.mean(logsumexp(utils, axis=1) - utils[np.arange(len(utils)), chosen]))


def chosen_moment(features, chosen):
    return features[np.arange(len(chosen)), chosen].mean(axis=0)


def predicted_moment(features, predicted):
    return np.einsum('ij,ijp->p', predicted, features) / len(features)


def fit_likelihood(surface, features, mask, choice, penalty):
    weights = np.full(len(features), 1 / len(features))
    moment = chosen_moment(features, choice)

    def regularized(beta):
        loss, score, curvature = surface(features, weights, moment, beta, mask)
        return loss + penalty * (beta @ beta) / 2, score + penalty * beta, curvature + penalty * np.eye(len(beta))

    out = minimize(lambda b: regularized(b)[:2], np.zeros(features.shape[2]), jac=True,
                   hess=lambda b: regularized(b)[2], method='trust-exact', options={'gtol': 1e-10})
    assert max(abs(regularized(out.x)[1])) < 1e-8, out.message
    return out.x


class DesignSupport:
    """Sparse linear programs over available entries of each choice menu."""

    def __init__(self, features, available):
        self.features = features
        self.mask = available
        self.menu, self.alternative = np.nonzero(available)
        n, _, p = features.shape
        self.n = n
        self.p = p
        self.m = len(self.menu)
        row_sums = coo_matrix((np.ones(self.m), (self.menu, np.arange(self.m))), shape=(n, self.m))
        moments = csc_matrix(features[self.menu, self.alternative].T)
        self.matrix = vstack((row_sums, moments), format='csc')
        self.uniform = available / available.sum(axis=1, keepdims=True)

    def optimize(self, rhs, extension):
        matrix = hstack((self.matrix, csc_matrix(extension[:, None])), format='csc')
        objective = np.r_[np.zeros(self.m), -1.0]
        bounds = [(0, None)] * self.m + [(0, 1)]
        sol = linprog(objective, A_eq=matrix, b_eq=rhs, bounds=bounds, options=TOL, method='highs-ipm')
        if sol.status not in (0, 2):
            sol = linprog(objective, A_eq=matrix, b_eq=rhs, bounds=bounds, options=TOL, method='highs-ds')
        if sol.success:
            error = max(abs(matrix @ sol.x - rhs))
            assert error < 1e-7
        return sol

    def frontier(self, w0, w1, b0, b1):
        optimum = self.optimize(np.r_[w0, b0], np.r_[w0-w1, b0-b1])
        assert optimum.success, optimum.message
        cut = float(optimum.x[-1])
        assert -1e-9 <= cut <= 1+1e-9
        return min(1.0, max(0.0, cut)), optimum.eqlin.marginals[self.n:]

    def interior(self, weights, target):
        uniform_mean = np.sum(self.uniform[:, :, None] * self.features * weights[:, None, None], axis=(0, 1))
        if np.linalg.matrix_rank(self.features[self.mask & (weights > 0)[:, None]]) < self.p:
            return 'rank_deficient', None
        optimum = self.optimize(np.r_[weights, target], np.r_[weights, uniform_mean])
        if optimum.status == 2:
            return 'outside', None
        assert optimum.success, optimum.message
        margin = float(optimum.x[-1])
        return ('interior' if margin > CAP_TOL else 'boundary_or_unresolved'), margin

    def normal(self, cap, normal, w0, w1, b0, b1):
        if not CAP_TOL < cap < 1-CAP_TOL:
            return {}
        assert np.linalg.norm(normal) > 0
        d = normal / np.linalg.norm(normal)
        utilities = np.where(self.mask, self.features @ d, -np.inf)
        maximum = utilities.max(axis=1)
        gaps = maximum[:, None] - utilities
        tied = gaps <= 1e-8
        first = self.features[np.arange(self.n), tied.argmax(axis=1)]
        tied_contrasts = (self.features - first[:, None, :])[tied]
        rank = int(np.linalg.matrix_rank(tied_contrasts, tol=1e-8))
        slack0 = float(w0 @ maximum - b0 @ d)
        slack1 = float(w1 @ maximum - b1 @ d)
        slope = slack0 - slack1
        assert slope > 0
        balance = (1-cap)*slack0 + cap*slack1
        assert abs(balance) < 1e-7, balance
        positive = gaps[(gaps > 1e-8) & self.mask]
        face_state = 'tied_contrasts_below_facet_rank'
        face_margin = None
        if rank == self.p-1:
            tangent = null_space(d[None, :])
            weights = (1-cap)*w0+cap*w1
            face_shift = weights @ first
            face_target = ((1-cap)*b0+cap*b1-face_shift) @ tangent
            face_support = DesignSupport((self.features-first[:, None, :]) @ tangent, tied)
            face_state, face_margin = face_support.interior(weights, face_target)
        return dict(facet_rank=rank, face_support=face_state, face_margin=face_margin,
                    normal_support_slack_error=abs(balance), normal_cap_error=abs(cap-slack0/slope),
                    normal=d.tolist(), gap_min=float(positive.min()), slack_slope=slope)


def audit(args, plan):
    sys.path.insert(0, str(args.numerics.parent))
    from audit_path import objective, solve

    assert sha(args.input) == plan['data_sha256']
    ids, choices, available, designs = load_data(args.input)
    designs = {name: center_features(f, available) for name, f in designs.items()}
    assert len(choices) == 10719 and len(np.unique(ids)) == 1191
    people = np.random.default_rng(plan['split_seed']).permutation(np.unique(ids))
    n_teacher = plan['teacher_count']
    n_eval = plan['evaluation_count']
    teacher, evaluation, pool = np.split(people, [n_teacher, n_teacher+n_eval])
    assert len(pool) == 351
    teacher_idx = np.isin(ids, teacher)
    evaluation_idx = np.isin(ids, evaluation)
    pool_idx = np.isin(ids, pool)
    assert np.all(teacher_idx.astype(int) + evaluation_idx + pool_idx == 1)
    teacher_f = designs['canonical_4']
    teachers = fit_likelihood(objective, teacher_f[teacher_idx], available[teacher_idx], choices[teacher_idx], 0)
    proxy = probability(teacher_f[pool_idx], available[pool_idx], teachers)
    assert np.all(proxy[~available[pool_idx]] == 0)
    avg_baseline = logloss(teacher_f[evaluation_idx], available[evaluation_idx], choices[evaluation_idx], teachers)
    choice = choices[pool_idx]
    mask = available[pool_idx]
    ev_mask = available[evaluation_idx]
    ev_choice = choices[evaluation_idx]
    pool_members = ids[pool_idx]
    orders = [np.random.default_rng(plan['allocation_seed'] + i).permutation(pool) for i in range(plan['allocations'])]
    input_summary = dict(rows=len(choices), teacher_people=len(teacher), teacher_rows=int(sum(teacher_idx)),
                         evaluation_people=len(evaluation), evaluation_rows=int(sum(evaluation_idx)),
                         pool_people=len(pool), pool_rows=int(sum(pool_idx)), teacher_coefs=teachers.tolist(),
                         teacher_evaluation_logloss=avg_baseline)
    args.out.mkdir(parents=True)
    allocation_records, paths, coefficients = [], [], []
    for name, original in designs.items():
        f = original[pool_idx]
        ef = original[evaluation_idx]
        support = DesignSupport(f, mask)
        human_full = fit_likelihood(objective, f, mask, choice, 0)
        full_loss = logloss(ef, ev_mask, ev_choice, human_full)
        direct_coef = human_full.tolist()
        all_human_n = len(f)
        for trial, order in enumerate(orders):
            for budget in plan['label_budgets']:
                taken = np.isin(pool_members, order[:budget])
                untouched = ~taken
                ni = int(taken.sum())
                w0, w1 = taken.astype(float) / ni, untouched.astype(float) / (all_human_n-ni)
                lf, lm, lc = f[taken], mask[taken], choice[taken]
                b0 = chosen_moment(lf, lc)
                b1 = b0 + predicted_moment(f[untouched], proxy[untouched]) - predicted_moment(lf, proxy[taken])
                labelled_state, labelled_margin = support.interior(w0, b0)
                cap, normal = support.frontier(w0, w1, b0, b1)
                geometry = support.normal(cap, normal, w0, w1, b0, b1)
                ridge = fit_likelihood(objective, lf, lm, lc, plan['ridge_penalty'])
                ridge_loss = logloss(ef, ev_mask, ev_choice, ridge)
                endpoint = 'zero_or_near_zero'
                unit_state = 'infeasible'
                endpoint_margin = None
                midpoint_state = None
                if cap > CAP_TOL:
                    midpoint_state, _ = support.interior((1-cap/2)*w0+(cap/2)*w1, (1-cap/2)*b0+(cap/2)*b1)
                    endpoint = 'positive_uncapped'
                    if cap >= 1-CAP_TOL:
                        unit_state, endpoint_margin = support.interior(w1, b1)
                        endpoint = 'unit_interior' if unit_state == 'interior' else 'unit_noninterior'
                record_number = len(allocation_records)
                record = dict(record=record_number, specification=name, allocation=trial, labelled_people=budget, labelled_tasks=ni,
                              labelled_support=labelled_state, labelled_margin=labelled_margin, endpoint=endpoint,
                              cap=cap, unit_support=unit_state, unit_margin=endpoint_margin,
                              midpoint_support=midpoint_state, ridge_loss=ridge_loss, full_human_loss=full_loss, **geometry)
                allocation_records.append(record)
                coefficients.append(dict(record=record_number, method='labelled_ridge', beta=ridge.tolist()))
                if midpoint_state != 'interior':
                    continue
                if endpoint not in ('positive_uncapped', 'unit_interior'):
                    continue
                if geometry and geometry['face_support'] == 'interior':
                    d = np.array(geometry['normal'])
                    eu = np.where(ev_mask, ef @ d, -np.inf)
                    gap = eu.max(axis=1) - eu[np.arange(len(ef)), ev_choice]
                    record['holdout_log_slope'] = float(gap.mean() / geometry['gap_min'])
                beta = ridge.copy()
                for fraction in plan['paths']['fractions_of_upper_limit']:
                    weight = cap * fraction
                    w, b = (1-weight)*w0+weight*w1, (1-weight)*b0+weight*b1
                    beta = solve(f, w, b, beta, mask)
                    _, score, curvature = objective(f, w, b, beta, mask)
                    score_norm = float(max(abs(score)))
                    eig_min = float(np.linalg.eigvalsh(curvature)[0])
                    ok = np.all(np.isfinite(beta)) and score_norm < 1e-8 and eig_min > 0.0
                    path_record = dict(record=record_number, fraction=fraction, power=weight,
                                       score=score_norm, min_curvature=eig_min,
                                       fit='score_verified' if ok else 'numerical_failure')
                    if ok:
                        cost_index = 3 if name == 'canonical_4' else 5
                        time_index = 2 if name == 'canonical_4' else 3
                        cost_beta = float(beta[cost_index])
                        time_beta = float(beta[time_index])
                        path_record.update(holdout_loss=logloss(ef, ev_mask, ev_choice, beta), cost_beta=cost_beta,
                                           swissmetro_time_beta=time_beta,
                                           swissmetro_time_value_per_hour=60*time_beta/cost_beta if cost_beta < -1e-8 else None)
                        coefficients.append(dict(record=record_number, method=f'path_{fraction:g}', beta=beta.tolist()))
                    else:
                        beta = ridge.copy()
                    paths.append(path_record)
            if (trial+1) % 16 == 0:
                print(json.dumps(dict(specification=name, allocation_count=trial+1, total=plan['allocations'])), flush=True)
        input_summary[f'full_human_coefs_{name}'] = direct_coef
    save_csv(args.out / 'allocations.csv', allocation_records)
    save_csv(args.out / 'paths.csv', paths)
    save_json(args.out / 'coefficients.json', coefficients)
    summary = summarize(plan, allocation_records, paths)
    save_json(args.out / 'summary.json', dict(
        status='complete', input_summary=input_summary, source_sha256=sha(args.input),
        plan_sha256=sha(ROOT / 'plan.json'), script_sha256=sha(Path(__file__)), numerical_helper_sha256=sha(args.numerics),
        output_sha256={name: sha(args.out / name) for name in ['allocations.csv', 'paths.csv', 'coefficients.json']},
        allocation_records=len(allocation_records), path_fits=len(paths), summaries=summary,
    ))


def summarize(plan, allocations, paths):
    lookup = {(p['record'], p['fraction']): p for p in paths}
    result = []
    for specification in plan['target_specifications']:
        for budget in plan['label_budgets']:
            rows = [r for r in allocations if r['specification'] == specification and r['labelled_people'] == budget]
            assert len(rows) == plan['allocations']
            row_summary = dict(specification=specification, budget=budget, n=len(rows),
                               endpoint_counts=dict(Counter(r['endpoint'] for r in rows)),
                               labelled_state_counts=dict(Counter(r['labelled_support'] for r in rows)),
                               labelled_ridge_median_loss=float(np.median([r['ridge_loss'] for r in rows])),
                               full_human_loss=rows[0]['full_human_loss'], path_groups={})
            for name in ('positive_uncapped', 'unit_interior'):
                group = [r for r in rows if r['endpoint'] == name]
                qualified = [r for r in group if 'holdout_log_slope' in r]
                comparisons = dict(n=len(group), single_facet_cases=len(qualified),
                                   positive_slope_cases=sum(r['holdout_log_slope'] > 0 for r in qualified), paths={})
                for fraction in plan['paths']['fractions_of_upper_limit']:
                    available = [lookup[r['record'], fraction] for r in group
                                 if (r['record'], fraction) in lookup and lookup[r['record'], fraction]['fit'] == 'score_verified']
                    vals = [q['holdout_loss'] for q in available]
                    comparisons['paths'][str(fraction)] = dict(n=len(vals),
                                                               loss_median=float(np.median(vals)) if vals else None,
                                                               negative_cost_count=sum(q['cost_beta'] < -1e-8 for q in available))
                loss_diff = [lookup[q['record'], .9]['holdout_loss'] - lookup[q['record'], .5]['holdout_loss'] for q in group
                             if all('holdout_loss' in lookup.get((q['record'], f), {}) for f in (.5, .9))]
                comparisons.update(paired_n=len(loss_diff), paired_positive=sum(x>0 for x in loss_diff),
                                   paired_loss_median=float(np.median(loss_diff)) if loss_diff else None)
                row_summary['path_groups'][name] = comparisons
            result.append(row_summary)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--numerics', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    options = parser.parse_args()
    with threadpool_limits(limits=1):
        audit(options, json.loads((ROOT / 'plan.json').read_text()))
