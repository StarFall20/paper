"""Validate the boundary random-intercept stress control at higher quadrature.

The original 61-node results remain an audit trail. Only the five artificial
rank-aligned human fits require repair; human and AI primary fits are unchanged.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import roots_hermitenorm
from threadpoolctl import threadpool_limits
import dh36_observational_analysis as old
from application_support_models import build, save
from application_panel_comparison import Panel, initial_and_bounds


class StableRI(Panel):
    def __init__(self, x, y, nodes):
        super().__init__(x, y, 'RI', 61)
        self.nodes = nodes
        self.z, weights = roots_hermitenorm(nodes)
        # At very high order, negligible tail weights underflow to zero.
        self.lw = np.full_like(weights, -np.inf)
        positive = weights > 0
        self.lw[positive] = np.log(weights[positive]) - .5*np.log(2*np.pi)
        assert abs(np.exp(self.lw).sum()-1) < 1e-12


def fit_high(x, y, nodes, warm):
    obj = StableRI(x, y, nodes)
    basic = np.zeros(x.shape[-1]); basic[2] = -.2
    starts, bounds = initial_and_bounds('RI', len(basic), basic)
    starts = [np.asarray(warm)] + starts
    opts = [minimize(obj.eval, start, jac=True, method='L-BFGS-B',
                     bounds=bounds, options={'ftol': 1e-14, 'gtol': 1e-7,
                                            'maxiter': 1800, 'maxls': 50})
            for start in starts]
    opt = min(opts, key=lambda z: z.fun)
    value, grad = obj.eval(opt.x)
    projected = grad.copy()
    at_bound = []
    for j, (lo, hi) in enumerate(bounds):
        if min(abs(opt.x[j]-lo), abs(opt.x[j]-hi)) < 1e-5:
            at_bound.append(j)
        if opt.x[j] <= lo+1e-5 and grad[j] > 0:
            projected[j] = 0
        if opt.x[j] >= hi-1e-5 and grad[j] < 0:
            projected[j] = 0
    assert np.isfinite(value) and max(abs(projected)) < 2e-3
    return opt.x, {'nodes': nodes, 'loglik': -value,
                   'solver_success': bool(opt.success),
                   'projected_gradient_max': float(max(abs(projected))),
                   'at_bound': at_bound, 'bounds': bounds,
                   'start_logliks': [-float(z.fun) for z in opts]}


def main():
    p = argparse.ArgumentParser()
    for key in ['input-dir', 'instrument', 'folds', 'out']:
        p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args()
    spec = json.loads(a.instrument.read_text())
    people, *_ = old.read_exports(a.input_dir, spec)
    people = people.loc[~people.ai_label].reset_index(drop=True)
    assert len(people) == 300
    forms, *_ = build(spec, people); x = forms['linear']
    y = people[[f'q{j}' for j in range(1, 9)]].to_numpy(int)-1
    versions = people.version2.to_numpy()
    folds = people[['participant_id']].merge(pd.read_csv(a.folds),
        on='participant_id', how='left', validate='one_to_one').fold.to_numpy()
    assert not pd.isna(folds).any()
    previous = json.loads((a.out/'human_negative_control_parameters.json').read_text())
    scores = pd.read_csv(a.out/'human_negative_control_fold_scores.csv')
    repairs = []
    for fold in np.unique(folds):
        train = folds != fold; test = ~train
        yt = y[train].copy()
        for version in [0, 1]:
            use = np.where(versions[train] == version)[0]
            for task in range(8):
                before = yt[use, task].copy()
                yt[use, task] = np.sort(before)
                assert np.array_equal(np.bincount(before, minlength=3),
                                      np.bincount(yt[use, task], minlength=3))
        prior = next(z for z in previous if z['condition']=='human_rank_aligned'
                     and z['model']=='RI' and z['fold']==fold)
        beta, check = fit_high(x[train], yt, 641, prior['beta'])
        for refine in [False, True]:
            nodes = 1281 if refine else 641
            high_nodes = 2561 if refine else 1281
            if refine:
                beta, check = fit_high(x[train], yt, nodes, beta)
            low_train = StableRI(x[train], yt, nodes).eval(beta)[0]
            high_train = StableRI(x[train], yt, high_nodes).eval(beta)[0]
            lo = StableRI(x[test], y[test], nodes).eval(beta, True)
            hi = StableRI(x[test], y[test], high_nodes).eval(beta, True)
            err_train = abs(low_train-high_train)
            err_test = abs(lo[0]-hi[0])
            err_prob = float(np.max(abs(lo[4]-hi[4])))
            err_outside = float(np.max(abs(lo[5]-hi[5])))
            if max(err_train, err_test) < 1e-6 and max(err_prob, err_outside) < 1e-8:
                break
        else:
            raise ValueError('High-node quadrature tolerance failed')
        use_score = ((scores.condition=='human_rank_aligned') &
                     (scores.model=='RI') & (scores.fold==fold))
        assert use_score.sum() == 1
        scores.loc[use_score, 'joint_logscore_sum'] = float(lo[3].sum())
        marginal = np.log(lo[4][np.arange(test.sum())[:, None],
                              np.arange(8)[None], y[test]]).sum()
        scores.loc[use_score, 'marginal_logscore_sum'] = float(marginal)
        scores.loc[use_score, 'predicted_all_outside'] = float(lo[5][:, 8].sum())
        repairs.append({'fold': int(fold), 'beta': beta, 'fit': check,
            'service_intercept_sd': float(np.exp(beta[-1])),
            'verification_nodes': high_nodes,
            'training_loglik_absolute_difference': float(err_train),
            'heldout_loglik_absolute_difference': float(err_test),
            'marginal_probability_max_difference': err_prob,
            'refusal_probability_max_difference': err_outside,
            'interpretation': 'Constrained artificial stress fit; do not interpret as an unconstrained population heterogeneity estimate'})
        print('validated rank-aligned fold', int(fold), 'nodes', nodes,
              'training difference', err_train, flush=True)
    summary = scores.groupby(['condition','seed','model'],as_index=False).agg(
        records=('human_heldout_records','sum'), joint_logscore=('joint_logscore_sum','sum'),
        marginal_logscore=('marginal_logscore_sum','sum'),
        predicted_all_outside=('predicted_all_outside','sum'))
    assert summary.records.eq(300).all()
    summary['joint_nll_per_choice'] = -summary.joint_logscore/2400
    summary['marginal_nll_per_choice'] = -summary.marginal_logscore/2400
    scores.to_csv(a.out/'human_negative_control_validated_fold_scores.csv',index=False)
    summary.to_csv(a.out/'human_negative_control_validated_summary.csv',index=False)
    save(a.out/'human_rank_alignment_quadrature_audit.json', {
        'reason': '61-versus-161 discrepancies of 1.33 to 4.52 total training log likelihood units in artificial rank-aligned RI fits',
        'original_61_node_outputs_retained': True, 'fits': repairs,
        'all_five_boundary_stress_fits_validated': True,
        'AI_or_primary_human_fits_changed': False})
    print(summary.groupby(['condition','model']).joint_nll_per_choice.agg(['mean','min','max']).to_string(),flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
