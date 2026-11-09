"""Exploratory known-source and constant-pattern prediction benchmarks.

The pattern hurdle is a transparent predictive baseline, not a new behavioral
model. Constant A/B/outside and nonconstant vectors are disjoint outcomes.
Training counts estimate their probabilities; prediction uses no held-out
answers. The regular component is MNL conditional on a nonconstant vector.
"""
from pathlib import Path
import argparse, itertools, json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from application_panel_comparison import Panel, fit as panel_fit
from application_support_models import build, save
import dh36_observational_analysis as old


def pattern(y):
    constant = np.all(y == y[:, :1], axis=1)
    return np.where(constant, y[:, 0] + 1, 0)  # 0 nonconstant; 1/2/3 A/B/O


def regular_distribution(x, beta):
    u = np.einsum('ntjk,k->ntj', x, beta)
    lp = u - logsumexp(u, axis=2, keepdims=True)
    p = np.exp(lp)
    lc = lp.sum(axis=1)
    log_total_constant = logsumexp(lc, axis=1)
    if np.any(log_total_constant >= 0):
        raise FloatingPointError('Nonconstant component has numerically empty support')
    logden = np.log(-np.expm1(log_total_constant))
    c = np.exp(lc)
    den = np.exp(logden)
    expected = np.einsum('ntj,ntjk->nk', p, x)
    constant_scores = x.sum(axis=1) - expected[:, None, :]
    adjustment = np.einsum('nj,njk->nk', c, constant_scores) / den[:, None]
    return lp, p, c, logden, adjustment


def conditional_objective(x, y, beta):
    lp, p, c, logden, adjustment = regular_distribution(x, beta)
    chosen = lp[np.arange(len(y))[:, None], np.arange(y.shape[1])[None], y].sum(axis=1)
    expected = np.einsum('ntj,ntjk->nk', p, x)
    observed = x[np.arange(len(y))[:, None], np.arange(y.shape[1])[None], y].sum(axis=1)
    scores = observed - expected + adjustment
    return -float((chosen - logden).sum()), -scores.sum(axis=0)


def fit_regular(x, y):
    use = pattern(y) == 0
    xx, yy = x[use], y[use]
    assert len(yy) > x.shape[-1]
    initial = np.asarray(panel_fit(xx, yy, 'MNL')['beta'])
    obj = lambda b: conditional_objective(xx, yy, b)
    bounds = [(-12., 12.)] * x.shape[-1]
    opt = minimize(obj, initial, jac=True, method='L-BFGS-B', bounds=bounds,
                   options={'ftol': 1e-14, 'gtol': 1e-7, 'maxiter': 1800, 'maxls': 50})
    val, gradient = obj(opt.x)
    probe = initial * .8 + .03
    eps = 2e-5
    directions = np.eye(len(probe))
    finite = np.array([(obj(probe + eps*e)[0] - obj(probe - eps*e)[0]) / (2*eps) for e in directions])
    error = float(np.max(np.abs(finite - obj(probe)[1])))
    assert error < 5e-5, error
    projected = gradient.copy()
    at_bound = []
    for j, (lo, hi) in enumerate(bounds):
        if min(abs(opt.x[j]-lo), abs(opt.x[j]-hi)) < 1e-5:
            at_bound.append(j)
            if (opt.x[j] <= lo+1e-5 and gradient[j] > 0) or (opt.x[j] >= hi-1e-5 and gradient[j] < 0):
                projected[j] = 0
    assert np.max(abs(projected)) < 2e-3
    return {'beta': opt.x, 'checks': {'conditional_loglik': -val, 'nonconstant_training_records': int(use.sum()),
            'gradient_max': float(np.max(abs(gradient))), 'projected_gradient_max': float(np.max(abs(projected))),
            'gradient_check_error': error, 'at_bound': at_bound, 'solver_success': bool(opt.success)}}


def gate_fit(y, source, aware, alpha):
    groups = source.astype(int) if aware else np.zeros(len(y), dtype=int)
    z = pattern(y)
    counts = np.stack([np.bincount(z[groups == g], minlength=4) for g in range(2 if aware else 1)])
    assert np.all(counts.sum(axis=1) > 0)
    weights = (counts + alpha) / (counts.sum(axis=1, keepdims=True) + 4*alpha)
    return counts, weights


def hurdle_predictions(x, beta, source, aware, weights):
    """This prediction function deliberately has no answer-vector argument."""
    groups = source.astype(int) if aware else np.zeros(len(source), dtype=int)
    w = weights[groups]
    lp, p, c, logden, _ = regular_distribution(x, beta)
    den = np.exp(logden)
    regular_marginal = (p - c[:, None, :]) / den[:, None, None]
    marginal = w[:, 0, None, None] * regular_marginal + w[:, None, 1:]
    n, T = x.shape[:2]
    refusal = np.zeros((n, T+1)); refusal[:, 0] = 1
    for t in range(T):
        q = p[:, t, 2]; previous = refusal.copy()
        refusal = previous*(1-q[:, None]); refusal[:, 1:] += previous[:, :-1]*q[:, None]
    refusal[:, 0] -= c[:, :2].sum(axis=1)
    refusal[:, T] -= c[:, 2]
    refusal = refusal / den[:, None] * w[:, :1]
    refusal[:, 0] += w[:, 1:3].sum(axis=1); refusal[:, T] += w[:, 3]
    assert marginal.min() > -1e-10 and refusal.min() > -1e-10
    assert np.max(abs(marginal.sum(axis=2)-1)) < 1e-10
    assert np.max(abs(refusal.sum(axis=1)-1)) < 1e-10
    return lp, logden, w, marginal, refusal


def score_hurdle(x, y, beta, source, aware, weights):
    lp, logden, w, marginal, refusal = hurdle_predictions(x, beta, source, aware, weights)
    z = pattern(y)
    chosen = lp[np.arange(len(y))[:, None], np.arange(y.shape[1])[None], y].sum(axis=1)
    joint = np.log(w[np.arange(len(y)), z]) + np.where(z == 0, chosen-logden, 0)
    return joint, marginal, refusal


def enumerate_check(x, beta, weights):
    x = x[:1]
    vectors = np.array(list(itertools.product(range(3), repeat=x.shape[1])), dtype=int)
    lp, _, c, logden, _ = regular_distribution(x, beta)
    z = pattern(vectors)
    chosen = lp[0, np.arange(x.shape[1])[None], vectors].sum(axis=1)
    probabilities = np.exp(np.log(weights[z]) + np.where(z == 0, chosen-logden[0], 0))
    expected = np.array([[(probabilities*(vectors[:, t] == j)).sum() for j in range(3)] for t in range(x.shape[1])])
    _, _, _, predicted, refusal = hurdle_predictions(x, beta, np.zeros(1), False, weights[None])
    refusal_exact = np.bincount((vectors == 2).sum(axis=1), weights=probabilities, minlength=x.shape[1]+1)
    errors = {'joint_total_error': abs(float(probabilities.sum())-1),
              'marginal_enumeration_error': float(np.max(abs(expected-predicted[0]))),
              'refusal_enumeration_error': float(np.max(abs(refusal_exact-refusal[0])))}
    assert max(errors.values()) < 1e-10
    return errors


def main():
    p = argparse.ArgumentParser()
    for name in ['input-dir', 'instrument', 'folds', 'out', 'private-out']:
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True); a.private_out.mkdir(parents=True, exist_ok=True)
    spec = json.loads(a.instrument.read_text()); d, provenance, *_ = old.read_exports(a.input_dir, spec)
    frozen_hashes = {'V1': 'dbe667ecc7511b69fcb5c36147f986499dbbbf23ec1d7f638eb137cfbb080e13',
                     'V2': '40b3cc7daee37b5ad0e1060b77c12cdccc9c5dff9cff1a57a75e32b8d5b7ce01'}
    assert {v['version']: v['sha256'] for v in provenance} == frozen_hashes
    forms, *_ = build(spec, d); source = d.ai_label.to_numpy(); y = d[[f'q{i}' for i in range(1, 9)]].to_numpy(int)-1
    folds = d[['participant_id']].merge(pd.read_csv(a.folds), on='participant_id', how='left', validate='one_to_one').fold.to_numpy()
    assert len(y) == 360 and not pd.isna(folds).any() and len(np.unique(folds)) == 5
    counts = []
    for group in [0, 1]:
        use = source == group
        for z, n in enumerate(np.bincount(pattern(y[use]), minlength=4)):
            counts.append({'source_group': 'AI_label' if group else 'other_labels', 'pattern': ['nonconstant','all_A','all_B','all_outside'][z], 'records': int(n)})
    pd.DataFrame(counts).to_csv(a.out/'constant_pattern_source_counts.csv', index=False)
    definitions = [('MNL_linear','linear','MNL',False), ('RI_linear','linear','RI',False),
                   ('spike_linear','linear','spike',False), ('source_ASC_MNL_linear','linear','MNL',True),
                   ('hurdle_blind_linear','linear','hurdle',False), ('hurdle_source_linear','linear','hurdle',True),
                   ('source_ASC_MNL_cells','cells','MNL',True), ('hurdle_blind_cells','cells','hurdle',False),
                   ('hurdle_source_cells','cells','hurdle',True)]
    fits = {}; diagnostics = []; predictions = {}; checks = {}; regular_cache = {}; summary = []; weights_rows = []
    for label, form, kind, aware in definitions:
        x = forms[form]
        if kind != 'hurdle' and aware:
            x = np.concatenate([x, x[:, :, :, :2]*source[:, None, None, None]], axis=3)
        rank = np.linalg.matrix_rank((x[:, :, :2]-x[:, :, 2:]).reshape(-1, x.shape[-1])); assert rank == x.shape[-1]
        joint = np.zeros(len(y)); marginal_score = np.zeros(len(y)); oo = np.zeros(len(y))
        for fold in [None] + list(np.unique(folds)):
            train = np.ones(len(y), dtype=bool) if fold is None else folds != fold
            test = np.ones(len(y), dtype=bool) if fold is None else ~train
            if kind == 'hurdle':
                key = (form, fold)
                if key not in regular_cache:
                    regular_cache[key] = fit_regular(x[train], y[train])
                result = regular_cache[key]
                ct, w = gate_fit(y[train], source[train], aware, .5)
                ll, pr, rf = score_hurdle(x[test], y[test], result['beta'], source[test], aware, w)
                if fold is None:
                    fits[label] = {'regular_component': result, 'gate_counts': ct, 'gate_weights_alpha_half': w, 'nominal_parameters': int(x.shape[-1] + (6 if aware else 3))}
                    for group in range(len(w)):
                        for z in range(4):
                            weights_rows.append({'model': label, 'source': ['other_labels','AI_label'][group] if aware else 'pooled', 'pattern': ['nonconstant','all_A','all_B','all_outside'][z], 'training_count': int(ct[group,z]), 'alpha': .5, 'weight': w[group,z]})
                    _, representatives = np.unique(x.reshape(len(x), -1), axis=0, return_index=True)
                    checks[label+'_enumeration'] = [
                        {'design_number': design+1, 'gate_group': group,
                         **enumerate_check(x[index:index+1], result['beta'], w[group])}
                        for design, index in enumerate(representatives) for group in range(len(w))]
            else:
                result = panel_fit(x[train], y[train], kind, fold is None)
                _, _, _, ll, pr, rf = Panel(x[test], y[test], kind).eval(result['beta'], True)
                if fold is None: fits[label] = result
            if fold is not None:
                joint[test] = ll; marginal_score[test] = np.log(pr[np.arange(test.sum())[:,None],np.arange(8)[None],y[test]]).sum(axis=1); oo[test] = rf[:, 8]
                diagnostics.append({'model': label, 'fold': int(fold), **result['checks']})
        predictions[label] = {'joint': joint/8, 'marginal': marginal_score/8, 'all_outside': oo}
        for group, use in [('all_360', np.ones(len(y), bool)), ('300_other_labels', ~source), ('60_AI_label', source)]:
            summary.append({'model': label, 'population': group, 'records': int(use.sum()), 'joint_nll_per_choice': -joint[use].mean()/8, 'marginal_nll_per_choice': -marginal_score[use].mean()/8, 'predicted_all_outside': oo[use].sum(), 'observed_all_outside': int(np.all(y[use] == 2, axis=1).sum())})
        print(label, [r for r in summary if r['model']==label and r['population']=='all_360'][0], flush=True)
        save(a.out/'model_fits.json', fits); save(a.out/'fold_fit_checks.json', diagnostics)
        pd.DataFrame(summary).to_csv(a.out/'source_stratified_prediction.csv', index=False)
    paired = []; weighted = []
    comparisons = [('hurdle_source_linear','hurdle_blind_linear'), ('hurdle_source_linear','spike_linear'), ('hurdle_source_linear','source_ASC_MNL_linear'), ('source_ASC_MNL_linear','MNL_linear'), ('hurdle_source_cells','hurdle_blind_cells'), ('hurdle_source_cells','source_ASC_MNL_cells')]
    for new, reference in comparisons:
        delta = predictions[new]['joint'] - predictions[reference]['joint']
        gains = {}
        for group, use in [('all_360', np.ones(len(y),bool)), ('300_other_labels',~source), ('60_AI_label',source)]:
            v = delta[use]; se = v.std(ddof=1)/np.sqrt(len(v)); gains[group] = v.mean()
            paired.append({'model':new, 'reference':reference, 'population':group, 'records':int(use.sum()), 'mean_joint_logscore_gain_per_choice':v.mean(), 'descriptive_ci_low':v.mean()-1.96*se, 'descriptive_ci_high':v.mean()+1.96*se})
        assert abs(gains['all_360'] - (5*gains['300_other_labels']+gains['60_AI_label'])/6) < 1e-12
        for q in [0., .05, .1, 1/6, .25, .5, 1.]:
            weighted.append({'model':new,'reference':reference,'hypothetical_AI_label_share':q,'standardized_joint_logscore_gain':(1-q)*gains['300_other_labels']+q*gains['60_AI_label']})
    pd.DataFrame(paired).to_csv(a.out/'paired_descriptive_prediction_gains.csv', index=False)
    pd.DataFrame(weighted).to_csv(a.out/'hypothetical_source_weighting.csv', index=False)
    pd.DataFrame(weights_rows).to_csv(a.out/'full_sample_pattern_gate_weights.csv', index=False)
    # A fixed additive-smoothing sensitivity changes gates only, without retuning utilities.
    smooth = []
    for form in ['linear','cells']:
        for aware in [False, True]:
            for alpha in [.5, 1.]:
                ll = np.zeros(len(y))
                for fold in np.unique(folds):
                    tr = folds != fold; te = ~tr; _, w = gate_fit(y[tr], source[tr], aware, alpha)
                    ll[te] = score_hurdle(forms[form][te], y[te], regular_cache[(form,fold)]['beta'], source[te], aware, w)[0]
                smooth.append({'form':form,'source_aware':aware,'alpha':alpha,'joint_nll_per_choice':-ll.mean()/8})
    pd.DataFrame(smooth).to_csv(a.out/'fixed_gate_smoothing_sensitivity.csv', index=False)
    checks.update({'records':len(y), 'choice_rows':int(y.size), 'source_AI_records':int(source.sum()),
                   'raw_hashes_match_frozen_exports': True,
                   'raw_hashes':[{'version':v['version'],'sha256':v['sha256']} for v in provenance],
                   'record_fold_ids':list(map(int,np.unique(folds))), 'heldout_patterns_used_as_predictor':False,
                   'probability_prediction_arguments':'features, fitted training parameters, source label only',
                   'public_record_prediction_file_exported':False,
                   'private_record_predictions_saved':True,'new_theory_or_identity_claim':False})
    save(a.out/'numerical_checks.json', checks)
    # Full per-record scores/folds stay private for later matched analyses.
    private = pd.DataFrame({'participant_id':d.participant_id,'fold':folds,'source_group':source.astype(int)})
    for label in predictions:
        private[label+'_joint_logscore_per_choice'] = predictions[label]['joint']
    private.to_csv(a.private_out/'heldout_predictions_private.csv', index=False)
    print('Complete: nine fixed benchmarks, source-stratified scores and fixed smoothing sensitivity.', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1): main()
