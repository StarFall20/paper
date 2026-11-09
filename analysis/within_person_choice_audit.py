"""Classical conditional fixed-effect logit and association diagnostics.

Conditions on each record's alternative counts to eliminate stable alternative
intercepts. This is the established Chamberlain conditional likelihood, not a
new model or an unconditional prediction method.
"""
from pathlib import Path
from itertools import product, combinations
from math import factorial
import argparse
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from scipy.stats import fisher_exact, norm
from threadpoolctl import threadpool_limits
import dh36_observational_analysis as old
from application_support_models import build, save


class CountConditional:
    def __init__(self, x, y):
        assert x.shape[:2] == (8, 3) and y.shape[1] == 8
        self.x, self.y = x, y
        self.paths = np.array(list(product(range(3), repeat=8)), dtype=int)
        self.counts = np.stack([(self.paths == j).sum(1) for j in range(3)], 1)
        self.path_features = x[np.arange(8)[None], self.paths].sum(1)
        self.observed_features = x[np.arange(8)[None], y].sum(1)
        counts = np.stack([(y == j).sum(1) for j in range(3)], 1)
        self.groups = []
        for count in np.unique(counts, axis=0):
            use = np.where((counts == count).all(1))[0]
            candidate = self.path_features[(self.counts == count).all(1)]
            assert len(candidate) == factorial(8)//np.prod([factorial(int(c)) for c in count])
            self.groups.append((use, candidate, count))
        self.informative = sum(len(use) for use, f, _ in self.groups if len(f) > 1)

    def evaluate(self, b, details=False):
        k = len(b); ll = np.zeros(len(self.y)); score = np.zeros((len(self.y), k)); h = np.zeros((k, k))
        for use, features, _ in self.groups:
            utility = features @ b; weights = softmax(utility)
            mean = weights @ features
            ll[use] = self.observed_features[use] @ b-logsumexp(utility)
            score[use] = self.observed_features[use]-mean
            centered = features-mean
            h += len(use)*centered.T @ (weights[:, None]*centered)
        result = (-float(ll.sum()), -score.sum(0))
        return (*result, h, score, ll) if details else result

    def direct_conditioning_check(self):
        # Independent check from normalized products of task probabilities with
        # two arbitrary fixed intercept vectors, rather than feature scores.
        b = np.linspace(-.17, .23, self.x.shape[-1]); maximum = 0.
        for alpha in [np.array([-1.3, .7, 0.]), np.array([2.1, -1.9, 0.])]:
            u = self.x @ b+alpha
            lp = u-logsumexp(u, axis=1, keepdims=True)
            path_lp = lp[np.arange(8)[None], self.paths].sum(1)
            for _, features, count in self.groups:
                keep = (self.counts == count).all(1)
                full = path_lp[keep]-logsumexp(path_lp[keep])
                conditional = features @ b-logsumexp(features @ b)
                maximum = max(maximum, float(max(abs(full-conditional))))
        assert maximum < 1e-11
        return maximum


def fit_model(obj):
    k = obj.x.shape[-1]
    opts = [minimize(obj.evaluate, np.full(k, start), method='BFGS', jac=True,
                     options={'gtol': 1e-8, 'maxiter': 1200}) for start in [0., -.2, .2]]
    opt = min(opts, key=lambda z: z.fun); b = opt.x.copy()
    for _ in range(5):
        val, g, h, _, _ = obj.evaluate(b, True)
        if max(abs(g)) < 1e-7:
            break
        delta = np.linalg.solve(h, g)
        step = 1.
        while obj.evaluate(b-step*delta)[0] > val+1e-10:
            step /= 2
            if step < 1e-8:
                raise ValueError('Newton refinement failed')
        b -= step*delta
    val, g, h, scores, _ = obj.evaluate(b, True)
    eig = np.linalg.eigvalsh(h); assert min(eig) > 1e-7 and max(abs(g)) < 1e-5
    inverse = np.linalg.inv(h)
    covariance = obj.informative/(obj.informative-1)*inverse @ scores.T @ scores @ inverse
    # Finite differences at an interior nonoptimum validate both derivatives.
    test = np.linspace(-.11, .19, k); eps = 1e-5; eye = np.eye(k)
    fd = np.array([(obj.evaluate(test+eps*e)[0]-obj.evaluate(test-eps*e)[0])/(2*eps) for e in eye])
    fd_h = np.column_stack([(obj.evaluate(test+eps*e)[1]-obj.evaluate(test-eps*e)[1])/(2*eps) for e in eye])
    error_g = float(max(abs(fd-obj.evaluate(test)[1])))
    error_h = float(np.max(abs(fd_h-obj.evaluate(test, True)[2])))
    assert error_g < 2e-5 and error_h < 2e-5
    return {'beta': b, 'covariance': covariance, 'checks': {
        'human_records': len(obj.y), 'informative_human_records': obj.informative,
        'parameters': k, 'observed_information_rank': int(np.linalg.matrix_rank(h)),
        'minimum_information_eigenvalue': float(min(eig)),
        'gradient_max': float(max(abs(g))), 'gradient_difference_max': error_g,
        'hessian_difference_max': error_h,
        'conditional_loglik': -val, 'start_logliks': [-float(z.fun) for z in opts],
        'direct_task_probability_conditioning_error': obj.direct_conditioning_check(),
        'full_enumeration_sequences': len(obj.paths), 'conditional_strata_used': len(obj.groups),
        'unconditional_prediction_target': False}}


def contrast(fit, c):
    effect = float(c @ fit['beta']); se = float(np.sqrt(c @ fit['covariance'] @ c))
    return {'estimate': effect, 'record_sandwich_se': se,
            'ci_low': effect-norm.ppf(.975)*se, 'ci_high': effect+norm.ppf(.975)*se}


def main():
    p = argparse.ArgumentParser()
    for key in ['input-dir', 'instrument', 'classification', 'out']:
        p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    role = json.loads(a.classification.read_text()); assert role['synthetic_records']['total'] == 60
    spec = json.loads(a.instrument.read_text())
    people, provenance, *_ = old.read_exports(a.input_dir, spec)
    forms, cell_names, *_ = build(spec, people)
    humans = ~people.ai_label.to_numpy(); ai = ~humans
    assert humans.sum() == 300 and ai.sum() == 60
    y = people[[f'q{t}' for t in range(1, 9)]].to_numpy(int)-1
    assert np.all(np.all(y[ai] == y[ai, :1], axis=1))
    versions = people.version2.to_numpy()
    # The remaining attributes are identical by task across versions. Individual
    # alternative intercepts absorb all version/label-specific constants.
    definitions = [('linear', forms['linear'][..., 2:5], ['price_per_10_CNY', 'G1_vs_G0', 'G2_vs_G0']),
                   ('observed_cells', forms['cells'][..., 4:], cell_names[4:])]
    fits = {}; rows = []; contrasts = []; invariance = []
    for name, xall, names in definitions:
        assert np.max(abs(xall-xall[:1])) < 1e-12
        x = xall[0]; obj = CountConditional(x, y[humans]); mixed = CountConditional(x, y)
        result = fit_model(obj); result['names'] = names; fits[name] = result
        for b in [np.zeros(len(names)), result['beta'], np.linspace(-.2, .3, len(names))]:
            hv, hg = obj.evaluate(b); mv, mg = mixed.evaluate(b)
            err = abs(hv-mv); graderr = float(max(abs(hg-mg)))
            assert err < 1e-9 and graderr < 1e-9
            invariance.append({'model': name, 'objective_difference': err, 'gradient_difference': graderr})
        for j, term in enumerate(names):
            c = np.eye(len(names))[j]; rows.append({'model': name, 'term': term, **contrast(result, c)})
        c = np.zeros(len(names))
        if name == 'linear':
            c[names.index('G2_vs_G0')] = 1; c[names.index('G1_vs_G0')] = -1
        else:
            c[names.index('G2_fee_89')] = 1
        contrasts.append({'model': name, 'contrast': 'G2_minus_G1_at_CNY89', **contrast(result, c)})
        print(name, contrasts[-1], result['checks'], flush=True)
    pair_rows = []
    for version in [0, 1]:
        yy = y[humans & (versions == version)]; z = (yy != 2).astype(int)
        for t, s in combinations(range(8), 2):
            table = np.zeros((2, 2), int)
            np.add.at(table, (z[:, t], z[:, s]), 1)
            odds, pvalue = fisher_exact(table, alternative='less')
            covariance = float(np.mean(z[:, t]*z[:, s])-np.mean(z[:, t])*np.mean(z[:, s]))
            pair_rows.append({'version': version+1, 'task_1': t+1, 'task_2': s+1,
                'records': len(z), 'n00': table[0, 0], 'n01': table[0, 1],
                'n10': table[1, 0], 'n11': table[1, 1], 'participation_covariance': covariance,
                'odds_ratio': odds, 'one_sided_Fisher_p': pvalue})
    pairs = pd.DataFrame(pair_rows); assert len(pairs) == 56
    order = np.argsort(pairs.one_sided_Fisher_p.to_numpy())
    adjusted = np.minimum(1., np.maximum.accumulate((56-np.arange(56))*pairs.one_sided_Fisher_p.to_numpy()[order]))
    holm = np.empty(56); holm[order] = adjusted; pairs['Holm_family_p'] = holm
    pairs.to_csv(a.out/'participation_association_tests.csv', index=False)
    pd.DataFrame(rows).to_csv(a.out/'conditional_fixed_effect_coefficients.csv', index=False)
    pd.DataFrame(contrasts).to_csv(a.out/'conditional_fixed_effect_contrasts.csv', index=False)
    save(a.out/'conditional_fixed_effect_fits.json', fits)
    save(a.out/'audit_checks.json', {'canonical_human_records': 300, 'AI_pretests': 60,
        'raw_hashes': {z['version']: z['sha256'] for z in provenance},
        'conditional_objective_AI_invariance': invariance,
        'pairwise_tests': 56, 'negative_sample_covariances': int((pairs.participation_covariance < 0).sum()),
        'minimum_raw_one_sided_p': float(pairs.one_sided_Fisher_p.min()),
        'minimum_Holm_p': float(pairs.Holm_family_p.min()),
        'Holm_rejections_at_005': int((pairs.Holm_family_p < .05).sum()),
        'raw_responses_changed': False, 'new_method_claim': False,
        'interpretation': 'Necessary-association diagnostics plus classical shared-slope conditional logit. Nonrejection establishes neither full latent-model adequacy nor sufficient power.'})


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
