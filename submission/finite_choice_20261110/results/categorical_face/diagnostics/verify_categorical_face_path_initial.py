"""Independent original-row checks for the categorical compensated path.

Does not import the primary face/selector/distance implementation.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog
from scipy.special import expit
from public_conjoint_optimizer_audit import load_data, save
from public_conjoint_matched_audit import nuisance


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def original_support(x, y, g, m, power, kind):
    n, p = x.shape
    weights = np.r_[np.full(m, (1-power)/m), np.full(n-m, power/(n-m))]
    moment = x[:m].T@(y[:m]-power*g[:m])/m + power*x[m:].T@g[m:]/(n-m)
    if kind == 'interior':
        matrix = (x*weights[:, None]).T
        one = sparse.csr_matrix(np.ones((n, 1)))
        aub = sparse.vstack([sparse.hstack([-sparse.eye(n), one]),
                             sparse.hstack([sparse.eye(n), one])], format='csr')
        fit = linprog(np.r_[np.zeros(n), -1], A_ub=aub,
            b_ub=np.r_[np.zeros(n), np.ones(n)],
            A_eq=sparse.hstack([sparse.csr_matrix(matrix), sparse.csr_matrix((p, 1))]),
            b_eq=moment, bounds=[(0, 1)]*n+[(0, .5)], method='highs-ipm')
        assert fit.success and fit.x[-1] > 1e-8
        error = float(abs(matrix@fit.x[:-1]-moment).max())
        assert error < 1e-7
        return dict(status='interior', power=power, rank=int(np.linalg.matrix_rank(x)),
                    probability_margin=float(fit.x[-1]), residual=error,
                    probabilities=fit.x[:-1])
    fit = linprog(np.r_[-moment, weights], A_ub=sparse.hstack([sparse.csr_matrix(x), -sparse.eye(n)]),
        b_ub=np.zeros(n), bounds=[(-1, 1)]*p+[(0, None)]*n, method='highs-ipm')
    assert fit.success
    v = fit.x[:p]
    cost = float(weights@np.maximum(x@v, 0)-moment@v)
    assert cost < -1e-9
    return dict(status='outside_mean_support', power=power, direction=v,
                recession=cost)


def original_selector(x, y, g, m, mask):
    n, p = x.shape; N = n-m
    signed = x[:m]*(2*y[:m]-1)[:, None]
    linear = ((g[:m]-y[:m])[:, None]*x[:m]).mean(axis=0) - (g[m:, None]*x[m:]).mean(axis=0)
    c = np.r_[linear, np.full(N, 1/N)]
    aub = sparse.vstack([
        sparse.hstack([sparse.csr_matrix(-signed[mask]), sparse.csr_matrix((mask.sum(), N))]),
        sparse.hstack([sparse.csr_matrix(x[m:]), -sparse.eye(N)])], format='csr')
    bub = np.r_[-np.ones(mask.sum()), np.zeros(N)]
    aeq = sparse.hstack([sparse.csr_matrix(signed[~mask]), sparse.csr_matrix(((~mask).sum(), N))], format='csr')
    problem = dict(c=c, A_ub=aub, b_ub=bub, A_eq=aeq,
                   b_eq=np.zeros((~mask).sum()), bounds=[(None, None)]*p+[(0, None)]*N)
    fit = linprog(**problem, method='highs-ipm')
    assert fit.success
    return problem, fit


def independent_distance(problem, optimum, target, tol):
    # Replace v by target + delta; bound each delta by a new radius.
    p = len(target); d = len(problem['c'])
    shift = np.r_[target, np.zeros(d-p)]
    zero = sparse.csr_matrix((problem['A_ub'].shape[0], 1))
    box = sparse.hstack([sparse.eye(p), sparse.csr_matrix((p, d-p)), -np.ones((p, 1))])
    opposite = sparse.hstack([-sparse.eye(p), sparse.csr_matrix((p, d-p)), -np.ones((p, 1))])
    aub = sparse.vstack([sparse.hstack([problem['A_ub'], zero]),
                          sparse.csr_matrix(np.r_[problem['c'], 0][None, :]), box, opposite], format='csr')
    bub = np.r_[problem['b_ub']-problem['A_ub']@shift,
                optimum+tol-problem['c']@shift, np.zeros(2*p)]
    aeq = sparse.hstack([problem['A_eq'], sparse.csr_matrix((problem['A_eq'].shape[0], 1))])
    fit = linprog(np.r_[np.zeros(d), 1], A_ub=aub, b_ub=bub,
        A_eq=aeq, b_eq=problem['b_eq']-problem['A_eq']@shift,
        bounds=problem['bounds']+[(0, None)], method='highs-ipm')
    assert fit.success
    return float(fit.fun)


def exact_examples():
    records = []
    # One exposed row and two opposing neutral rows. The corrected recession
    # is negative at -e2, while strictly positive on the human cone.
    xl = np.array([[1., 0.], [0., 1.], [0., 1.]])
    xu = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    x = np.r_[xl, xu]; y = np.r_[1., 1., 0., np.zeros(4)]
    g = np.r_[.75, .9, .9, np.full(4, .5)]
    _, selector = original_selector(x, y, g, 3, np.array([True, False, False]))
    assert np.max(abs(selector.x[:2]-np.array([1., 0.]))) < 1e-9
    a_minus_e2 = np.maximum(xu@np.array([0., -1.]), 0).mean() + (
        ((g[:3]-y[:3])[:, None]*xl).mean(axis=0)
        - (g[3:, None]*xu).mean(axis=0))@np.array([0., -1.])
    assert abs(a_minus_e2+1/60) < 1e-12
    for lam in [.5, .1, .001, 1e-6, 1e-10, 1e-16]:
        b = np.array([np.log(2/lam), np.log((20-21*lam)/(20+11*lam))])
        human = -(xl.T@((2*y[:3]-1)*expit(-(2*y[:3]-1)*(xl@b))))/3
        gradient = (1-lam)*human + lam*(xu.T@(expit(xu@b)-g[3:]))/4 + lam*xl.T@(g[:3]-y[:3])/3
        assert np.linalg.norm(gradient)/lam < 1e-10
        records.append(dict(example='quasi_signed_recession', power=lam,
            first_normalized=float(b[0]/np.log(1/lam)),
            second_normalized=float(b[1]/np.log(1/lam)),
            distance_inf=float(np.max(abs(b/np.log(1/lam)-[1, 0]))),
            gradient_over_power=float(np.linalg.norm(gradient)/lam)))
    # A nonunique selector face. A returned vertex need not be the actual
    # limiting direction; the whole-face distance is the relevant property.
    xl = np.array([[1., 0.]])
    xu = np.array([[1., 1.], [1., -1.], [-1., 1.], [-1., -1.]])
    x = np.r_[xl, xu]; y = np.r_[1., np.zeros(4)]; g = np.r_[.75, np.full(4, .5)]
    problem, fit = original_selector(x, y, g, 1, np.array([True]))
    assert abs(fit.fun-.25) < 1e-12
    for v2 in [-1., 0., 1.]:
        v = np.array([1., v2]); a = np.maximum(xu@v, 0).mean()-.25*v[0]
        assert abs(a-.25) < 1e-12
    for lam in [.1, .001, 1e-8, 1e-16]:
        b = np.array([np.log(4/lam-1), 0.]); L = np.log(1/lam)
        distance = independent_distance(problem, fit.fun, b/L, 1e-10)
        assert abs(distance-abs(b[0]/L-1)) < 1e-7
        records.append(dict(example='nonunique_selector', power=lam,
            first_normalized=float(b[0]/L), second_normalized=0.,
            distance_inf=distance, gradient_over_power=None))
    return records


def run(args):
    assert not args.out.exists(), 'Use a fresh verification directory.'
    args.out.mkdir(parents=True)
    plan = json.loads(args.plan.read_text())
    certificates = json.loads((args.results/'certificates.json').read_text())
    recorded = pd.read_csv(args.results/'path_records.csv')
    hashes = pd.read_csv(args.results/'input_hashes.csv').set_index(['trial', 'prediction'])
    old = json.loads(args.coefficients.read_text())
    xall, yall, z, _, _ = load_data(args.input_dir)
    n = plan['input_prefix_rows']; m = plan['label_rows']; p = xall.shape[1]
    rng = np.random.RandomState(plan['seed']); face_rows = []; path_checks = []; supports = []
    selector_checks = []; maximum = dict(dual_annihilator_residual=0., selector_value_difference=0.,
        distance_difference=0., gradient_over_power=0., recession_identity_difference=0.,
        softplus_gap_violation=0., closest_set_primal_violation=0.)
    for trial in range(plan['allocations']):
        order = (rng.permutation(240)[:, None]*5+np.arange(5)[None]).reshape(-1)
        x, y = xall[:n][order], yall[:n][order]
        signed = x[:m]*(2*y[:m]-1)[:, None]
        detail = certificates[trial]; face = detail['face']
        mask = np.zeros(m, dtype=bool); mask[face['exposed_indices']] = True
        u = np.asarray(face['neutral_annihilator']); v = np.asarray(face['exposing_vector'])
        residual = float(abs(signed.T@u).max())
        assert residual < 1e-7 and u.min() > -1e-7
        assert not mask.any() or abs(u[mask]).max() < 1e-7
        assert not (~mask).any() or u[~mask].min() > 1-1e-7
        assert np.max(abs(signed[~mask]@v), initial=0.) < 1e-7
        maximum['dual_annihilator_residual'] = max(maximum['dual_annihilator_residual'], residual)
        for i in range(m):
            fit = linprog(np.zeros(p), A_ub=np.r_[-signed, -signed[i:i+1]],
                b_ub=np.r_[np.zeros(m), -1.], bounds=[(None, None)]*p, method='highs-ipm')
            assert fit.status in [0, 2]
            assert fit.success == mask[i]
            face_rows.append(dict(trial=trial, row=i, exposed=bool(mask[i]), LP_status=int(fit.status)))
        for pred in detail['predictions']:
            name = pred['prediction']; g, _ = nuisance(x, y, z[order], m, name=='AI')
            for key, values in [('x', x), ('y', y), ('g', g)]:
                assert hashlib.sha256(values.tobytes()).hexdigest() == hashes.loc[(trial, name), key+'_sha256']
            oldfits = [f for f in old[trial]['fits'] if f['prediction']==name]
            good = [f for f in oldfits if f.get('fit_status')=='numerical_finite_root']
            if good:
                threshold = max(f['power'] for f in good)
                sc = original_support(x, y, g, m, threshold, 'interior')
                supports.append(dict(trial=trial, prediction=name, **sc))
            else:
                sc = original_support(x, y, g, m, min(plan['powers']), 'outside')
                supports.append(dict(trial=trial, prediction=name, **sc))
            if good:
                problem, fit = original_selector(x, y, g, m, mask)
                original_value = pred['selector']['value']
                difference = abs(fit.fun-original_value)
                assert difference < 1e-7
                maximum['selector_value_difference'] = max(maximum['selector_value_difference'], difference)
                candidate = np.asarray(pred['selector']['vector'])
                direct_cost = float(np.maximum(x[m:]@candidate, 0).mean()
                    + np.mean((g[:m]-y[:m])*(x[:m]@candidate))
                    - np.mean(g[m:]*(x[m:]@candidate)))
                assert abs(direct_cost-original_value) < 1e-7
                selector_checks.append(dict(trial=trial, prediction=name, value=original_value,
                    independent_value=float(fit.fun), difference=difference,
                    original_row_recession=direct_cost))
            for f, pr in zip(oldfits, pred['paths']):
                lam = f['power']; assert pr['power']==lam
                part = recorded[(recorded.trial==trial)&(recorded.prediction==name)&(recorded.power==lam)]
                assert len(part)==1
                row = part.iloc[0]
                if f.get('fit_status') != 'numerical_finite_root':
                    assert not np.isfinite(row.get('distance_inf', np.nan))
                    path_checks.append(dict(trial=trial, prediction=name, power=lam, finite=False))
                    continue
                b = np.asarray(f['coefficient']); L = np.log(1/lam); target = b/L
                human = -signed.T@expit(-signed@b)/m
                gradient = (1-lam)*human + lam*x[m:].T@(expit(x[m:]@b)-g[m:])/(n-m) + lam*x[:m].T@(g[:m]-y[:m])/m
                ratio = float(np.linalg.norm(gradient)/lam)
                assert ratio < 1.1e-7
                maximum['gradient_over_power'] = max(maximum['gradient_over_power'], ratio)
                direct = float(np.mean(np.maximum(x[m:]@target, 0)-g[m:]*(x[m:]@target))
                               +np.mean((g[:m]-y[:m])*(x[:m]@target)))
                delta = abs(direct-row.scaled_recession)
                maximum['recession_identity_difference'] = max(maximum['recession_identity_difference'], delta)
                assert delta < 1e-8
                test = (2*yall[n:]-1)*(xall[n:]@b)
                loss = float(np.logaddexp(0, -test).mean()/L)
                hinge = float(np.maximum(-test/L, 0).mean())
                assert abs(loss-row.scaled_test_logloss) < 1e-8
                assert abs(hinge-row.scaled_test_hinge) < 1e-8
                violation = max(0., hinge-loss, loss-hinge-np.log(2)/L)
                assert violation < 1e-8
                maximum['softplus_gap_violation'] = max(maximum['softplus_gap_violation'], violation)
                closest = np.asarray(pr['closest_selector'])
                close_cost = float(np.maximum(x[m:]@closest, 0).mean()
                    + np.mean((g[:m]-y[:m])*(x[:m]@closest))-np.mean(g[m:]*(x[m:]@closest)))
                violation = max(0., close_cost-original_value-1e-8,
                    float(abs(signed[~mask]@closest).max()) if (~mask).any() else 0.,
                    float(max(0., 1-(signed[mask]@closest).min())) if mask.any() else 0.)
                maximum['closest_set_primal_violation'] = max(maximum['closest_set_primal_violation'], violation)
                assert violation < 1e-6
                distance = independent_distance(problem, original_value, target, 1e-8)
                difference = abs(distance-row.distance_inf)
                maximum['distance_difference'] = max(maximum['distance_difference'], difference)
                assert difference < 1e-5
                path_checks.append(dict(trial=trial, prediction=name, power=lam, finite=True,
                    gradient_over_power=ratio, distance_inf=distance, distance_difference=difference,
                    closest_selector_primal_violation=violation))
        if (trial+1)%10==0:
            print('independently checked allocations', trial+1, flush=True)
    pd.DataFrame(face_rows).to_csv(args.out/'row_exposure_checks.csv', index=False)
    pd.DataFrame(selector_checks).to_csv(args.out/'selector_checks.csv', index=False)
    pd.DataFrame(path_checks).to_csv(args.out/'path_checks.csv', index=False)
    save(args.out/'support_witnesses.json', supports)
    pd.DataFrame(exact_examples()).to_csv(args.out/'analytic_examples.csv', index=False)
    save(args.out/'verification.json', dict(state='passed', row_programs=len(face_rows),
        original_support_programs=len(supports), original_row_selector_programs=len(selector_checks),
        paths=len(path_checks), finite_roots=sum(r['finite'] for r in path_checks),
        maxima=maximum, input_results_sha256={p.name:digest(p) for p in sorted(args.results.iterdir()) if p.is_file()},
        source_sha256=digest(Path(__file__)), newly_fitted_paths=0,
        existing_raw_coefficients_modified=False, priority_or_journal_readiness_proven=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--coefficients', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args())
