"""Finite-design consequence of the existing compensated-logit path.

Classical separation faces and margin limits are attributed in the report.
No new response, fit, estimator, or population claim is generated here.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog
from public_conjoint_optimizer_audit import load_data, save
from public_conjoint_matched_audit import nuisance
from vanishing_power_choice import rescaled_support


OPTIONS = dict(primal_feasibility_tolerance=1e-9,
               dual_feasibility_tolerance=1e-9)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def maximal_face(matrix):
    m, p = matrix.shape
    fit = linprog(np.r_[np.zeros(p), -np.ones(m)],
                  A_ub=np.c_[-matrix, np.eye(m)], b_ub=np.zeros(m),
                  bounds=[(None, None)] * p + [(0, 1)] * m,
                  method='highs', options=OPTIONS)
    assert fit.success, fit.message
    slacks = fit.x[p:]
    assert np.max(np.minimum(abs(slacks), abs(slacks-1))) < 1e-7
    exposed = slacks > .5
    vector = fit.x[:p]
    annihilator = -fit.ineqlin.marginals
    margin = matrix @ vector
    assert margin.min() > -1e-7
    assert not exposed.any() or margin[exposed].min() > 1-1e-7
    assert np.max(abs(matrix.T @ annihilator)) < 1e-7
    assert np.min(annihilator) > -1e-7
    assert not (~exposed).any() or annihilator[~exposed].min() > 1-1e-7
    assert not exposed.any() or abs(annihilator[exposed]).max() < 1e-7
    return exposed, dict(exposed_indices=np.where(exposed)[0].tolist(),
                        neutral_indices=np.where(~exposed)[0].tolist(),
                        exposing_vector=vector, neutral_annihilator=annihilator,
                        primal_objective=float(-fit.fun),
                        signed_margins=margin,
                        type='overlap' if not exposed.any() else
                             'complete' if exposed.all() else 'quasi')


def selector_problem(x, y, g, m, exposed):
    """Compress repeated auxiliary profiles without changing the objective."""
    p = x.shape[1]
    xu, count = np.unique(x[m:], axis=0, return_counts=True)
    weights = count / (len(x)-m)
    linear = x[:m].T @ (g[:m]-y[:m])/m - x[m:].T @ g[m:]/(len(x)-m)
    signed = x[:m] * (2*y[:m]-1)[:, None]
    k = len(xu)
    objective = np.r_[linear, weights]
    inequalities = sparse.vstack([
        sparse.hstack([-sparse.csr_matrix(signed[exposed]), sparse.csr_matrix((exposed.sum(), k))]),
        sparse.hstack([sparse.csr_matrix(xu), -sparse.eye(k)])
    ], format='csr')
    rhs = np.r_[-np.ones(exposed.sum()), np.zeros(k)]
    equality = sparse.hstack([sparse.csr_matrix(signed[~exposed]),
                              sparse.csr_matrix(((~exposed).sum(), k))], format='csr')
    bounds = [(None, None)] * p + [(0, None)] * k
    return dict(c=objective, A_ub=inequalities, b_ub=rhs,
                A_eq=equality, b_eq=np.zeros((~exposed).sum()),
                bounds=bounds), xu, weights, linear


def recession(x, y, g, m, v):
    u = x @ v
    return float(np.maximum(u[m:], 0).mean() - np.mean(g[m:]*u[m:])
                 + np.mean((g[:m]-y[:m])*u[:m]))


def solve_selector(problem, p):
    fit = linprog(**problem, method='highs', options=OPTIONS)
    if not fit.success:
        return fit, dict(status='LP_'+str(fit.status), message=fit.message)
    dual = float(problem['b_ub'] @ fit.ineqlin.marginals
                 + problem['b_eq'] @ fit.eqlin.marginals)
    stationarity = (problem['c'] - problem['A_ub'].T @ fit.ineqlin.marginals
                    - problem['A_eq'].T @ fit.eqlin.marginals
                    - fit.lower.marginals - fit.upper.marginals)
    assert abs(fit.fun-dual) < 1e-7
    assert np.max(abs(stationarity)) < 1e-7
    return fit, dict(status='optimal', value=float(fit.fun), vector=fit.x[:p],
                     epigraph=fit.x[p:], inequality_dual=fit.ineqlin.marginals,
                     equality_dual=fit.eqlin.marginals,
                     lower_dual=fit.lower.marginals,
                     dual_objective=dual,
                     primal_dual_gap=float(fit.fun-dual),
                     stationarity_max=float(np.max(abs(stationarity))))


def distance_to_set(problem, optimal_value, target, tolerance):
    """Distance in infinity norm to the entire numerical optimal LP face."""
    p = len(target)
    d = len(problem['c'])
    pad = sparse.csr_matrix((problem['A_ub'].shape[0], 1))
    signed = sparse.hstack([sparse.eye(p), sparse.csr_matrix((p, d-p)),
                            -np.ones((p, 1))], format='csr')
    opposite = sparse.hstack([-sparse.eye(p), sparse.csr_matrix((p, d-p)),
                              -np.ones((p, 1))], format='csr')
    aub = sparse.vstack([sparse.hstack([problem['A_ub'], pad]),
                         sparse.csr_matrix(np.r_[problem['c'], 0][None, :]),
                         signed, opposite], format='csr')
    bub = np.r_[problem['b_ub'], optimal_value+tolerance, target, -target]
    aeq = sparse.hstack([problem['A_eq'], sparse.csr_matrix((problem['A_eq'].shape[0], 1))], format='csr')
    fit = linprog(np.r_[np.zeros(d), 1], A_ub=aub, b_ub=bub,
                  A_eq=aeq, b_eq=problem['b_eq'],
                  bounds=problem['bounds']+[(0, None)],
                  method='highs', options=OPTIONS)
    assert fit.success, fit.message
    return float(fit.fun), fit.x[:p]


def run(args):
    assert not args.out.exists(), 'Use a fresh output directory.'
    args.out.mkdir(parents=True)
    plan = json.loads(args.plan.read_text())
    xall, yall, zall, _, _ = load_data(args.input_dir)
    m = plan['label_rows']; n = plan['input_prefix_rows']; p = xall.shape[1]
    previous = json.loads(args.coefficients.read_text())
    assert len(previous) == plan['allocations']
    rng = np.random.RandomState(plan['seed'])
    faces = []; selectors = []; rows = []
    details = []; digests = []
    for trial in range(plan['allocations']):
        order = (rng.permutation(240)[:, None]*5+np.arange(5)[None]).reshape(-1)
        x, y, z = xall[:n][order], yall[:n][order], zall[order]
        old = previous[trial]
        assert old['trial'] == trial
        signed = x[:m]*(2*y[:m]-1)[:, None]
        exposed, face = maximal_face(signed)
        face.update(trial=trial, exposed_rows=int(exposed.sum()), neutral_rows=int((~exposed).sum()),
                    labelled_rank=int(np.linalg.matrix_rank(x[:m])))
        faces.append({k: v for k, v in face.items() if k in
                      ['trial', 'type', 'exposed_rows', 'neutral_rows', 'labelled_rank', 'primal_objective']})
        detail = dict(trial=trial, face=face, predictions=[])
        for name in plan['prediction_routines']:
            g, fallback = nuisance(x, y, z, m, name == 'AI')
            oldfits = [f for f in old['fits'] if f['prediction'] == name]
            assert sorted(f['power'] for f in oldfits) == sorted(plan['powers'])
            # Recompute the original fixed-power event; do not infer it from a solver flag.
            state = rescaled_support(x, y, g, m, min(plan['powers']))
            endpoint = next(f for f in oldfits if f['power'] == min(plan['powers']))
            assert state['status'] == endpoint['support']['status']
            digest = dict(trial=trial, prediction=name,
                          x_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                          y_sha256=hashlib.sha256(y.tobytes()).hexdigest(),
                          g_sha256=hashlib.sha256(g.tobytes()).hexdigest())
            digests.append(digest)
            one = dict(prediction=name, support=state, nuisance_fallbacks=fallback, paths=[])
            sr = dict(trial=trial, prediction=name, face_type=face['type'],
                      exposed_rows=int(exposed.sum()), support_status=state['status'],
                      eligible=state['status']=='interior')
            problem = None; fit = None
            if sr['eligible']:
                problem, xu, weights, linear = selector_problem(x, y, g, m, exposed)
                fit, certificate = solve_selector(problem, p)
                assert fit.success, certificate
                one.update(selector=certificate, unique_auxiliary_rows=len(xu))
                sr.update(selector_status=certificate['status'], value=certificate['value'],
                          primal_dual_gap=certificate['primal_dual_gap'])
            else:
                sr['selector_status'] = 'not_eligible'
            selectors.append(sr)
            for oldfit in oldfits:
                lam = oldfit['power']; L = np.log(1/lam)
                original_status = oldfit.get('fit_status', 'not_fitted_'+oldfit['support']['status'])
                row = dict(trial=trial, prediction=name, face_type=face['type'],
                           exposed_rows=int(exposed.sum()), power=lam,
                           original_fit_status=original_status, selector_eligible=sr['eligible'])
                pr = dict(power=lam, original_fit_status=original_status)
                if original_status == 'numerical_finite_root':
                    beta = np.asarray(oldfit['coefficient']); target = beta/L
                    # Recession is evaluated in original, uncompressed row coordinates.
                    u = xall[n:] @ beta
                    risk = float(np.logaddexp(0, -(2*yall[n:]-1)*u).mean())
                    hinge = float(np.maximum(-(2*yall[n:]-1)*(xall[n:]@target), 0).mean())
                    row.update(coefficient_norm=float(np.linalg.norm(beta)),
                               scaled_recession=recession(x, y, g, m, target),
                               scaled_test_logloss=risk/L, scaled_test_hinge=hinge,
                               softplus_gap=risk/L-hinge)
                    assert -1e-10 <= row['softplus_gap'] <= np.log(2)/L+1e-10
                    if problem is not None:
                        distance, closest = distance_to_set(problem, fit.fun, target,
                            plan['numerical_conventions']['selector_face_objective_tolerance'])
                        row.update(distance_inf=distance,
                                   normalized_distance_inf=distance/(1+np.linalg.norm(closest, ord=np.inf)),
                                   exposed_margin_deficit=float(max(0., 1-(signed[exposed]@target).min())) if exposed.any() else 0.,
                                   neutral_margin_max=float(abs(signed[~exposed]@target).max()) if (~exposed).any() else 0.,
                                   closest_selector_test_hinge=float(np.maximum(-(2*yall[n:]-1)*(xall[n:]@closest), 0).mean()))
                        pr['closest_selector'] = closest
                rows.append(row); one['paths'].append(pr)
            detail['predictions'].append(one)
        details.append(detail)
        if (trial+1)%10 == 0:
            print('classified allocations', trial+1, 'of', plan['allocations'], flush=True)
    pd.DataFrame(faces).to_csv(args.out/'face_summary.csv', index=False)
    pd.DataFrame(selectors).to_csv(args.out/'selector_summary.csv', index=False)
    pd.DataFrame(rows).to_csv(args.out/'path_records.csv', index=False)
    save(args.out/'certificates.json', details)
    pd.DataFrame(digests).to_csv(args.out/'input_hashes.csv', index=False)
    save(args.out/'run_status.json', dict(state='complete', face_allocations=len(faces),
        predictor_allocations=len(selectors), path_records=len(rows),
        plan_sha256=sha(args.plan), source_sha256=sha(Path(__file__)),
        existing_coefficients_sha256=sha(args.coefficients), new_fits=0,
        new_human_or_AI_responses=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--coefficients', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args())
