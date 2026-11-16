"""Independent objective/metric reconciliation for the menu-quality follow-up.

The former moment objective checks every retained numerical root. A larger,
separately coded Gaussian integral checks a fixed subset and the largest norms.
These are numerical validations, not external proof or originality reviews.
"""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.special import logsumexp, ndtri, softmax
from scipy.stats import qmc
from threadpoolctl import threadpool_limits
from menu_corrected_logit import MenuLoss, menu_features
from menu_probability_quality import StableMenuLoss, GaussianMenuRisk, source_closure


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def software_checks(out):
    rng = np.random.default_rng(2026111532)
    errors = {name: 0. for name in ['objective', 'gradient', 'hessian', 'fd_gradient', 'fd_hessian', 'pool_identity']}
    cases = 0
    for case in range(15):
        m = 6 + case % 4
        n, J = 2*m, 3
        f = menu_features(rng.normal(size=(n, J, 2+case % 5)))
        y = rng.integers(J, size=n)
        g = np.eye(J)[rng.integers(J, size=n)]
        beta = rng.normal(size=f.shape[-1])/3
        eye, step = np.eye(len(beta)), 1e-5
        for power in [0., .2195760406786864, 1.]:
            old, new = MenuLoss(f, y, g, m, power), StableMenuLoss(f, y, g, m, power)
            errors['objective'] = max(errors['objective'], abs(old.value(beta)[0]-new.value(beta)[0]))
            errors['gradient'] = max(errors['gradient'], float(np.max(abs(old.value(beta)[1]-new.value(beta)[1]))))
            errors['hessian'] = max(errors['hessian'], float(np.max(abs(old.hessian(beta)-new.hessian(beta)))))
            fd = np.array([(new.value(beta+e*step)[0]-new.value(beta-e*step)[0])/(2*step) for e in eye])
            fh = np.column_stack([(new.value(beta+e*step)[1]-new.value(beta-e*step)[1])/(2*step) for e in eye])
            errors['fd_gradient'] = max(errors['fd_gradient'], float(np.max(abs(fd-new.value(beta)[1]))))
            errors['fd_hessian'] = max(errors['fd_hessian'], float(np.max(abs(fh-new.hessian(beta)))))
            cases += 1
        pool_g = np.vstack([np.eye(J)[y[:m]], g[m:]])
        old = MenuLoss(f, y, pool_g, m, .5)
        new = StableMenuLoss(f, y, g, m, 0., pooling=True)
        errors['pool_identity'] = max(errors['pool_identity'], abs(old.value(beta)[0]-new.value(beta)[0]),
                                      float(np.max(abs(old.value(beta)[1]-new.value(beta)[1]))))
    risk = GaussianMenuRisk(3, 1.5, 11, 8, 2026111531)
    zero = np.zeros(7)
    truth = zero.copy()
    truth[2] = 1.5
    z, t = risk.evaluate(zero), risk.evaluate(truth)
    assert abs(z['human_log_loss']-np.log(3)) < 1e-12
    assert abs(z['human_Brier_loss']-2/3) < 1e-12
    assert abs(t['human_log_loss']-risk.truth_loss) < 1e-12
    assert abs(t['human_Brier_loss']-risk.truth_brier) < 1e-12
    # Rotate an actual multi-coordinate Gaussian contrast onto its fitted index.
    rotation_error = 0.
    for q in [2, 5, 12, 26, 52]:
        w = rng.normal(size=(32, 3, q))
        b = rng.normal(size=q)
        direction = b/np.linalg.norm(b)
        e = np.eye(q)[0]
        v = direction-e
        matrix = np.eye(q)-2*np.outer(v, v)/(v@v)
        rotated = w@matrix
        rotation_error = max(rotation_error, float(np.max(abs(w@b-np.linalg.norm(b)*rotated[..., 0]))))
    assert max(errors.values()) < 1e-7 and rotation_error < 1e-12
    result = dict(state='passed', objective_fixtures=cases, errors=errors,
                  Gaussian_rotation_error=rotation_error, exact_zero_and_truth_risk_identities=True,
                  new_model_cohorts=0, independent_expert_review=False)
    save(out/'software_checks.json', result)
    return result


def large_integral(beta, gamma=1.5, bits=14, scrambles=8, seed=2026111591):
    J = 3
    a = np.r_[beta[:2], 0.]
    a -= a.mean()
    b, scale = beta[2], np.linalg.norm(beta[3:])
    values = []
    for scramble in range(scrambles):
        nodes = ndtri(qmc.Sobol(6, scramble=True, seed=seed+scramble).random_base2(bits))
        signal = nodes[:, :J]
        signal -= signal.mean(axis=1, keepdims=True)
        truth = softmax(gamma*signal, axis=1)
        noise = nodes[:, J:]
        noise -= noise.mean(axis=1, keepdims=True)
        # Pair +/- noise to integrate its linear contribution exactly. Centered
        # constants have zero expected human utility by alternative symmetry.
        positive = a+b*signal+scale*noise
        negative = a+b*signal-scale*noise
        partition = (logsumexp(positive, axis=1)+logsumexp(negative, axis=1))/2
        loss = partition-b*np.sum(truth*signal, axis=1)
        pp, pn = softmax(positive, axis=1), softmax(negative, axis=1)
        brier = 1+(np.sum(pp**2-2*truth*pp, axis=1)+np.sum(pn**2-2*truth*pn, axis=1))/2
        values.append([float(loss.mean()), float(brier.mean())])
    values = np.asarray(values)
    return values.mean(axis=0), values.std(axis=0, ddof=1)/np.sqrt(scrambles)


def verify(args):
    repo = Path(__file__).resolve().parents[1]
    plan = json.loads(args.plan.read_text())
    original = json.loads((repo/plan['source_plan']).read_text())
    checkpoint = json.loads(gzip.decompress((args.results/'checkpoint.json.gz').read_bytes()))
    status = json.loads((args.results/'quality_status.json').read_text())
    assert checkpoint['state'] == status['state'] == 'complete'
    assert checkpoint['plan_sha256'] == sha(args.plan)
    assert checkpoint['source_hashes'] == source_closure()
    rows = checkpoint['records']
    assert len(rows) == 15000 and len(checkpoint['completed_cohorts']) == 1000
    lookup = {(r['label_menus'], r['trial'], r['orthogonal_slopes'], r['method']): r for r in rows}
    assert len(lookup) == len(rows)
    reference = pd.read_csv(repo/plan['source_trials'])
    for record in reference.itertuples():
        assert lookup[(record.label_menus, record.trial, record.orthogonal_slopes, record.method)]['support_status'] == record.status
    frame = pd.read_csv(args.results/'quality_records.csv')
    assert len(frame) == len(rows) and frame.groupby(['label_menus', 'orthogonal_slopes', 'method']).size().eq(500).all()
    summary = pd.read_csv(args.results/'quality_summary.csv')
    assert len(summary) == 30
    numeric = [r for r in rows if r['fit_status'] == 'numerical_finite_root']
    chosen = set(range(0, len(numeric), 239))
    chosen.update(sorted(range(len(numeric)), key=lambda i:numeric[i]['coefficient_norm'], reverse=True)[:20])
    rng = np.random.default_rng(original['seed'])
    checks = 0
    max_score = max_objective_error = max_parameter_error = 0.
    for m in original['label_menus']:
        N, J = int(original['tau']*m), original['alternatives']
        n = m+N
        dimensions = sorted({max(1, round(m*k)) for k in original['dimension_ratios']})
        for trial in range(500):
            S = rng.normal(size=(n, J, 1))
            W = rng.normal(size=(n, J, max(dimensions)))
            mu = softmax(original['gamma']*S[..., 0], axis=1)
            y = np.sum(rng.random(n)[:, None]>np.cumsum(mu, axis=1), axis=1)
            z = (y+(rng.random(n)<original['epsilon'])*rng.integers(1, J, size=n)) % J
            g = np.eye(J)[z]
            for q in dimensions:
                f = menu_features(np.concatenate([S, W[..., :q]], axis=2))
                truth = np.zeros(f.shape[-1])
                truth[J-1] = original['gamma']
                for method in plan['methods']:
                    record = lookup[(m, trial, q, method)]
                    if record['fit_status'] != 'numerical_finite_root':
                        continue
                    beta = np.asarray(record['beta'])
                    if method == 'hard_pool':
                        gg = np.vstack([np.eye(J)[y[:m]], g[m:]])
                        old = MenuLoss(f, y, gg, m, N/n)
                    else:
                        old = MenuLoss(f, y, g, m, record['power'])
                    value, gradient = old.value(beta)
                    value += record['penalty']*(beta@beta)/2
                    gradient += record['penalty']*beta
                    error = float(abs(np.sum((beta-truth)**2)-record['squared_coefficient_error']))
                    max_parameter_error = max(max_parameter_error, error)
                    max_score = max(max_score, float(np.linalg.norm(gradient)))
                    max_objective_error = max(max_objective_error, abs(value-record['objective']))
                    assert np.linalg.norm(gradient) < 1e-8, (m, trial, q, method)
                    assert abs(value-record['objective']) < 1e-8*(1+abs(value))
                    assert error <= 1e-10*(1+record['squared_coefficient_error'])
                    checks += 1
    assert checks == len(numeric)
    integral_checks = []
    for i in sorted(chosen):
        record = numeric[i]
        beta = np.asarray(record['beta'])
        means, ses = large_integral(beta)
        difference = abs(float(means[0])-record['human_log_loss'])
        tolerance = 8*(float(ses[0])+record['integration_log_loss_scramble_SE'])+2e-4
        assert difference <= tolerance, (i, difference, tolerance)
        assert abs(float(means[1])-record['human_Brier_loss']) < .005
        integral_checks.append(dict(label_menus=record['label_menus'], trial=record['trial'],
                                    orthogonal_slopes=record['orthogonal_slopes'], method=record['method'],
                                    larger_integral_log_loss=float(means[0]), larger_integral_scramble_SE=float(ses[0]),
                                    absolute_log_loss_difference=difference, diagnostic_tolerance=tolerance,
                                    larger_integral_Brier=float(means[1]),
                                    Brier_difference=float(means[1]-record['human_Brier_loss'])))
    pd.DataFrame(integral_checks).to_csv(args.out/'larger_integral_checks.csv', index=False)
    summary_error = 0.
    measures = ['squared_coefficient_error', 'coefficient_norm', 'human_log_loss',
                'excess_human_log_loss', 'human_Brier_loss', 'human_classification_error']
    for record in summary.itertuples():
        part = frame[(frame.label_menus==record.label_menus)&(frame.orthogonal_slopes==record.orthogonal_slopes)&frame.method.eq(record.method)]
        good = part[part.fit_status.eq('numerical_finite_root')]
        assert len(good) == record.numerical_finite
        assert part.fit_status.eq('numerical_unresolved').sum() == record.numerical_unresolved
        for measure in measures:
            if len(good):
                values = good[measure].to_numpy()
                for prefix, computed in [('mean_', np.mean(values)), ('median_', np.median(values)),
                                         ('q95_', np.quantile(values, .95)), ('maximum_', np.max(values))]:
                    saved = getattr(record, prefix+measure)
                    relative = abs(saved-computed)/(1+abs(computed))
                    summary_error = max(summary_error, relative)
                    assert relative < 1e-12
    result = dict(state='passed', all_original_support_states_reconciled=9000,
                  all_records_retained=len(rows), numerical_roots_checked=checks,
                  maximum_original_objective_gradient_l2=max_score,
                  maximum_original_objective_value_difference=max_objective_error,
                  maximum_coefficient_error_reconstruction_difference=max_parameter_error,
                  summary_cells_reconciled=len(summary), maximum_summary_relative_error=summary_error,
                  larger_independent_integrals=len(integral_checks),
                  larger_integral_nodes_per_fit=8*2**14*2,
                  larger_integral_policy='Deterministic stride through numerical fits plus the 20 largest norms. Scramble diagnostics are not deterministic error certificates.',
                  numerical_unresolved=int(frame.fit_status.eq('numerical_unresolved').sum()),
                  primary_source_sha256=checkpoint['source_sha256'],
                  verifier_source_sha256=sha(Path(__file__)), new_primary_rerun=False,
                  independent_expert_or_originality_review=False)
    save(args.out/'complete_verification.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--results', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        print(json.dumps(software_checks(a.out)), flush=True)
        if a.results:
            verify(a)
