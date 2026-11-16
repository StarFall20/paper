"""Independent conditional moments, adaptive integrals and original support LPs.

Population verification uses scipy.quad_vec and symmetry-reduced nested scalar
roots, without CalibratedPopulation or the production Hermite saddle solver.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.integrate import quad_vec
from scipy.optimize import brentq
from scipy.special import expit, ndtr
from calibrated_choice_pair import draw_pair, interval_event
from power_tuned_choice_audit import objective, support


def conditional(mu, e, h, kind):
    """Enumerate joint (Y,g) directly, including the uninformative component."""
    qa, qb = e+(1-2*e)*mu, h+(1-2*h)*mu
    mix = ((1-2*e)/(1-2*h))**2 * qb*(1-qb)/(qa*(1-qa))
    error = e if kind == 'moderate' else h
    q = error+(1-2*error)*mu
    values = [(1-error)*mu/q, error*mu/(1-q)]
    yy, gg, ww = [], [], []
    for y in (0., 1.):
        py = mu if y else 1-mu
        for z, g in zip((1, 0), values):
            yy.append(y); gg.append(g)
            ww.append(py*((1-error) if y == z else error)*(1 if kind == 'moderate' else mix))
        if kind == 'selective':
            yy.append(y); gg.append(mu); ww.append(py*(1-mix))
    return np.array(yy), np.array(gg), np.array(ww)


def conditional_checks():
    error = 0.
    for mu in np.r_[np.linspace(.0001, .9999, 251), 0., 1.]:
        for e in (.1, .05):
            results = []
            for kind in ('moderate', 'selective'):
                y, g, w = conditional(mu, e, .001, kind)
                for value in np.unique(g):
                    keep = g == value
                    error = max(error, abs(float(w[keep] @ (y[keep]-g[keep]))))
                a, b = mu-y, mu-g
                results.append(np.array([w.sum(), w@g, w@(b*b), w@(a*b), w@((y-g)**2)]))
            error = max(error, max(abs(results[0]-results[1])))
    assert error < 1e-12, error
    return error


def adaptive_boundary(gamma, e, tau, power, kind):
    phi = lambda z: np.exp(-z*z/2)/np.sqrt(2*np.pi)
    f = lambda z: (1+z*z)*ndtr(-z)-z*phi(z)
    fp = lambda z: 2*(z*ndtr(-z)-phi(z))
    max_error = 0.

    def integrated(eta, t):
        nonlocal max_error
        def integrand(s):
            mu = expit(gamma*s)
            y, g, w = conditional(mu, e, .001, kind)
            sign = 1-2*y
            total = np.zeros(3)
            for orientation, a, b, weight in [
                (sign, power*np.where(y == 0, g, 1-g), 1-power+power*np.where(y == 0, g, 1-g), w),
                (np.ones(len(y)), -power*g/tau, power*(1-g)/tau, tau*w)]:
                shift = orientation*eta*s
                left, right = -shift-t*a, shift+t*b
                value = 1+shift**2-f(left)-f(right)
                deta = (2*shift+fp(left)-fp(right))*orientation*s
                dt = a*fp(left)-b*fp(right)
                total += np.array([weight@value, weight@deta, weight@dt])
            return phi(s)*total
        value, err = quad_vec(integrand, -12., 12., epsabs=2e-10, epsrel=2e-10, limit=300)
        max_error = max(max_error, float(err))
        return value

    def maximize(eta):
        upper = 1.
        while integrated(eta, upper)[2] >= 0:
            upper *= 2
            assert upper < 1e6
        t = brentq(lambda z: integrated(eta, z)[2], 0., upper, xtol=2e-10)
        value = integrated(eta, t)
        return value, t

    if gamma == 0:
        eta = 0.
    else:
        # Gaussian reflection puts the fitted-intercept multiplier at zero.
        lo, hi = -1., 1.
        while maximize(lo)[0][1] >= 0: lo *= 2
        while maximize(hi)[0][1] <= 0: hi *= 2
        eta = brentq(lambda z: maximize(z)[0][1], lo, hi, xtol=2e-10)
    val, t = maximize(eta)
    return {'critical_ratio': float(val[0]), 'signal_multiplier': float(eta),
            'scale': float(t), 'stationarity': float(max(abs(val[1:]))),
            'quad_error_estimate_max': max_error, 'truncation': 'S in [-12,12]; Gaussian tail controlled by a quadratic envelope, not an interval certificate'}


def verify(inputs, out):
    frame = pd.read_csv(inputs/'population.csv')
    assert len(frame) == 216
    piv = frame.pivot(index=['gamma','moderate_error','tau','power_name','kind'], columns='nodes', values='critical_ratio')
    quadrature = float(abs(piv[400]-piv[240]).max())
    assert quadrature < 2e-7, quadrature
    adaptive = []
    for gamma, e, tau, power, label in [(1., .05, 4., .8, 'variance_optimal'),
                                      (0., .1, .1, 1., 'unit'),
                                      (0., .1, 4., .8, 'variance_optimal')]:
        for kind in ('moderate', 'selective'):
            result = adaptive_boundary(gamma, e, tau, power, kind)
            selected = frame[(frame.gamma == gamma)&(frame.moderate_error == e)&(frame.tau == tau)&
                             (frame.power_name == label)&(frame.kind == kind)&(frame.nodes == 400)].iloc[0]
            result.update(gamma=gamma, moderate_error=e, tau=tau, power=power, kind=kind,
                          hermite_difference=abs(result['critical_ratio']-selected.critical_ratio))
            assert result['hermite_difference'] < 1e-7 and result['stationarity'] < 1e-7, result
            adaptive.append(result)
            print(json.dumps(result), flush=True)

    # Full truth/prediction/cross information blocks are compared, not just
    # the trace or a selected variance-optimal scalar.
    matrices = json.loads((inputs/'moments.json').read_text())
    matrix_error = 0.
    for gamma in (0., 1., 2.):
        for e in (.1, .05):
            pair = [r for r in matrices if r['gamma'] == gamma and r['moderate_error'] == e]
            assert len(pair) == 2
            for key in ('truth','prediction','cross','information','orthogonal_truth','orthogonal_prediction','orthogonal_cross','Brier'):
                matrix_error = max(matrix_error, float(np.max(abs(np.asarray(pair[0][key])-np.asarray(pair[1][key])))))
            for row in pair:
                matrix_error = max(matrix_error, float(np.max(abs(np.array(row['prediction'])-np.array(row['cross'])))))
    assert matrix_error < 1e-12, matrix_error

    rng = np.random.default_rng(20261114)
    counts = {}; residuals = []
    for j in range(60):
        m = 16+j%5
        s,y,ga,gb = draw_pair(rng, 5*m, gamma=(j%3), error=(.1 if j%2 else .05))
        x = np.column_stack([np.ones(5*m),s,rng.normal(size=(5*m,2+j%10))])
        power = [.2,.8,1.][j%3]
        g = ga if j%2 else gb
        state, residual = interval_event(x,y,g,m,power)
        original = support(objective(x,y,g,m,power))['status']
        expected = 'feasible' if original == 'interior' else 'outside' if original == 'outside_mean_support' else 'unresolved'
        assert state == expected and state != 'unresolved', (j,state,original)
        counts[original] = counts.get(original,0)+1
        if residual is not None: residuals.append(residual)

    records = pd.read_csv(inputs/'phase_records.csv')
    assert len(records) == 2400
    assert not records.status.eq('unresolved').any()
    groups = records.groupby(['labels','dimension_label_ratio','kind'])
    assert groups.size().eq(200).all()
    pd.DataFrame(adaptive).to_csv(out/'adaptive.csv', index=False)
    summary = {'conditional_joint_moment_max_error': conditional_checks(),
               'complete_covariance_block_max_error': matrix_error,
               '240_400_node_boundary_max_difference': quadrature,
               'adaptive_integral_max_difference': max(r['hermite_difference'] for r in adaptive),
               'original_LP_crosschecks': 60, 'original_support_statuses': counts,
               'LP_max_equality_residual': max(residuals), 'phase_records': len(records),
               'unresolved_phase_records': int(records.status.eq('unresolved').sum()),
               'independent_priority_verified': False, 'rigorous_interval_bounds': False}
    (out/'checks.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    verify(a.input,a.out)
