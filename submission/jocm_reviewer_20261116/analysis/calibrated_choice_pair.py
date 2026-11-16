"""Conditional-moment matched calibrated predictors in the existing choice loss.

This is a same-model example; general conic geometry and PPI power tuning
are established prior work. See the retained exploratory selection history.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.special import expit, roots_hermitenorm
from imperfect_predictor_choice_boundary import ImperfectPopulation


def posterior(mu, error, z):
    q = error + (1 - 2 * error) * mu
    return np.where(z == 1, (1 - error) * mu / q, error * mu / (1 - q))


def variance(mu, error):
    q = error + (1 - 2 * error) * mu
    return (1 - 2 * error)**2 * mu**2 * (1 - mu)**2 / (q * (1 - q))


def selection_probability(mu, error, strong_error):
    qa = error + (1 - 2 * error) * mu
    qb = strong_error + (1 - 2 * strong_error) * mu
    # Algebraic cancellation avoids 0/0 at rounded extreme quadrature nodes.
    return ((1 - 2 * error) / (1 - 2 * strong_error))**2 * qb * (1 - qb) / (qa * (1 - qa))


class CalibratedPopulation(ImperfectPopulation):
    def __init__(self, gamma, error, strong_error, tau, lam, kind, nodes=120):
        assert 0 < strong_error <= error < .5 and 0 < lam <= 1
        s, w = roots_hermitenorm(nodes)
        mu = expit(gamma * s)
        w = w / np.sqrt(2 * np.pi)
        mix = selection_probability(mu, error, strong_error)
        assert min(mix) >= 0 and max(mix) <= 1 + 1e-12
        rows = []
        for y in (0, 1):
            wy = w * (mu if y else 1 - mu)
            for z in (0, 1):
                e = error if kind == 'moderate' else strong_error
                wz = (1 - e) if y == z else e
                weights = wy * wz * (1 if kind == 'moderate' else mix)
                rows.append((s, np.full(nodes, y), posterior(mu, e, z), weights))
            if kind == 'selective':
                rows.append((s, np.full(nodes, y), mu, wy * (1 - mix)))
        self.s, self.y, self.g, self.weights = [np.concatenate([row[j] for row in rows]) for j in range(4)]
        self.signL = 1 - 2 * self.y
        self.signU = np.ones(len(self.s))
        r = np.where(self.y == 0, self.g, 1 - self.g)
        self.aL, self.bL = lam * r, 1 - lam + lam * r
        self.aU, self.bU = -lam / tau * self.g, lam / tau * (1 - self.g)
        self.tau = tau

    def moment_matrices(self, gamma):
        mu = expit(gamma * self.s)
        a, b = mu - self.y, mu - self.g
        x = np.column_stack([np.ones(len(mu)), self.s])
        matrix = lambda v: x.T @ ((self.weights * v)[:, None] * x)
        # Orthogonal independent Gaussian slopes have the scalar expectation;
        # all their cross terms with the fitted intercept/signal vanish.
        return {'truth': matrix(a*a), 'prediction': matrix(b*b), 'cross': matrix(a*b),
                'information': matrix(mu*(1-mu)),
                'orthogonal_truth': float(self.weights @ (a*a)),
                'orthogonal_prediction': float(self.weights @ (b*b)),
                'orthogonal_cross': float(self.weights @ (a*b)),
                'Brier': float(self.weights @ ((self.y-self.g)**2)),
                'accuracy': float(self.weights @ np.where(self.g > .5, self.y, np.where(self.g < .5, 1-self.y, .5))),
                'log_loss': float(self.weights @ (-self.y*np.log(np.maximum(self.g, 1e-300))-(1-self.y)*np.log(np.maximum(1-self.g, 1e-300))))}


def draw_pair(rng, n, gamma=1., error=.05, strong_error=.001):
    s = rng.normal(size=n)
    mu = expit(gamma*s)
    y = (rng.random(n) < mu).astype(float)
    za = np.logical_xor(y, rng.random(n) < error).astype(int)
    zb = np.logical_xor(y, rng.random(n) < strong_error).astype(int)
    flag = rng.random(n) < selection_probability(mu, error, strong_error)
    return s, y, posterior(mu, error, za), np.where(flag, posterior(mu, strong_error, zb), mu)


def interval_event(x, y, g, m, power):
    tau = (len(y)-m)/m
    signs = np.r_[1-2*y[:m], np.ones(len(y)-m)]
    r = np.where(y[:m] == 0, g[:m], 1-g[:m])
    lo = np.r_[power*r, -power*g[m:]/tau]
    hi = np.r_[1-power+power*r, power*(1-g[m:])/tau]
    oriented = (x*signs[:, None]).T
    fit = linprog(np.zeros(len(y)), A_eq=oriented, b_eq=np.zeros(x.shape[1]),
                  bounds=list(zip(lo, hi)), method='highs')
    if fit.status == 2:
        return 'outside', None
    if not fit.success or np.max(np.abs(oriented @ fit.x)) > 1e-7:
        return 'unresolved', None
    # Under the continuous full-rank law the boundary event has probability
    # zero. A small-case original LP independently distinguishes interior.
    return 'feasible', float(np.max(np.abs(oriented @ fit.x)))


def population(out):
    rows, moments = [], []
    for gamma in (0., 1., 2.):
        for error in (.1, .05):
            for tau in (.1, 1., 4.):
                for label, power in [('unit', 1.), ('variance_optimal', tau/(1+tau))]:
                    for kind in ('moderate', 'selective'):
                        for nodes in (120, 240, 400):
                            pop = CalibratedPopulation(gamma, error, .001, tau, power, kind, nodes)
                            rows.append({'gamma': gamma, 'moderate_error': error, 'strong_error': .001,
                                         'tau': tau, 'power_name': label, 'power': power, 'kind': kind,
                                         'nodes': nodes, **pop.critical_general()})
                            if nodes == 400 and tau == 4 and label == 'variance_optimal':
                                values = pop.moment_matrices(gamma)
                                moments.append({'gamma': gamma, 'moderate_error': error, 'kind': kind,
                                                **{k: v.tolist() if isinstance(v, np.ndarray) else v for k,v in values.items()}})
    pd.DataFrame(rows).to_csv(out/'population.csv', index=False)
    (out/'moments.json').write_text(json.dumps(moments, indent=2)+'\n')
    print('Population calculations complete: 216 records.', flush=True)


def phase(out):
    rng = np.random.default_rng(20261113)
    rows = []
    for m in (40, 80):
        for rep in range(200):
            s, y, ga, gb = draw_pair(rng, 5*m)
            orth = rng.normal(size=(5*m, int(round(1.35*m))))
            for ratio in (1.05, 1.2, 1.35):
                q = int(round(ratio*m))
                x = np.column_stack([np.ones(5*m), s, orth[:, :q]])
                for kind, g in [('moderate', ga), ('selective', gb)]:
                    status, residual = interval_event(x, y, g, m, .8)
                    rows.append({'labels': m, 'auxiliary_rows': 4*m, 'replication': rep,
                                 'orthogonal_dimensions': q, 'total_dimensions': q+2,
                                 'dimension_label_ratio': q/m, 'kind': kind,
                                 'power': .8, 'status': status, 'equality_residual': residual})
        pd.DataFrame(rows).to_csv(out/'phase_records.csv', index=False)
        print(f'Finite-size calculations complete for m={m}.', flush=True)
    frame = pd.DataFrame(rows)
    frame.groupby(['labels', 'dimension_label_ratio', 'kind', 'status'], as_index=False).size().to_csv(out/'phase_counts.csv', index=False)
    paired = []
    for (m, ratio), group in frame.groupby(['labels', 'dimension_label_ratio']):
        pivot = group.pivot(index='replication', columns='kind', values='status')
        difference = (pivot.selective == 'feasible').astype(float) - (pivot.moderate == 'feasible').astype(float)
        paired.append({'labels': m, 'dimension_label_ratio': ratio, 'replications': len(pivot),
                       'selective_minus_moderate_frequency': difference.mean(),
                       'paired_Monte_Carlo_SE': difference.std(ddof=1)/np.sqrt(len(pivot)),
                       'unresolved': int(pivot.eq('unresolved').sum().sum())})
    pd.DataFrame(paired).to_csv(out/'paired_phase.csv', index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--population-only', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    population(args.out)
    if not args.population_only:
        phase(args.out)
    (args.out/'run.json').write_text(json.dumps({'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'population_records': 216, 'simulation_performed': not args.population_only,
        'simulation_records_expected': 2400 if not args.population_only else 0,
        'independent_priority_verified': False}, indent=2)+'\n')


if __name__ == '__main__':
    main()
