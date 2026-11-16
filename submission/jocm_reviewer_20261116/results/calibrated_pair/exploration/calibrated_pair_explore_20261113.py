"""Exploration, after the null-confidence grid; not a prespecified experiment."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import expit, roots_hermitenorm
from imperfect_predictor_choice_boundary import ImperfectPopulation


def posterior(mu, error, z):
    q = error + (1 - 2 * error) * mu
    return np.where(z == 1, (1 - error) * mu / q,
                    error * mu / (1 - q))


def posterior_variance(mu, error):
    # Stable closed form for a binary symmetric signal.
    q = error + (1 - 2 * error) * mu
    return (1 - 2 * error)**2 * mu**2 * (1 - mu)**2 / (q * (1 - q))


class MatchedPopulation(ImperfectPopulation):
    def __init__(self, gamma, error, strong_error, tau, lam, kind, nodes=100):
        s, w = roots_hermitenorm(nodes)
        mu = expit(gamma * s)
        w = w / np.sqrt(2 * np.pi)
        qa = error + (1 - 2 * error) * mu
        qb = strong_error + (1 - 2 * strong_error) * mu
        # Cancel the shared mu^2(1-mu)^2 before evaluating extreme nodes.
        mix = ((1 - 2 * error) / (1 - 2 * strong_error))**2 * qb * (1 - qb) / (qa * (1 - qa))
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


if __name__ == '__main__':
    rows = []
    for gamma in (0., 1., 2.):
        for error in (.1, .05):
            for tau in (.1, 1., 4.):
                for label, power in [('unit', 1.), ('variance_optimal', tau / (1 + tau))]:
                    result = {'gamma': gamma, 'moderate_error': error, 'strong_error': .001,
                              'tau': tau, 'power_name': label, 'power': power}
                    for kind in ('moderate', 'selective'):
                        pop = MatchedPopulation(gamma, error, .001, tau, power, kind, 120)
                        fit = pop.critical_general()
                        result[kind] = fit['critical_ratio']
                        result[kind + '_gradient'] = fit['gradient_max']
                    result['gap'] = result['moderate'] - result['selective']
                    rows.append(result)
                    print(json.dumps(result), flush=True)
    pd.DataFrame(rows).to_csv(Path(__file__).with_name('calibrated_pair_exploration_20261113.csv'), index=False)
