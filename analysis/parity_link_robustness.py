"""Link-robustness check for the reflected utility-fibre test.

The benchmark uses the moderate-utility form P(A>B)=F(delta/D), with a
candidate-preserving gap delta and a reflection-even distance D. It checks
that parity separates an odd omitted utility direction from an even distance
(scale) nuisance under logistic and probit links.
"""
from __future__ import annotations

import argparse
import csv
import math
import os

import numpy as np


def sigmoid(x):
    x = np.clip(x, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x))


def probit(x):
    x = np.clip(x, -9.0, 9.0)
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def arm_grid():
    arms = []
    for u in (-45.0, -15.0, 15.0, 45.0, 75.0):
        c, t = 2.5, 1.5
        sa, sb = 90.0 + u / 2.0, 90.0 - u / 2.0
        zpa, zpb = c + t, c - t
        zma, zmb = c - t, c + t
        # Pair distance is reflection-even; the omitted z^2 difference is odd.
        d_plus = (sa - sb + zpa - zpb) ** 2 + (sa - sb - zpa + zpb) ** 2
        d_minus = (sa - sb + zma - zmb) ** 2 + (sa - sb - zma + zmb) ** 2
        h_plus, h_minus = zpa**2 - zpb**2, zma**2 - zmb**2
        arms.append((u, h_plus, h_minus, float(d_plus), float(d_minus)))
    return arms


def randomization_pvalue(scores, reps, rng):
    observed = abs(float(np.mean(scores)))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(reps, len(scores)))
    draws = np.abs((signs @ scores) / len(scores))
    return float((1.0 + np.sum(draws >= observed)) / (reps + 1.0))


def evaluate(arms, condition, link, respondents, permutation_reps, seed):
    rng = np.random.default_rng(seed)
    eta = 0.10 if condition in ("omitted", "combined") else 0.0
    scale = 0.005 if condition in ("even_scale", "combined") else 0.0
    link_fn = sigmoid if link == "logit" else probit
    scores = np.zeros(respondents)
    for rid in range(respondents):
        sign = 1.0 if rng.integers(0, 2) else -1.0
        y = []
        for u, hp, hm, dp, dm in arms:
            if sign > 0:
                h, d = hp, dp
            else:
                h, d = hm, dm
            delta = 0.030 * u + eta * h
            denom = 1.0 + scale * d
            y.append(rng.binomial(1, float(link_fn(np.asarray([delta / denom]))[0])))
        scores[rid] = sign * np.mean(y)
    pvalue = randomization_pvalue(
        scores, permutation_reps, np.random.default_rng(seed + 1001))
    return int(pvalue < 0.05), pvalue


def run(reps=100, respondents=600, permutation_reps=199,
        out="results/parity_link_robustness.csv"):
    arms = arm_grid()
    assert all(abs(a[3] - a[4]) < 1e-12 for a in arms)
    assert all(abs(a[1] + a[2]) < 1e-12 for a in arms)
    rows = []
    for link in ("logit", "probit"):
        for condition in ("null", "omitted", "even_scale", "combined"):
            for rep in range(reps):
                reject, pvalue = evaluate(
                    arms, condition, link, respondents, permutation_reps,
                    93000000 + rep)
                rows.append({"link": link, "condition": condition,
                             "rep": rep, "reject": reject,
                             "pvalue": pvalue})
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as handle:
        fields = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    for link in ("logit", "probit"):
        for condition in ("null", "omitted", "even_scale", "combined"):
            subset = [r for r in rows if r["link"] == link and
                      r["condition"] == condition]
            print(link, condition, "rejection_rate",
                  round(float(np.mean([r["reject"] for r in subset])), 3))
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--respondents", type=int, default=600)
    parser.add_argument("--permutation-reps", type=int, default=199)
    parser.add_argument("--out", default="results/parity_link_robustness.csv")
    run(**vars(parser.parse_args()))
