"""Recover limiting menu faces and refit the original objective at high precision."""
from argparse import ArgumentParser
from decimal import Decimal as D, localcontext
from fractions import Fraction
import json
from pathlib import Path

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import linprog


ROOT = Path(__file__).resolve().parent
TOL = 2e-8


def solve_lp(c, *, ub=None, rhs=None, eq=None, target=None, bounds=None):
    result = linprog(c, A_ub=ub, b_ub=rhs, A_eq=eq, b_eq=target,
                     bounds=bounds or [(None, None)] * len(c), method='highs-ds')
    assert result.success, result.message
    return result


def face_at_endpoint(x, weights, moment):
    """An alternative is in the minimal face iff some feasible mass uses it."""
    n, k, p = x.shape
    rows = np.kron(np.eye(n), np.ones((1, k)))
    eq = np.vstack((rows, x.reshape(-1, p).T))
    target = np.r_[weights, moment]
    tied = np.array([-solve_lp(-axis, eq=eq, target=target,
                               bounds=[(0, None)] * (n*k)).fun > TOL for axis in np.eye(n*k)]).reshape(n, k)
    assert np.all(tied.sum(axis=1)>0)
    anchors = x[np.arange(n), tied.argmax(axis=1)]
    contrast = anchors[:, None] - x
    normal = null_space(contrast[tied])
    return tied, anchors, contrast, normal


def rate_lp(a, s):
    dimension = len(s)
    lower = np.ones(len(a))
    best = solve_lp(s, ub=-a, rhs=-lower).fun

    def span(index):
        arguments = dict(ub=-a, rhs=-lower, eq=[s], target=[best])
        return -solve_lp(-a[index], **arguments).fun - solve_lp(a[index], **arguments).fun

    varying = [i for i in range(len(a)) if span(i) > TOL]
    levels = []
    objective = np.r_[np.zeros(dimension), -1.]
    while varying:
        mask = np.isin(np.arange(len(a)), varying).astype(float)
        eq = [np.r_[s, 0.]]
        ub = np.vstack((np.column_stack((-a, mask)), np.column_stack((-a, np.zeros(len(a))))))
        rhs = np.r_[-lower * (1-mask), -lower]
        level = -solve_lp(objective, ub=ub, rhs=rhs, eq=eq, target=[best]).fun
        lower[varying] = np.maximum(level, lower[varying])
        updated = [i for i in varying if span(i) > TOL]
        assert len(updated) < len(varying)
        levels.append(level)
        varying = updated
    direction = solve_lp(np.zeros(dimension), ub=-a, rhs=-lower, eq=[s], target=[best]).x
    return direction, levels, best


def design(gaps, changed):
    k, q = len(gaps), len(gaps[0])
    top = np.r_[np.zeros(q, dtype=int), 1]
    x = np.asarray([[top, -top, -np.r_[g, 0]] for g in gaps])
    x = np.concatenate((x, x.copy()))
    transform = np.eye(q+1, dtype=int)
    if changed:
        transform += np.triu(np.ones((q+1, q+1), dtype=int), 1)
        x = x @ transform
        for i in range(2*k):
            x[i] += np.array([(-1)**j*(j+i+1) for j in range(q+1)])
    return x, transform


def dec(s):
    a = Fraction(s)
    return D(a.numerator) / D(a.denominator)


def dot(x, y):
    return sum(a*b for a, b in zip(x, y, strict=True))


def linear_solve(matrix, target):
    equations = [list(row)+[rhs] for row, rhs in zip(matrix, target, strict=True)]
    n = len(target)
    for col in range(n):
        pivot = max(range(col, n), key=lambda k: abs(equations[k][col]))
        equations[col], equations[pivot] = equations[pivot], equations[col]
        scale = equations[col][col]
        assert scale != 0
        equations[col] = [a/scale for a in equations[col]]
        for i in range(n):
            if i != col:
                factor = equations[i][col]
                equations[i] = [a-factor*b for a, b in zip(equations[i], equations[col], strict=True)]
    return [r[-1] for r in equations]


def probability(x, beta):
    utilities = [dot(row, beta) for row in x]
    maximum = max(utilities)
    tails = [(u-maximum).exp() for u in utilities]
    total = sum(tails)
    return maximum+total.ln(), [w/total for w in tails]


def objective_data(x, lam, plan):
    k = len(x)//2
    y = list(map(dec, plan['tied_human_shares']))+[dec(plan['outside_choice_shares']['human'])]
    gl = dec(plan['outside_choice_shares']['labelled_predictor'])
    gu = dec(plan['outside_choice_shares']['auxiliary_predictor'])
    g0, g1 = [(1-gl)/2, (1-gl)/2, gl], [(1-gu)/2, (1-gu)/2, gu]
    w = [(1-lam)/k]*k+[lam/k]*k
    mass = [[(a-lam*b)/k for a,b in zip(y,g0,strict=True)]]*k+[[lam*a/k for a in g1]]*k
    target = [sum(row[j]*share for menu, vector in zip(x, mass, strict=True)
                  for row, share in zip(menu, vector, strict=True)) for j in range(len(x[0][0]))]
    return w, target, (y,g0,g1)


def evaluate(x, b, w, beta, full=True):
    p = len(beta)
    value = -dot(b, beta)
    score = [-a for a in b]
    hess = [[D(0)]*p for _ in beta]
    for menu, weight in zip(x,w,strict=True):
        norm, probabilities = probability(menu,beta)
        value += weight*norm
        if full:
            mu = [sum(a*row[j] for row,a in zip(menu, probabilities, strict=True)) for j in range(p)]
            for j in range(p):
                score[j] += weight*mu[j]
                for k in range(p):
                    hess[j][k] += weight*sum(a*(row[j]-mu[j])*(row[k]-mu[k])
                                              for row,a in zip(menu, probabilities, strict=True))
    return value, score, hess


def refit(x, target, weights, beta):
    for iteration in range(150):
        value, score, hess = evaluate(x,target,weights,beta)
        step = linear_solve(hess, score)
        if max(map(abs, step)) < D('1e-29'):
            return beta, iteration
        length = D(1)
        while length > D('1e-20'):
            trial = [b-length*d for b,d in zip(beta,step,strict=True)]
            if evaluate(x,target,weights,trial,False)[0] < value:
                beta = trial
                break
            length /= 2
        else:
            raise ArithmeticError('No objective decrease at the current precision')
    raise ArithmeticError('Iteration limit')


def direct_score(x, beta, source, lam):
    """The original three loss gradients, computed without the normalized target."""
    k = len(x)//2
    labelled, aux = x[:k],x[k:]
    y, gl, gu = source

    def sample_score(menus, response):
        result = [D(0)]*len(beta)
        for menu in menus:
            probs = probability(menu,beta)[1]
            for j in range(len(beta)):
                result[j] += sum((p-r)*xj[j] for p,r,xj in zip(probs,response,menu,strict=True))/k
        return result

    human = sample_score(labelled,y)
    extra = sample_score(aux,gu)
    adjust = sample_score(labelled,gl)
    return max(abs(a+lam*(b-c)) for a,b,c in zip(human,extra,adjust,strict=True))


def fit_paths(plan, case, changed):
    features, transformation = design(case['gaps'],changed)
    x = [[[D(int(z)) for z in row] for row in menu] for menu in features]
    c = dec(plan['endpoint'])
    w0,b0,_ = objective_data(x,c,plan)
    w1,b1,_ = objective_data(x,c-1,plan)
    tied, anchors, contrasts, normal = face_at_endpoint(features,np.array(w0,float),np.array(b0,float))
    assert tied.tolist()==[[True,True,False]]*len(x)
    a = contrasts[~tied] @ normal
    weight_change = np.array(w1,float)-np.array(w0,float)
    moment_change = np.array(b1,float)-np.array(b0,float)
    s = (weight_change@anchors-moment_change) @ normal
    rate, levels, _ = rate_lp(a,s)
    intrinsic = [D(z) for z in map(dec,case['rate'])]+[D(0)]
    expected = np.linalg.solve(transformation,np.array(intrinsic,float))
    assert np.max(abs(normal@rate-expected)) < TOL
    assert np.allclose(levels,[float(dec(z)) for z in case['levels']],atol=TOL,rtol=0)
    q = len(case['rate'])
    beta = [D(0)]*(q+1)
    latest = {}
    for exponent in sorted(set(plan['fit_decimals']) | set(range(1, max(plan['fit_decimals'])+1))):
        delta = D(10)**(-exponent)
        lam = c-delta
        weights, target, original = objective_data(x,lam,plan)
        beta, iterations = refit(x,target,weights,beta)
        error = direct_score(x,beta,original,lam)
        assert error < delta*D('1e-26')
        if exponent in plan['fit_decimals']:
            behavioral = [dot([D(int(z)) for z in row], beta) for row in transformation]
            t = -delta.ln()
            tangent_error = abs(behavioral[-1]-D(9).ln()/2+D(7).ln()/2)
            deficits = [float(a/t) for a in behavioral[:-1]]
            # Test menus compare zero utility with each axis, observing the zero-utility choice.
            loss = [float((D(1)+a.exp()).ln()/t) for a in behavioral]
            correction = [float(a-t*d) for a,d in zip(behavioral,intrinsic,strict=True)]
            latest[exponent] = dict(iterations=iterations,score_error=float(error),scaled_coefficients=deficits,
                                   scaled_held_out_losses=loss, bounded_remainder=correction,
                                   tangent_error=float(tangent_error))
    nearest = latest[max(latest)]
    assert max(abs(a-float(b)) for a,b in zip(nearest['scaled_held_out_losses'],intrinsic,strict=True)) < .08
    assert nearest['tangent_error'] < 1e-26
    assert max(map(abs,nearest['bounded_remainder'])) < 5
    return dict(case=case['name'], transformed=changed, normal_dimension=normal.shape[1],
                rate=expected.tolist(), gap_levels=levels, path=latest)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path)
    args = parser.parse_args()
    plan = json.loads((ROOT/'plan.json').read_text())
    results = []
    with localcontext() as context:
        context.prec=plan['precision']
        for case in plan['cases']:
            rows = [fit_paths(plan,case,changed) for changed in (False,True)]
            for exponent in plan['fit_decimals']:
                l,r = [row['path'][exponent]['scaled_held_out_losses'] for row in rows]
                assert max(abs(a-b) for a,b in zip(l,r,strict=True)) < 1e-22
            results.extend(rows)
    args.out.write_text(json.dumps(dict(status='passed',comparisons=len(results),
                                         recorded_fits=len(results)*len(plan['fit_decimals']), cases=results),indent=2)+'\n')
    print(f'{len(results)} designs and {len(results)*len(plan["fit_decimals"])} recorded fits passed')


if __name__=='__main__':
    main()
