"""Fit the original corrected menus and verify the unit-endpoint programs."""
import argparse
from decimal import Decimal as D, localcontext
from fractions import Fraction as F
import json
from pathlib import Path

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import linprog

HALF = F(1, 2)


def fraction_vector(vector):
    return tuple(F(x) for x in vector)


def binary(vector, y=1, g=HALF):
    return (((F(0),) * len(vector), fraction_vector(vector)), y, (1-g, g))


def auxiliary(menu, tied):
    q = tuple(F(1, len(menu)) for _ in menu)
    return (tuple(map(fraction_vector, menu)), 0, q), tuple(tied)


def model(name, human, aux_pairs, direction, offset, focus_y=0):
    return dict(name=name, human=tuple(human), aux=tuple(a for a, _ in aux_pairs),
                ties=tuple(t for _, t in aux_pairs), direction=fraction_vector(direction),
                offset=offset, focus_y=focus_y)


def examples():
    zero = (0, 0)
    paired = [auxiliary((zero, (1, 0)), (1,)), auxiliary((zero, (2, -1)), (1,))]
    basic = (binary((-2, 1)), binary((-1, -1)))
    return (
        model('rank_one_boundary', (*basic, binary((6, 0))), paired[:1], (1, '1/2'), 'rank_one'),
        model('full_rank_binary', (*basic, binary(('15/2', '-3/2'))), paired, (1, 0), 'binary'),
        model('three_alternative_menu', (*basic, binary((9, -2))),
              (auxiliary((zero, (-1, 0), (-2, 1)), (0,)),), (1, 0), 'grouped'),
        model('nonzero_face_dimension',
              (binary((-2, 1, 0)), binary((-1, -1, 0)), binary(('15/2', '-3/2', '3/2'))),
              (auxiliary(((0,0,0), (0,0,1), (-1,0,0), (-2,1,0)), (0,1)),), (1,0,0), 'tangent'),
    )


def ordinary_and_unit_moments(human, aux):
    def predicted(rows):
        return tuple(sum(sum(p*x[k] for p,x in zip(q,menu)) for menu,_,q in rows)/len(rows)
                     for k in range(len(rows[0][0][0])))
    a = tuple(sum(m[y][k] for m,y,_ in human)/len(human) for k in range(len(human[0][0][0])))
    pg_u,pg_h = predicted(aux),predicted(human)
    return a, tuple(h+u-l for h,u,l in zip(a,pg_u,pg_h))


def linear_program(cost, lhs, rhs, bounds, eq=None, eq_rhs=None):
    result = linprog(cost,A_ub=np.asarray(lhs) if len(lhs) else None,
                     b_ub=np.asarray(rhs) if len(rhs) else None,
                     A_eq=eq,b_eq=eq_rhs,bounds=bounds,method='highs')
    assert result.success, result.message
    return result


def direction_path(case):
    human, aux, tied = (case[k] for k in ('human','aux','ties'))
    p = len(case['direction'])
    contrasts = []
    auxiliary_gaps = []
    for (menu, _, _), indices in zip(aux,tied):
        m = np.array(menu,dtype=float)
        anchor = m[indices[0]]
        contrasts.extend(m[j]-anchor for j in indices)
        auxiliary_gaps.extend(anchor-x for j,x in enumerate(m) if j not in indices)
    basis = null_space(np.array(contrasts).reshape(-1,p))
    d = basis.shape[1]
    A = np.array(auxiliary_gaps) @ basis
    X = [np.array(menu,dtype=float) @ basis for menu,_,_ in human]
    m = len(human)
    bounds = [(None,None)]*d + [(0,None)]*m
    first = [np.r_[-r,np.zeros(m)] for r in A]
    rhs = [-1.0]*len(A)
    for i,(x,(_,y,_)) in enumerate(zip(X,human)):
        for diff in x-x[y]:
            aux_var = np.zeros(m)
            aux_var[i]=-1
            first.append(np.r_[diff,aux_var])
            rhs.append(0.0)
    objective = np.r_[np.zeros(d),np.full(m,1/m)]
    res = linear_program(objective,first,rhs,bounds)
    first.append(objective)
    rhs.append(res.fun)
    h_optimum = res.fun
    gaps = [(np.r_[v,np.zeros(m)],-1.0,'auxiliary') for v in A]
    for x in X:
        chosen = next(i for i in range(len(x)) if all(
            linear_program(np.r_[diff,np.zeros(m)],first,rhs,bounds).fun >= -1e-8
            for diff in x[i]-x))
        gaps.extend((np.r_[diff,np.zeros(m)],0.0,'human') for diff in x[chosen]-x)
    stages = []
    for _ in range(d+1):
        remaining = [(v,c,g) for v,c,g in gaps if
                     -linear_program(-v,first,rhs,bounds).fun-linear_program(v,first,rhs,bounds).fun>1e-8]
        if not remaining:
            direction = basis @ linear_program(np.zeros(d+m),first,rhs,bounds).x[:d]
            assert np.max(np.abs(direction-np.array(case['direction'],dtype=float)))<1e-8
            return dict(direction=direction.tolist(),ordinary_recession=h_optimum,stages=stages)
        stage = linear_program(np.r_[np.zeros(d+m),-1],
                               [np.r_[v,0] for v in first]+[np.r_[-v,1] for v,_,_ in remaining],
                               rhs+[c for _,c,_ in remaining],bounds+[(None,None)])
        value = stage.x[-1]
        first.extend(-v for v,_,_ in remaining)
        rhs.extend(c-value for _,c,_ in remaining)
        stages.append(dict(gap=float(value),families=sorted(set(g for _,_,g in remaining))))
    raise AssertionError('Nonterminal lexicographic path')


def decimal(v):
    v = F(v)
    return D(v.numerator) / D(v.denominator)


def dot(a, b):
    return sum(x*y for x,y in zip(a,b))


def softmax(menu, beta):
    scores = [dot(x,beta) for x in menu]
    high = max(scores)
    exp = [(s-high).exp() for s in scores]
    total = sum(exp)
    return high+total.ln(), [v/total for v in exp]


def score_terms(rows, targets, weights, beta):
    p = len(beta)
    loss = -dot(targets,beta)
    grad = [-t for t in targets]
    hess = [[D(0)]*p for _ in range(p)]
    for menu,w in zip(rows,weights):
        z, probs = softmax(menu,beta)
        loss += w*z
        means = [sum(a*x[j] for x,a in zip(menu,probs)) for j in range(p)]
        for k in range(p):
            grad[k] += w*means[k]
            for l in range(p):
                hess[k][l] += w*sum((x[k]-means[k])*(x[l]-means[l])*a for x,a in zip(menu,probs))
    return loss,grad,hess


def linear_solve(matrix, response):
    rows = [list(r)+[y] for r,y in zip(matrix,response)]
    for j in range(len(rows)):
        pivot = max(range(j,len(rows)),key=lambda i:abs(rows[i][j]))
        rows[j],rows[pivot] = rows[pivot],rows[j]
        scale = rows[j][j]
        assert abs(scale)>0
        rows[j] = [x/scale for x in rows[j]]
        for i in range(len(rows)):
            if i!=j:
                v=rows[i][j]
                rows[i]=[x-v*y for x,y in zip(rows[i],rows[j])]
    return [r[-1] for r in rows]


def fit(terms, beta):
    for _ in range(80):
        value,g,h = terms(beta)
        step = linear_solve(h,g)
        if max(map(abs,step))<D('1e-60'):
            return beta
        for power in range(32):
            eta = D(2)**-power
            trial = [u-eta*v for u,v in zip(beta,step)]
            if terms(trial)[0]<=value+D('1e-79'):
                beta=trial
                break
        else:
            raise AssertionError('Newton line search failed')
    raise AssertionError('Newton did not converge')


def independent_original_score(human, aux, s, beta):
    def average(rows, use_prediction):
        result = [D(0)]*len(beta)
        for menu,y,q in rows:
            x = tuple(tuple(map(decimal,row)) for row in menu)
            z = sum(dot(v,beta).exp() for v in x)
            for j,v in enumerate(x):
                prob = dot(v,beta).exp()/z
                target = decimal(q[j]) if use_prediction else D(j==y)
                for k,u in enumerate(v):
                    result[k] += (prob-target)*u/len(rows)
        return result
    a,b,c = average(human,False),average(aux,True),average(human,True)
    return [u+(1-s)*(v-w) for u,v,w in zip(a,b,c)]


def numeric_path(case, distances):
    human,aux,ties = (case[k] for k in ('human','aux','ties'))
    b0,b1 = ordinary_and_unit_moments(human,aux)
    tied_mean = tuple(sum(sum(rows[j][k] for j in idx)/len(idx)
                          for (rows,_,_),idx in zip(aux,ties))/len(aux) for k in range(len(b1)))
    assert b1 == tied_mean
    rows = [tuple(tuple(map(decimal,x)) for x in menu) for menu,_,_ in (*human,*aux)]
    expected = list(map(decimal,case['direction']))
    mode = case['offset']
    offsets = {
        'rank_one': (D(0),D(0)),
        'binary': (-D(2).ln(),-D(3).ln()/2),
        'grouped': (D(0),-D(3).ln()/2),
        'tangent': (-D(2).ln(),-D(3).ln()/2,D(0)),
        'attained': (D(0), D(3).ln()),
    }[mode]
    points = []
    for index,s in enumerate(distances):
        L = -s.ln()
        target = [s*decimal(h)+(1-s)*decimal(u) for h,u in zip(b0,b1)]
        sizes = [s/len(human)]*len(human)+[(1-s)/len(aux)]*len(aux)
        terms = lambda b:score_terms(rows,target,sizes,b)
        beta = fit(terms,[L*v+c for v,c in zip(expected,offsets)])
        zero_start_error = None
        if index in (0,len(distances)-1):
            numerical = fit(terms,[D(0)]*len(expected))
            zero_start_error = max(abs(x-y) for x,y in zip(beta,numerical))
            assert zero_start_error < D('1e-43')
        score = max(map(abs,independent_original_score(human,aux,s,beta)))
        assert score < D('1e-58'), (case['name'], str(s), str(score))
        error = max(abs(o-r-L*v) for o,r,v in zip(beta,offsets,expected))
        heldout_loss = (1+beta[1].exp()).ln()
        points.append(dict(s=str(s), beta=[str(x) for x in beta], original_score=str(score),
                           offset_error=str(error), heldout_logloss=str(heldout_loss),
                           zero_start_error=str(zero_start_error) if zero_start_error is not None else None))
    assert D(points[-1]['offset_error']) < D('1e-8')
    if mode=='attained':
        for p in points:
            s = D(p['s'])
            prob = (D('0.5')-D('0.125')*s)/(1-s/2)
            eq = (prob/(1-prob)).ln()
            assert abs(D(p['beta'][0])-eq)<D('1e-43')
            assert abs(D(p['beta'][1])-(D(3).ln()-eq))<D('1e-43')
    return dict(results=points,analytic_offset=list(map(str,offsets)),unit_target=list(map(str,b1)))


def run():
    distances = [D(repr(x)) for x in json.loads(Path(__file__).with_name('plan.json').read_text())['endpoint_distances']]
    reports = {}
    designs = examples()
    for x in designs:
        reports[x['name']] = direction_path(x) | numeric_path(x,distances)
    grouped = designs[2]
    change = ((2,1),(1,1))
    def recode(rows):
        return tuple((tuple(tuple(dot(v,col) for col in zip(*change)) for v in menu),y,g)
                     for menu,y,g in rows)
    transformed = grouped | dict(human=recode(grouped['human']),aux=recode(grouped['aux']),
                                 direction=(F(1),F(-1)))
    recoded_direction = direction_path(transformed)['direction']
    restored = np.array(change) @ recoded_direction
    assert np.max(np.abs(restored-np.array(grouped['direction'],dtype=float))) < 1e-8
    reports['equivalent_reencoding'] = dict(recoded_direction=recoded_direction,restored_direction=restored.tolist())
    attained = model('attained_rank_deficient',
                     tuple(binary(v,y,F(3,4)) for v in ((1,0),(1,1)) for y in (1,1,1,0)),
                     (auxiliary(((0,0),(1,0)),(0,1)),),(0,0),'attained')
    reports[attained['name']] = numeric_path(attained,distances)
    human = tuple(binary(x) for x in ((1,1),(1,-1)))
    aux = (binary((1,0)),)
    b0,b1 = ordinary_and_unit_moments(human,aux)
    assert b0[0]==b1[0]==F(1)
    assert all(menu[1][0]==1 for menu,_,_ in (*human,*aux))
    reports['no_finite_path'] = {'common_zero_recession_direction':[1,0], 'combined_rank':2}
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    with localcontext() as ctx:
        ctx.prec=90
        results=run()
    args.out.write_text(json.dumps(results,indent=2)+'\n')
    print({k:v.get('direction',v.get('unit_target',v.get('common_zero_recession_direction')))
           for k,v in results.items()})
