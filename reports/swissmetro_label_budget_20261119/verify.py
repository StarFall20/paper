"""Rebuild the original choice menus and verify the reported fits and weights."""
import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import qr
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csc_matrix, hstack, vstack
from scipy.special import softmax, logsumexp
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
WEIGHTS = ('zero', 'half', 'unit', 'cluster_plugin')


def input_data(path):
    data = pd.read_csv(path, sep='\t').query('CHOICE > 0')
    ids = data.ID.to_numpy()
    selected = data.CHOICE.to_numpy() - 1
    availability = np.column_stack((data.TRAIN_AV * data.SP, data.SM_AV, data.CAR_AV * data.SP)).astype(bool)
    free = (data.GA.to_numpy() == 0).astype(float)
    n = len(ids)
    intercepts = np.broadcast_to(np.array([[1, 0],[0, 0],[0, 1]]), (n,3,2))
    price = np.column_stack((data.TRAIN_CO * free, data.SM_CO * free, data.CAR_CO))
    time = np.column_stack((data.TRAIN_TT, data.SM_TT, data.CAR_TT))
    headways = np.zeros((n, 3, 2))
    headways[:, 0, 0] = data.TRAIN_HE
    headways[:, 1, 1] = data.SM_HE
    return ids, selected, availability, {
        'canonical_4': np.concatenate((100*intercepts, time[:,:,None], price[:,:,None]), axis=2) / 100,
        'mode_specific_8': np.concatenate((100*intercepts, time[:,:,None]*np.eye(3),
                                           price[:,:,None], headways), axis=2) / 100
    }


def utility(x, available, beta):
    return np.where(available, x @ np.asarray(beta), -np.inf)


def raw_loss(x, a, y, beta):
    u = utility(x, a, beta)
    return (logsumexp(u, axis=1) - u[np.arange(len(y)), y]).mean()


def means(x, p):
    return (x * p[:, :, None]).sum(axis=1)


def check_native(f, mask, w, target, beta):
    active = w > 0
    f, mask, w = f[active], mask[active], w[active]
    n = len(w)
    posterior = softmax(utility(f, mask, beta), axis=1)
    current = means(f, posterior)
    gap = target - w @ current
    score = float(np.linalg.norm(gap, ord=np.inf))
    menu, alt = np.nonzero(mask)
    reference = mask.shape[1] - 1 - mask[:, ::-1].argmax(axis=1)
    contr = f[menu,alt] - f[menu,reference[menu]]
    _, r, pivot = qr(contr.T, mode='economic', pivoting=True)
    d = len(target)
    assert sum(abs(np.diag(r)) > 1e-9) == d
    basis = pivot[:d]
    delta = np.linalg.solve(contr[basis].T, gap)
    pp = posterior.copy()
    for ix, mass in zip(basis,delta):
        i,j = menu[ix],alt[ix]
        pp[i,j] += mass / w[i]
        pp[i,reference[i]] -= mass / w[i]
    if min(pp[mask]) <= 0 or max(abs(pp.sum(axis=1)-1)) >= 1e-10:
        menu_sums = coo_matrix((np.ones(len(menu)),(menu,np.arange(len(menu)))),shape=(n,len(menu)))
        centered = f - f[np.arange(n), reference][:,None,:]
        moments = csc_matrix(centered[menu,alt].T)
        uniform = mask/mask.sum(axis=1,keepdims=True)
        uniform_target = w @ means(centered,uniform)
        anchor = w @ f[np.arange(n),reference]
        equality = vstack((menu_sums,moments),format='csc')
        equality = hstack((equality,csc_matrix(np.r_[w,uniform_target][:,None])),format='csc')
        lp = linprog(np.r_[np.zeros(len(menu)),-1], A_eq=equality, b_eq=np.r_[w,target-anchor],
                     bounds=[(0,None)]*len(menu)+[(0,1)], method='highs-ds',
                     options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
        assert lp.success and lp.x[-1] > 1e-8
        assert max(abs(equality@lp.x-np.r_[w,target-anchor])) < 1e-8
    else:
        assert max(abs(w @ means(f,pp)-target)) < 1e-10
    assert score < 1e-8
    return score


def verify_tuning(f, a, choices, predictions, picked_by_row, person_ids, ridge):
    p = softmax(utility(f,a,ridge),axis=1)
    feature_mean = means(f,p)
    centered_x = f - feature_mean[:,None,:]
    h = np.sum(p[:,:,None,None] * (centered_x[:,:,:,None] * centered_x[:,:,None,:]), axis=(0,1))/len(f)
    g = feature_mean - means(f,predictions)
    lf = f[picked_by_row]
    ly = choices[picked_by_row]
    y = feature_mean[picked_by_row] - lf[np.arange(len(ly)),ly]
    all_ids = np.unique(person_ids)
    lab_ids = np.unique(person_ids[picked_by_row])
    groups = pd.DataFrame(g).groupby(person_ids,sort=True).mean().loc[all_ids].to_numpy() @ np.linalg.inv(h)
    lab = pd.DataFrame(y).groupby(person_ids[picked_by_row],sort=True).mean().loc[lab_ids].to_numpy() @ np.linalg.inv(h)
    subset = np.isin(all_ids,lab_ids)
    gl, gu = groups[subset], groups[~subset]
    d = f.shape[-1]
    covariance = np.cov(np.column_stack((lab,gl)),rowvar=False)
    cross = float(np.trace(covariance[:d,d:]))
    variance = float(np.trace(covariance[d:,d:]) + len(gl)/len(gu)*np.trace(np.cov(gu,rowvar=False)))
    return (float(np.clip(cross/variance,0,1)) if variance > 1e-14 else 0.0,cross,variance)


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def run(args):
    plan = json.loads((HERE/'plan.json').read_text())
    assert sha256(args.input.read_bytes()).hexdigest() == plan['source_sha256']
    ids, y, available, features = input_data(args.input)
    people = np.random.default_rng(plan['split_seed']).permutation(np.unique(ids))
    tr, test, pool = np.split(people, [plan['teacher_respondents'],
                            plan['teacher_respondents']+plan['evaluation_respondents']])
    teacher, evaluation, pooled = [np.isin(ids,p) for p in (tr,test,pool)]
    source = json.loads(args.precedent_summary.read_text())
    reference_beta = source['input_summary']['teacher_coefs']
    frozen = softmax(utility(features['canonical_4'][pooled],available[pooled],reference_beta),axis=1)
    pool_y, pool_mask, pool_ids = y[pooled], available[pooled], ids[pooled]
    assert len(pool_ids)==3159 and len(y[evaluation])==2160
    orders = [np.random.default_rng(plan['allocation_seed']+k).permutation(pool) for k in range(plan['allocations'])]
    old_states = {(r['specification'],int(r['allocation']),int(r['labelled_people'])):r['endpoint']=='unit_interior'
                   for r in read_csv(args.precedent_allocations)}
    counts, errors = Counter(), Counter()
    for result_dir in args.results:
        summary = json.loads((result_dir/'summary.json').read_text())
        assert np.max(np.abs(np.array(summary['teacher_beta'])-reference_beta)) < 1e-11
        records = {(r['specification'],int(r['allocation']),int(r['label_budget']),r['method']):r
                   for r in read_csv(result_dir/'records.csv')}
        for line in (result_dir/'coefficients.jsonl').read_text().splitlines():
            fits = json.loads(line)
            name, trial, budget = (fits[k] for k in ('model','allocation','label_budget'))
            all_x = features[name]
            x, loss_x = all_x[pooled], all_x[evaluation]
            observed = np.isin(pool_ids,orders[trial][:budget])
            rest = ~observed
            xl, yl = x[observed], pool_y[observed]
            b0 = xl[np.arange(len(yl)),yl].mean(axis=0)
            shift = means(x[rest],frozen[rest]).mean(axis=0)-means(xl,frozen[observed]).mean(axis=0)
            w_l, w_u = observed/np.sum(observed), rest/np.sum(rest)
            ridge_loss = raw_loss(loss_x,available[evaluation],y[evaluation],fits['ridge'])
            direct_loss = raw_loss(loss_x,available[evaluation],y[evaluation],fits['total_label_direct'])
            direct_rows = teacher.copy()
            direct_rows[pooled] = observed
            for label, f, mask, human, penalty in [
                ('ridge', xl, pool_mask[observed], yl, plan['same_label_ridge']),
                ('total_label_direct', all_x[direct_rows], available[direct_rows], y[direct_rows], 0.0)]:
                b = np.asarray(fits[label])
                direct_p = softmax(utility(f,mask,b),axis=1)
                score = float(max(abs((means(f,direct_p)-f[np.arange(len(human)),human]).mean(axis=0)+penalty*b)))
                assert score < 1e-8
                errors['direct_or_ridge_score'] = max(errors['direct_or_ridge_score'],score)
                counts[label+'_optimality_checks'] += 1
            tuning = verify_tuning(x,pool_mask,pool_y,frozen,observed,pool_ids,fits['ridge'])
            cache = {}
            for method in WEIGHTS:
                key = (name,trial,budget,method)
                row = records.pop(key)
                w = float(row['weight'])
                if method == 'cluster_plugin':
                    deltas = [abs(u-v) for u,v in zip(tuning,(w,float(row['plugin_cross']),float(row['plugin_variance'])))]
                    assert deltas[0] < 2e-11 and max(deltas) < 1e-7
                    errors['plugin_weight'] = max(errors['plugin_weight'],deltas[0])
                    counts['plugin_weight_checks'] += 1
                assert abs(ridge_loss-float(row['ridge_loss'])) < 1e-12
                assert abs(direct_loss-float(row['direct_loss'])) < 1e-12
                case, loss = row['state'], ridge_loss
                weights, target = (1-w)*w_l+w*w_u, b0+w*shift
                if case == 'interior':
                    if w not in cache:
                        try:
                            cache[w] = check_native(x,pool_mask,weights,target,fits[method])
                        except AssertionError as exc:
                            raise AssertionError((name,trial,budget,method,'native support failed')) from exc
                    errors['native_score'] = max(errors['native_score'],cache[w])
                    loss = raw_loss(loss_x,available[evaluation],y[evaluation],fits[method])
                    assert abs(loss-float(row['native_loss'])) < 1e-12
                elif case == 'exterior':
                    d = np.array(json.loads(row['direction']))
                    rec = weights @ utility(x,pool_mask,d).max(axis=1) - target @ d
                    assert rec < -1e-8 and abs(rec-float(row['recession'])) < 1e-12
                elif case == 'boundary':
                    direction = json.loads(row['exact_boundary_direction'])
                    assert w == 0
                    grid = np.rint(100*(xl-xl[np.arange(len(yl)),yl][:,None,:])).astype(object)
                    grid = np.vectorize(int)(grid).astype(object)
                    signs = grid[pool_mask[observed]] @ direction
                    assert max(signs) <= 0 and min(signs) < 0
                elif case == 'rank_deficient':
                    cross = x[weights>0]
                    avail = pool_mask[weights>0]
                    contrasts = (cross - cross[np.arange(len(cross)),avail.argmax(axis=1)][:,None,:])[avail]
                    assert np.linalg.matrix_rank(contrasts,tol=1e-9) < x.shape[-1]
                else:
                    counts['unresolved_or_fit_failure'] += 1
                assert abs(loss-float(row['policy_loss'])) < 1e-12
                counts[case+'_records'] += 1
                counts['policy_loss_checks'] += 1
                if method == 'unit' and (name,trial,budget) in old_states:
                    assert (case=='interior')==old_states[name,trial,budget],(name,trial,budget,case)
                    counts['archived_unit_status_matches'] += 1
        assert not records
    assert counts['policy_loss_checks'] == 2688 * args.specification_count
    assert counts['archived_unit_status_matches'] == 288 * args.specification_count
    args.out.write_text(json.dumps(dict(status='passed',checks=counts,maximum_errors=errors),indent=2)+'\n')
    print(dict(counts),dict(errors))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--results',type=Path,nargs='+',required=True)
    parser.add_argument('--precedent-summary',type=Path,required=True)
    parser.add_argument('--precedent-allocations',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--specification-count',type=int,choices=(1,2),default=2)
    with threadpool_limits(limits=1):
        run(parser.parse_args())
