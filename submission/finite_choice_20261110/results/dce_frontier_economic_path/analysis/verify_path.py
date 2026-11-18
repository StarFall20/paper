"""Check raw menu objectives, endpoint geometry and economics independently."""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
DISTANCES = (0.5, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001)

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def load(p):
    return json.loads(p.read_text())

def csv_rows(p):
    with p.open(newline='') as stream:
        return list(csv.DictReader(stream))

def rank_mod_prime(matrix):
    """A modular rank lower bound certifies rational rank with an upper bound."""
    prime = 1009
    x = np.array(matrix, dtype=np.int64) % prime
    row = 0
    for col in range(x.shape[1]):
        ids = np.flatnonzero(x[row:, col])
        if not len(ids):
            continue
        candidate = row + ids[0]
        x[[row, candidate]] = x[[candidate, row]]
        x[row] = x[row] * pow(int(x[row, col]), -1, prime) % prime
        other = slice(row+1, None)
        x[other] = (x[other] - x[other, col, None]*x[row]) % prime
        row += 1
    return row

def probability(f, beta, mask=None):
    z = np.sum(f * beta, axis=2)
    if mask is not None:
        z = np.where(mask, z, -np.inf)
    return np.exp(z-logsumexp(z, axis=1)[:, None])

def raw_check(fL, y, gL, fU, gU, power, pL, pU):
    return np.mean(np.sum(((1-power)*pL - y + power*gL)[:, :, None]*fL, axis=1), axis=0) +         power*np.mean(np.sum((pU-gU)[:, :, None]*fU, axis=1), axis=0)

def check_economics(row, beta):
    bere, cost = beta[2], beta[10]
    assert float(row['price_beta']) == cost and float(row['bere_beta']) == bere
    sign = 'negative' if cost < -1e-9 else 'positive' if cost > 1e-9 else 'numerical_zero'
    assert row['price_state'] == sign
    ratio = -10.0 * bere / cost if cost else None
    assert (float(row['signed_Bere_GBP']) if row['signed_Bere_GBP'] else None) == ratio
    assert (float(row['negative_price_Bere_GBP']) if row['negative_price_Bere_GBP'] else None) == (ratio if sign == 'negative' else None)
    for premium in [0.0, 2.5, 5.0]:
        compare = (ratio > premium) if sign == 'negative' else None
        actual = row[f'accept_GBP_{premium:g}'] or None
        assert actual == (str(compare) if compare is not None else None)

def main(args):
    assert not args.out.exists()
    status = load(args.results / 'status.json')
    assert status['state'] == 'complete'
    assert sha(args.input) == status['workbook_sha256']
    assert sha(ROOT/'plan.json') == status['plan_sha256']
    assert sha(ROOT/'analysis/audit_path.py') == status['analysis_sha256']
    for group in ['frontier', 'baseline']:
        for name, h in status[group].items():
            assert sha(getattr(args, group) / name) == h, name
    sys.path.insert(0, str(args.frontier/'analysis'))
    from existing_panel_model import load as load_panel
    from verify_frontier import raw_frontier
    source = load(args.frontier/'inputs/model_plan.json')
    data = load_panel(args.input, source)
    teacher = np.asarray(load(args.frontier/'inputs/preflight.json')['teacher_coefficients'])
    teacher_logit = data['teacher'] @ teacher
    g = np.exp(teacher_logit-logsumexp(teacher_logit, axis=2, keepdims=True))
    rng = np.random.default_rng(source['allocation_seed'])
    orders = [rng.permutation(data['pool']) for _ in range(500)]
    evaluation = data['f'][data['test']].reshape(-1, 3, 11)
    choice = data['y'][data['test']].ravel()
    base_coefficients = {row['record']: np.asarray(row['beta']) for row in load(args.baseline/'results/coefficients.json')}
    base_ids = {(int(x['allocation']), int(x['labelled_people']), x['method']): int(x['record'])
                for x in csv_rows(args.baseline/'results/method_records.csv')}
    saved_cases = {row['frontier_record']: row for row in load(args.results/'cases.json')}
    coefs = {rid: np.asarray(beta) for rid,beta in json.loads(gzip.decompress((args.results/'coefficients.json.gz').read_bytes()))}
    path_rows = csv_rows(args.results/'paths.csv')
    groups = {}
    for row in path_rows:
        groups.setdefault(int(row['frontier_record']), []).append(row)
    upstream = csv_rows(args.frontier/'results/frontiers.csv')
    expected = {
        int(r['frontier_record']): 'uncapped_positive' if r['frontier_capped_at_one']=='False' else 'capped_low_budget'
        for r in upstream if float(r['frontier'])>1e-10 and (r['frontier_capped_at_one']=='False' or int(r['labelled_people'])<=10)
    }
    assert set(expected) == set(saved_cases) == set(groups)
    assert len(path_rows) == len(coefs) == 7536 and set(coefs) == set(range(7536))
    counts = Counter()
    maxima = Counter()
    endpoints = []
    for record_id, group in expected.items():
        case = saved_cases[record_id]
        m, trial = case['labelled_people'], case['allocation']
        assert group == case['group'] and tuple(float(x['relative_distance']) for x in groups[record_id]) == DISTANCES
        l, u = orders[trial][:m], orders[trial][m:]
        fL, fU = data['f'][l].reshape(-1,3,11), data['f'][u].reshape(-1,3,11)
        gL,gU = g[l].reshape(-1,3),g[u].reshape(-1,3)
        y = np.eye(3)[data['y'][l].ravel()]
        f = np.r_[fL,fU]
        state = case['facet_state']
        counts[state] += 1
        for name in ['labelled_ridge', 'all_human_MLE']:
            beta = base_coefficients[base_ids[(trial, m, name)]]
            control = case[name]
            formatted = {k: str(v) if v is not None else '' for k, v in control.items()}
            check_economics(formatted, beta)
            logits = evaluation @ beta
            loss = float(np.mean(logsumexp(logits, axis=1) - logits[np.arange(len(choice)), choice]))
            assert abs(loss-control['heldout_logloss'])<1e-12
        if group=='uncapped_positive':
            independent_cap = raw_frontier(data,l,u,g)
            difference = abs(independent_cap-case['saved_cap'])
            assert difference<1e-8 and abs(independent_cap-case['recomputed_cap'])<1e-8
            maxima['raw_endpoint_difference'] = max(maxima['raw_endpoint_difference'], difference)
            assert state=='transverse_facet', case['facet_state']
            v = np.array(case['exact_integer_normal'], dtype=np.int64)
            assert math.gcd(*(int(x) for x in v)) == 1
            d = v / math.sqrt(int(v @ v))
            contrasts = np.rint(20*(f-f[:, -1:, :])).astype(np.int64)
            assert np.max(abs(contrasts/20 - (f-f[:, -1:, :])))<1e-12
            logits = contrasts @ v
            gap = (logits.max(axis=1)[:, None] - logits) / (20*math.sqrt(int(v@v)))
            ties = gap==0
            tied = np.concatenate([z[t]-z[t][0] for z,t in zip(contrasts,ties)])
            assert rank_mod_prime(contrasts.reshape(-1,11))==11
            assert rank_mod_prime(tied)==10 and np.all(tied@v==0)
            rate_gap = float(gap[gap>0].min())
            assert abs(rate_gap-case['minimum_normal_gap'])<1e-14
            theta = np.array(case['tangent_coefficients'])
            assert abs(theta @ d) < 1e-10
            # Map the compressed certificate to individual menu rows, with no loss function imported.
            mapped_indices = np.r_[data['design_index'][l].ravel(), data['design_index'][u].ravel()]
            index = np.searchsorted(np.unique(mapped_indices), mapped_indices)
            witness = np.array(case['conditional_probability_witness'])[index]
            assert np.max(abs(witness.sum(axis=1)-1))<1e-9 and np.min(witness[ties])>1e-8
            assert np.max(abs(witness[~ties]))==0
            for prediction, key in [(witness, 'face_primal_moment'), (probability(f,theta,ties), 'face_score')]:
                s = raw_check(fL,y,gL,fU,gU,case['recomputed_cap'],prediction[:len(fL)],prediction[len(fL):])
                err = float(np.max(abs(s)))
                assert err < 1e-9
                maxima[key] = max(maxima[key],err)
            gapL, gapU = gap[:len(fL)], gap[len(fL):]
            s0 = math.fsum((gapL*y).ravel()) / len(fL)
            slope = math.fsum((gapL*gL).ravel()) / len(fL) - math.fsum((gapU*gU).ravel()) / len(fU)
            assert slope>0 and abs(slope-case['slack_slope'])<1e-12
            cap = s0/slope
            assert abs(cap-case['recomputed_cap'])<1e-10
            theta_utilities = f @ theta
            normalized = np.exp(np.where(gap==rate_gap, theta_utilities-logsumexp(np.where(ties,theta_utilities,-np.inf),axis=1)[:,None],-np.inf)).sum(axis=1)
            A = rate_gap * ((1-cap)*normalized[:len(fL)].mean() + cap*normalized[len(fL):].mean())
            assert math.isclose(A, case['leading_slack_constant'], rel_tol=1e-10, abs_tol=1e-10), (record_id, A, case['leading_slack_constant'])
            v_eval = evaluation@d
            loss_slope = float(np.mean(v_eval.max(axis=1) - v_eval[np.arange(len(choice)),choice]))
            assert abs(loss_slope-case['heldout_slope_per_normal_unit'])<1e-10
            endpoints.append(dict(frontier_record=record_id, uncompressed_frontier=independent_cap, cap_from_original_gap=cap,
                                  minimum_gap=rate_gap, min_tied_probability=float(witness[ties].min()),
                                  loss_slope_per_log_distance=loss_slope/rate_gap))
        for row in groups[record_id]:
            assert row['fit_state']=='verified_numerical_fit'
            beta = coefs[int(row['record'])]
            power = float(row['power'])
            assert 0 < power < case['saved_cap']
            assert abs(power-case['saved_cap']*(1-float(row['relative_distance'])))<1e-11
            pL,pU = probability(fL,beta),probability(fU,beta)
            raw_score = float(max(abs(raw_check(fL,y,gL,fU,gU,power,pL,pU))))
            assert raw_score < 1e-10, (record_id,row['record'],raw_score)
            maxima['raw_score'] = max(maxima['raw_score'],raw_score)
            lu = evaluation@beta
            raw_loss = float(np.mean(logsumexp(lu,axis=1)-lu[np.arange(len(choice)),choice]))
            maxima['holdout_logloss'] = max(maxima['holdout_logloss'],abs(raw_loss-float(row['heldout_logloss'])))
            assert abs(raw_loss-float(row['heldout_logloss']))<1e-10
            check_economics(row,beta)
            if group=='uncapped_positive':
                s = math.fsum([s0, -power*slope])
                graph = ((1-power)*math.fsum((pL*gapL).ravel())/len(fL)
                         + power*math.fsum((pU*gapU).ravel())/len(fU))
                assert graph>0.0 and s>0.0
                err = abs(graph-s)/s
                maxima['relative_normal_slack'] = max(maxima['relative_normal_slack'],err)
                assert err<1e-5, (record_id,row['record'],err)
                reconstructed = float(beta@d)
                assert abs(reconstructed-float(row['normal_coefficient']))<1e-9
                assert abs(np.linalg.norm(beta-reconstructed*d-theta)-float(row['tangent_error']))<1e-9
        if group=='capped_low_budget':
            beta = base_coefficients[base_ids[(trial,m,'corrected_unit')]]
            s = raw_check(fL,y,gL,fU,gU,1.0,probability(fL,beta),probability(fU,beta))
            assert max(abs(s))<1e-7
            check_economics({k:str(v) if v is not None else '' for k,v in case['unit_control'].items()},beta)

    args.out.mkdir(parents=True)
    (args.out/'original_row_endpoints.json').write_text(json.dumps(endpoints,indent=2)+'\n')
    verified = dict(state='passed', raw_scores_forecasts_and_economics_checked=len(path_rows), original_row_endpoints=len(endpoints),
                    geometric_rank_and_witness_checks=counts, maximum_errors=maxima,
                    source_reader_and_earlier_original_menu_LP_reused=True, primary_optimizer_imported=False, verifier_sha256=sha(Path(__file__)),
                    results_sha256={p.name:sha(p) for p in sorted(args.results.iterdir()) if p.is_file()})
    (args.out/'verification.json').write_text(json.dumps(verified,indent=2)+'\n')
    print(json.dumps(verified),flush=True)

if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    for name in ['input','frontier','baseline','results','out']:
        cli.add_argument('--'+name,type=Path,required=True)
    with threadpool_limits(limits=1):
        main(cli.parse_args())
