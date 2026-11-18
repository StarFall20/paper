"""Follow the original compensated-choice fit to its observed upper power."""
import argparse
import csv
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import linprog, root
from scipy.special import logsumexp, softmax
from threadpoolctl import threadpool_limits

DISTANCES = (0.5, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001)
PRICE, BERE = 10, 2
CUTOFFS = (0.0, 2.5, 5.0)
MARGINS = {'primal_feasibility_tolerance': 1e-9, 'dual_feasibility_tolerance': 1e-9}
BASE = Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_json(path):
    return json.loads(path.read_text())

def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False, indent=2) + '\n')

def read_rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))

def write_rows(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

def check_saved_file(package, relative):
    hashes = dict(line.rstrip().split('  ', 1)[::-1] for line in (package / 'MANIFEST.sha256').read_text().splitlines())
    path = package / relative
    assert sha(path) == hashes[relative], path
    return path

def integer_normal(contrasts, orientation):
    """Return an exact primitive normal when tied contrasts have rank d-1."""
    n, dim = contrasts.shape
    span = [[Fraction(int(z)) for z in row] for row in contrasts]
    pivots = []
    for col in range(dim):
        idx = next((i for i in range(len(pivots), n) if span[i][col]), None)
        if idx is None:
            continue
        j = len(pivots)
        span[idx], span[j] = span[j], span[idx]
        pivot = span[j][col]
        span[j] = [z / pivot for z in span[j]]
        for i, row in enumerate(span):
            if i != j and row[col]:
                span[i] = [a - row[col] * b for a, b in zip(row, span[j])]
        pivots.append(col)
    if len(pivots) != dim - 1:
        return None, len(pivots)
    free = next(i for i in range(dim) if i not in pivots)
    d = [Fraction(0)] * dim
    d[free] = Fraction(1)
    for j, col in enumerate(pivots):
        d[col] = -span[j][free]
    multiple = math.lcm(*(z.denominator for z in d))
    v = np.array([int(z * multiple) for z in d], dtype=np.int64)
    divisor = math.gcd(*(int(z) for z in v))
    v //= divisor
    return (v if v @ orientation >= 0 else -v), len(pivots)

def objective(features, weights, target, beta, mask=None):
    utility = features @ beta
    if mask is not None:
        utility = np.where(mask, utility, -np.inf)
    normalizer = logsumexp(utility, axis=1)
    p = np.exp(utility - normalizer[:, None])
    means = np.einsum('nj,njp->np', p, features)
    score = weights @ means - target
    centered = features - means[:, None, :]
    hessian = np.einsum('n,nj,nja,njb->ab', weights, p, centered, centered)
    return weights @ normalizer - target @ beta, score, hessian

def solve(features, weights, target, start, mask=None):
    def terms(x):
        return objective(features, weights, target, x, mask)
    beta = start.copy()
    for _ in range(100):
        value, score, hessian = terms(beta)
        try:
            step = np.linalg.solve(hessian, score)
        except np.linalg.LinAlgError:
            break
        if max(abs(score)) <= 3e-13 and max(abs(step)) <= 1e-6:
            return beta
        projected = score @ step
        if projected <= 0.0:
            break
        for exponent in range(30):
            alpha = 2.0 ** -exponent
            proposal = beta - alpha * step
            if terms(proposal)[0] <= value - 1e-4 * alpha * projected + 2e-14:
                beta = proposal
                break
        else:
            break

    candidate = root(lambda x: terms(x)[1], beta, jac=lambda x: terms(x)[2], method='hybr',
                     options={'xtol': 1e-10, 'maxfev': 1000}).x
    if np.all(np.isfinite(candidate)) and np.linalg.norm(terms(candidate)[1]) < np.linalg.norm(terms(beta)[1]):
        beta = candidate
    return beta

def relative_margin(features, weights, target, ties):
    ni, nj = np.nonzero(ties)
    n, _, p = features.shape
    size = len(ni)
    eq = np.zeros((n+p, size+1))
    eq[ni, np.arange(size)] = 1
    eq[n:, :size] = (weights[ni, None] * features[ni, nj]).T
    ub = np.c_[-np.eye(size), np.ones(size)]
    optimum = linprog(np.r_[np.zeros(size), -1.0], A_ub=ub, b_ub=np.zeros(size),
                      A_eq=eq, b_eq=np.r_[np.ones(n), target],
                      bounds=[(0, 1)] * size + [(0, 1)], options=MARGINS, method='highs')
    if optimum.success:
        row_probabilities = np.zeros_like(ties, dtype=float)
        row_probabilities[ni, nj] = optimum.x[:-1]
        return float(optimum.x[-1]), row_probabilities.tolist()
    return None, None

def facet_endpoint(f, w0, w1, b0, b1, old_cap, saved_direction):
    orientation = np.asarray(saved_direction)
    orientation /= np.linalg.norm(orientation)
    utility = f @ orientation
    provisional = utility.max(axis=1, keepdims=True) - utility <= 1e-8
    # The raw within-menu differences are multiples of 1/20 under recorded coding.
    differences = f - f[:, -1:, :]
    grid = np.rint(20 * differences).astype(np.int64)
    assert np.max(np.abs(differences - grid/20)) < 1e-12
    tie_differences = np.vstack([x[mask] - x[mask][0] for x, mask in zip(grid, provisional)])
    v, rank = integer_normal(tie_differences, orientation)
    report = dict(facet_state='tied_contrast_rank_below_facet', tangent_rank=rank,
                  whole_contrast_rank=int(np.linalg.matrix_rank(grid.reshape(-1, grid.shape[-1]))))
    if v is None:
        return report, None
    if report['whole_contrast_rank'] != f.shape[-1]:
        report['facet_state'] = 'full_design_rank_deficient'
        return report, None
    exact_utility = grid @ v
    integer_gaps = exact_utility.max(axis=1, keepdims=True) - exact_utility
    ties = integer_gaps == 0
    if not np.array_equal(ties, provisional):
        report['facet_state'] = 'floating_and_exact_tie_sets_disagree'
        return report, None
    norm = np.linalg.norm(v)
    d = v / norm
    gaps = integer_gaps / (20 * norm)
    values = (f @ d).max(axis=1)
    s0 = float(w0 @ values - b0 @ d)
    s1 = float(w1 @ values - b1 @ d)
    slope = s0 - s1
    if slope <= 0:
        report['facet_state'] = 'normal_not_transverse'
        return report, None
    cap = s0 / slope
    report.update(exact_integer_normal=v.tolist(), recomputed_cap=float(cap), cap_agreement=abs(cap - old_cap),
                  slack_slope=slope)
    if abs(cap - old_cap) > 1e-8:
        report['facet_state'] = 'normals_give_different_endpoint'
        return report, None
    w = (1-cap) * w0 + cap * w1
    b = (1-cap) * b0 + cap * b1
    margin, probabilities = relative_margin(f, w, b, ties)
    report.update(tied_probability_margin=margin)
    if margin is None or margin <= 1e-8:
        report['facet_state'] = 'endpoint_not_certified_in_relative_interior'
        return report, None

    basis = null_space(d[None, :])
    tangent_features = f @ basis
    tangent_target = b @ basis
    z = solve(tangent_features, w, tangent_target, np.zeros(f.shape[-1]-1), ties)
    _, grad, hess = objective(tangent_features, w, tangent_target, z, ties)
    face_beta = basis @ z
    smallest = integer_gaps[~ties].min()
    smallest_gap = float(smallest / (20 * norm))
    on_minimum = integer_gaps == smallest
    utility = f @ face_beta
    tied_normalizer = logsumexp(np.where(ties, utility, -np.inf), axis=1)
    leading = float(smallest_gap * (w @ np.exp(np.where(on_minimum, utility - tied_normalizer[:, None], -np.inf)).sum(axis=1)))
    min_hess = float(np.linalg.eigvalsh(hess)[0])
    report.update(minimum_normal_gap=smallest_gap, leading_slack_constant=leading, tangent_score=float(max(abs(grad))),
                  tangent_minimum_curvature=min_hess, tangent_coefficients=face_beta.tolist(),
                  conditional_probability_witness=probabilities)
    if max(abs(grad)) > 1e-10 or min_hess <= 0.0 or leading <= 0.0:
        report['facet_state'] = 'face_fit_failed'
        return report, None
    report['facet_state'] = 'transverse_facet'
    return report, (d, gaps, face_beta)

def economics(beta):
    price, bere = float(beta[PRICE]), float(beta[BERE])
    sign = 'negative' if price < -1e-9 else 'positive' if price > 1e-9 else 'numerical_zero'
    ratio = -10.0 * bere / price if price != 0.0 else None
    answer = dict(price_beta=price, bere_beta=bere, price_state=sign, signed_Bere_GBP=ratio,
                  negative_price_Bere_GBP=ratio if sign == 'negative' else None)
    for premium in CUTOFFS:
        answer[f'accept_GBP_{premium:g}'] = bool(bere + (premium/10) * price > 0.0) if sign == 'negative' else None
    return answer

def loss_eval(features, chosen, beta):
    u = features @ beta
    return float(np.mean(logsumexp(u, axis=1) - u[np.arange(len(chosen)), chosen]))

def main(args):
    assert not args.out.exists()
    source_files = ('plan.json', 'results/frontiers.csv', 'results/certificates.json.gz', 'inputs/preflight.json',
                    'inputs/model_plan.json', 'inputs/baseline_records.csv', 'inputs/labelled_ridge_coefficients.json',
                    'analysis/existing_panel_model.py')
    pinned = {rel: check_saved_file(args.frontier, rel) for rel in source_files}
    baseline_files = {rel: check_saved_file(args.baseline, rel) for rel in ['results/coefficients.json', 'results/method_records.csv']}
    assert sha(baseline_files['results/method_records.csv']) == sha(pinned['inputs/baseline_records.csv'])
    sys.path.insert(0, str(args.frontier / 'analysis'))
    from existing_panel_model import load, corrected

    plan = load_json(pinned['inputs/model_plan.json'])
    data = load(args.input, plan)
    teacher = np.array(load_json(pinned['inputs/preflight.json'])['teacher_coefficients'])
    predicted = softmax(data['teacher'] @ teacher, axis=2)
    frontiers = read_rows(pinned['results/frontiers.csv'])
    cert = json.loads(gzip.decompress(pinned['results/certificates.json.gz'].read_bytes()))
    original = {(int(r['allocation']), int(r['labelled_people']), r['method']): r for r in read_rows(baseline_files['results/method_records.csv'])}
    betas = {r['record']: np.asarray(r['beta']) for r in load_json(baseline_files['results/coefficients.json'])}
    eval_f = data['f'][data['test']].reshape(-1, 3, 11)
    eval_y = data['y'][data['test']].ravel()
    state = {'uncapped_positive': 0, 'capped_low_budget': 0, 'capped_higher_budget': 0, 'zero_or_near_zero': 0}
    cases, records, coefficient_records = [], [], []
    rng = np.random.default_rng(plan['allocation_seed'])
    orders = [rng.permutation(data['pool']) for _ in range(500)]
    for entry in frontiers:
        ticket, trial, m = (int(entry[k]) for k in ('frontier_record','allocation','labelled_people'))
        old_cap = float(entry['frontier'])
        if old_cap <= 1e-10:
            group = 'zero_or_near_zero'
        elif entry['frontier_capped_at_one'] == 'True':
            group = 'capped_low_budget' if m <= 10 else 'capped_higher_budget'
        else:
            group = 'uncapped_positive'
        state[group] += 1
        if group not in ('uncapped_positive', 'capped_low_budget'):
            continue
        case = dict(frontier_record=ticket, allocation=trial, labelled_people=m, group=group, saved_cap=old_cap,
                    old_human_support=entry['human_support'])
        order = orders[trial]
        L, U = order[:m], order[m:]
        zero, unit = corrected(data, L, U, predicted, 0.0), corrected(data, L, U, predicted, 1.0)
        keep = (zero.w > 0) | (unit.w > 0)
        f, w0, w1, b0, b1 = zero.f[keep], zero.w[keep], unit.w[keep], zero.b, unit.b
        face_params = None
        ridge_record = original[(trial, m, 'labelled_ridge')]
        all_human = original[(trial, m, 'all_human_MLE')]
        ridge_beta = betas[int(ridge_record['record'])]
        human_beta = betas[int(all_human['record'])]
        for name, beta in [('labelled_ridge', ridge_beta), ('all_human_MLE', human_beta)]:
            case[name] = {**economics(beta), 'heldout_logloss': loss_eval(eval_f, eval_y, beta)}
        if group == 'uncapped_positive':
            face, face_params = facet_endpoint(f, w0, w1, b0, b1, old_cap, cert[ticket]['direction'])
            case.update(face)
            if face_params is not None:
                d, gaps, theta = face_params
                eval_utility = eval_f @ d
                eval_gaps = eval_utility.max(axis=1) - eval_utility[np.arange(len(eval_y)), eval_y]
                case['heldout_slope_per_normal_unit'] = float(eval_gaps.mean())
                case['heldout_slope_per_log_distance'] = float(eval_gaps.mean()/case['minimum_normal_gap'])
                v = case['exact_integer_normal']
                if v[PRICE] != 0:
                    case['face_ratio_limit_GBP'] = -10.0 * v[BERE] / v[PRICE]
                    case['ratio_limit_state'] = 'negative_price_normal' if v[PRICE] < 0 else 'positive_price_normal'
                elif v[BERE] == 0:
                    case['ratio_limit_state'] = 'both_normal_components_zero'
                    case['face_ratio_limit_GBP'] = -10.0 * theta[BERE] / theta[PRICE] if abs(theta[PRICE]) > 1e-9 else None
                else:
                    case['ratio_limit_state'] = 'zero_price_nonzero_Bere_normal'
                    case['face_ratio_limit_GBP'] = None
        else:
            assert original[(trial, m, 'corrected_unit')]['fit'] == 'passed_score'
            case['facet_state'] = 'unit_endpoint_interior'
            unit_beta = betas[int(original[(trial, m, 'corrected_unit')]['record'])]
            case['unit_control'] = economics(unit_beta)

        beta = ridge_beta.copy()
        if face_params is not None:
            d, gaps, _ = face_params
            tangent = null_space(d[None, :])
            rotation = np.column_stack([d, tangent])
            reference = f[np.arange(len(f)), gaps.argmin(axis=1)]
            local_f = np.concatenate([-gaps[:, :, None], (f-reference[:, None, :]) @ tangent], axis=2)
        else:
            rotation = np.eye(len(beta))
            local_f = f
        for dist in DISTANCES:
            power = (old_cap if group == 'uncapped_positive' else 1.0) * (1.0-dist)
            weights, target = (1-power)*w0+power*w1, (1-power)*b0+power*b1
            local_target = target
            if face_params is not None:
                small_slack = case['slack_slope'] * (case['recomputed_cap'] - power)
                local_target = np.r_[-small_slack, (target - weights @ reference) @ tangent]
            local_beta = solve(local_f, weights, local_target, rotation.T @ beta)
            beta = rotation @ local_beta
            _, _, score_hessian = objective(f, weights, target, beta)
            _, score, hessian = objective(local_f, weights, local_target, local_beta)
            curvature = float(np.linalg.eigvalsh(score_hessian)[0])
            step = float(max(abs(rotation @ np.linalg.solve(hessian, score)))) if curvature > 0 else None
            fit_state = 'verified_numerical_fit' if step is not None and step < 3e-5 and max(abs(score)) < 1e-10 else 'unresolved_numerical_fit'
            rid = len(records)
            rec = dict(record=rid, frontier_record=ticket, allocation=trial, labelled_people=m, group=group,
                       relative_distance=dist, power=power, fit_state=fit_state, score=float(max(abs(score))),
                       minimum_curvature=curvature, Newton_step=step)
            if fit_state == 'verified_numerical_fit':
                rec.update(economics(beta), beta_norm=float(np.linalg.norm(beta)),
                           heldout_logloss=loss_eval(eval_f, eval_y, beta))
                if face_params is not None:
                    d, gaps, theta = face_params
                    t = float(beta @ d)
                    tangential = beta - t*d
                    predicted_t = np.log(case['leading_slack_constant']/(case['slack_slope']*(case['recomputed_cap']-power))) / case['minimum_normal_gap']
                    normal_moment = float(weights @ (softmax(f @ beta, axis=1) * gaps).sum(axis=1))
                    rec.update(normal_coefficient=t, predicted_normal_coefficient=float(predicted_t),
                               normal_prediction_error=t-float(predicted_t), tangent_error=float(np.linalg.norm(tangential-theta)),
                               predicted_slack=float(case['slack_slope']*(case['recomputed_cap']-power)),
                               probability_slack=normal_moment)
            coefficient_records.append((rid, beta.tolist()))
            records.append(rec)
        if group == 'capped_low_budget':
            case['unit_last_beta_difference'] = float(np.linalg.norm(unit_beta-beta))
        cases.append(case)
        if len(cases) % 150 == 0:
            print(json.dumps({'cases_complete': len(cases), 'path_records': len(records)}), flush=True)

    assert state == {'uncapped_positive': 107, 'capped_low_budget': 835, 'capped_higher_budget': 1000, 'zero_or_near_zero': 58}
    args.out.mkdir(parents=True)
    write_rows(args.out / 'paths.csv', records)
    write_json(args.out / 'cases.json', cases)
    packed = json.dumps(coefficient_records, allow_nan=False, separators=(',', ':')).encode()
    (args.out / 'coefficients.json.gz').write_bytes(gzip.compress(packed, mtime=0))
    write_json(args.out / 'status.json', dict(state='complete', cases=state, fitted_path_points=len(records),
               verified_path_points=sum(row['fit_state']=='verified_numerical_fit' for row in records),
               analysis_sha256=sha(Path(__file__)), plan_sha256=sha(BASE/'plan.json'), workbook_sha256=sha(args.input),
               frontier={n:sha(p) for n,p in pinned.items()}, baseline={n:sha(p) for n,p in baseline_files.items()}))
    print((args.out/'status.json').read_text(), flush=True)

if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'frontier', 'baseline', 'out'):
        cli.add_argument('--'+name, type=Path, required=True)
    with threadpool_limits(limits=1):
        main(cli.parse_args())
