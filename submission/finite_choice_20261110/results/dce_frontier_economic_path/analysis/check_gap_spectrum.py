"""Evaluate the post-hoc fixed-face scalar approximation and its bound."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp, softmax
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load(path):
    return json.loads(path.read_text())

def main(args):
    assert not args.out.exists()
    status = load(args.results/'status.json')
    assert status['verified_path_points'] == 7536 and sha(args.input) == status['workbook_sha256']
    for filename, h in status['frontier'].items():
        assert sha(args.frontier/filename) == h, filename
    sys.path.insert(0, str(args.frontier/'analysis'))
    from existing_panel_model import load as load_panel, corrected
    source = load(args.frontier/'inputs/model_plan.json')
    data = load_panel(args.input, source)
    teacher = np.asarray(load(args.frontier/'inputs/preflight.json')['teacher_coefficients'])
    g = softmax(data['teacher'] @ teacher, axis=2)
    rng = np.random.default_rng(source['allocation_seed'])
    orders = [rng.permutation(data['pool']) for _ in range(500)]
    evaluation = data['f'][data['test']].reshape(-1,3,11)
    chosen = data['y'][data['test']].ravel()
    cases = {c['frontier_record']:c for c in load(args.results/'cases.json') if c['facet_state']=='transverse_facet'}
    with (args.results/'paths.csv').open(newline='') as stream:
        paths = list(csv.DictReader(stream))
    params = {}
    for fid,case in cases.items():
        m = case['labelled_people']
        order = orders[case['allocation']]
        L,U = order[:m],order[m:]
        zero,unit = [corrected(data,L,U,g,z) for z in [0.0,1.0]]
        keep = (zero.w>0)|(unit.w>0)
        features,w0,w1 = zero.f[keep],zero.w[keep],unit.w[keep]
        d = np.asarray(case['exact_integer_normal'])
        d = d/np.linalg.norm(d)
        tangent = features-(features@d)[:,:,None]*d
        grid = np.rint(20*(features-features[:,-1:,:])).astype(np.int64)
        v = np.asarray(case['exact_integer_normal'])
        utility = grid@v
        gap = (utility.max(axis=1,keepdims=True)-utility)/(20*np.linalg.norm(v))
        tied = gap==0
        difference = np.linalg.norm(tangent[:,:,None,:]-tangent[:,None,:,:],axis=3)
        gap_diff = abs(gap[:,:,None]-gap[:,None,:])
        kmax = float((difference[gap_diff>0]/gap_diff[gap_diff>0]).max())
        diameter = float(difference.max())
        bprime = unit.b-zero.b
        bq = bprime-(bprime@d)*d
        bound = (diameter*case['slack_slope']/case['minimum_normal_gap']
                 +abs(w1-w0)@np.max(np.where(tied,np.linalg.norm(tangent,axis=2),0.0),axis=1)+np.linalg.norm(bq))
        r = 1/(2*diameter)
        curvature = case['tangent_minimum_curvature']/np.e
        ustar = np.asarray(case['tangent_coefficients'])
        params[fid] = (d,gap,ustar,features@ustar,w0,w1,kmax,bound,r,curvature)
    records=[]
    for path in paths:
        fid=int(path['frontier_record'])
        if fid not in cases:
            continue
        d,gaps,ustar,utilities,w0,w1,kmax,bound,r,curvature = params[fid]
        power=float(path['power'])
        eps = cases[fid]['recomputed_cap']-power
        slack = cases[fid]['slack_slope']*eps
        w = (1-power)*w0+power*w1
        def normal_score(t):
            p = softmax(utilities-t*gaps,axis=1)
            return w@np.sum(p*gaps,axis=1)-slack
        left,right=-1.0,1.0
        while normal_score(left)<0:
            left *= 2
        while normal_score(right)>0:
            right *= 2
        t=brentq(normal_score,left,right,xtol=1e-12)
        decision_bound=None
        tangent_bound=None
        if bound*eps < r*curvature:
            tangent_bound = float(bound*eps/curvature)
            decision_bound = kmax*tangent_bound
        err=abs(t-float(path['normal_coefficient']))
        tang_err=float(path['tangent_error'])
        assert err <= kmax*tang_err+1e-5
        if tangent_bound is not None:
            assert err <= decision_bound+1e-5 and tang_err <= tangent_bound + 1e-8
        z = evaluation@(t*d+ustar)
        loss = float(np.mean(logsumexp(z,axis=1)-z[np.arange(len(chosen)),chosen]))
        records.append(dict(record=int(path['record']),frontier_record=fid,labelled_people=int(path['labelled_people']),
                            relative_distance=float(path['relative_distance']),scalar_normal=t,
                            scalar_score_abs_error=abs(normal_score(t)),normal_absolute_error=err,
                            leading_gap_absolute_error=abs(float(path['normal_prediction_error'])),
                            tangent_kmax=kmax,ex_post_error_bound=kmax*tang_err,
                            sufficient_tangent_bound=tangent_bound,sufficient_normal_bound=decision_bound,
                            scalar_holdout_logloss=loss,full_holdout_logloss=float(path['heldout_logloss']),
                            holdout_absolute_error=abs(loss-float(path['heldout_logloss']))))
    assert len(records)==856
    args.out.mkdir(parents=True)
    with (args.out/'spectrum.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summaries=[]
    for dist in sorted({r['relative_distance'] for r in records},reverse=True):
        rows=[r for r in records if r['relative_distance']==dist]
        assert len(rows)==107
        select=lambda name: np.quantile([r[name] for r in rows],[.05,.5,.95]).tolist()
        summaries.append(dict(distance=dist,records=len(rows),sufficient_bound_applicable=sum(r['sufficient_normal_bound'] is not None for r in rows),
                              normal_error_5_50_95=select('normal_absolute_error'),leading_gap_error_5_50_95=select('leading_gap_absolute_error'),
                              loss_approximation_error_5_50_95=select('holdout_absolute_error')))
    result=dict(state='passed',posthoc_path_points=len(records),max_scalar_score_abs_error=max(r['scalar_score_abs_error'] for r in records),
                fixed_u_uses_observed_evaluation_choices=False, no_retuned_weights=True,by_distance=summaries,
                source_plan_sha256=sha(ROOT/'diagnostics/gap_spectrum_plan.json'),analysis_sha256=sha(Path(__file__)),
                csv_sha256=sha(args.out/'spectrum.csv'))
    (args.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    for name in ['input','frontier','results','out']:
        cli.add_argument('--'+name,type=Path,required=True)
    with threadpool_limits(limits=1):
        main(cli.parse_args())
