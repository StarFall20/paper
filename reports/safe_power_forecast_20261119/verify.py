"""Verify original row scores, probability losses and policy comparisons."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from inputs import ROOT, digest, read, sources, sets

SAFETY=('safe_half','safe_edge')
FIXED=('fixed_half','fixed_unit')


def row_prob(x,a,beta):
    # Perform the original row calculation without the fitted objective or logsumexp.
    u=x@beta
    peak=np.max(np.where(a,u,-np.inf),axis=1)
    exponent=np.zeros_like(u)
    exponent[a]=np.exp((u-peak[:,None])[a])
    z=exponent.sum(axis=1)
    return u,peak,z,exponent/z[:,None]


def original_score(panel,beta,take,proxy=False):
    x,available=panel.x[take],panel.available[take]
    _,_,_,mu=row_prob(x,available,beta)
    target=panel.proxy[take] if proxy else np.eye(mu.shape[1])[panel.y[take]]
    return ((mu-target)[:,:,None]*x).sum(axis=1).mean(axis=0)


def check_metrics(panel,beta):
    x,available,y=panel.x[panel.test],panel.available[panel.test],panel.y[panel.test]
    row=np.arange(len(y))
    if beta is None:
        p=panel.proxy[panel.test]
        loss=-np.log(p[row,y])
    else:
        u,m,z,p=row_prob(x,available,beta)
        loss=(m-u[row,y])+np.log(z)
    brier=np.square(p-np.eye(p.shape[1])[y]).sum(axis=1)
    ids=panel.ids[panel.test]
    people=np.unique(ids)
    return np.mean([np.mean(loss[ids==person]) for person in people]),np.mean([
        np.mean(brier[ids==person]) for person in people])


def verify_derivative(panel,beta,label,aux,row,base,tol):
    x=panel.x[label]
    p=row_prob(x,panel.available[label],beta)[-1]
    means=(p[:,:,None]*x).sum(axis=1)
    flat=x.reshape(-1,x.shape[2])
    hessian=(flat.T@(p.reshape(-1,1)*flat)-means.T@means)/len(x)
    delta=original_score(panel,beta,label,True)-original_score(panel,beta,aux,True)
    speed=np.linalg.solve(hessian,delta)
    test=panel.test
    x_test=panel.x[test]
    p=row_prob(x_test,panel.available[test],beta)[-1]
    y=panel.y[test]
    seen=(p[:,:,None]*x_test).sum(axis=1)-x_test[np.arange(len(y)),y]
    ids=panel.ids[test]
    ev_grad=np.mean([np.mean(seen[ids==person],axis=0) for person in np.unique(ids)],axis=0)
    slope=float(np.dot(ev_grad,speed))
    assert abs(slope-row['slope'])<tol
    beta_small=np.asarray(row['beta'])
    power=row['small_power']
    g=original_score(panel,beta_small,label)+power*(
        original_score(panel,beta_small,aux,True)-original_score(panel,beta_small,label,True))
    assert np.max(abs(g))<tol
    chg=(check_metrics(panel,beta_small)[0]-base['loss'])/power
    assert abs(chg-row['secant'])<tol
    return abs(chg-slope)


def compare(group, method,ref,tie):
    pairs=[(row[method],row[ref]) for row in group if row[method]['loss'] is not None and row[ref]['loss'] is not None]
    if not pairs:
        return dict(pairs=0)
    diff=np.array([a['loss']-b['loss'] for a,b in pairs])
    sq=np.array([a['brier']-b['brier'] for a,b in pairs])
    return dict(pairs=len(pairs),lower_logloss=int((diff<-tie).sum()),higher_logloss=int((diff>tie).sum()),
                tied_logloss=int((abs(diff)<=tie).sum()),paired_mean_logloss_difference=float(diff.mean()),
                paired_median_logloss_difference=float(np.median(diff)),
                paired_mean_Brier_difference=float(sq.mean()),lower_Brier=int((sq<-tie).sum()),
                higher_Brier=int((sq>tie).sum()),tied_Brier=int((abs(sq)<=tie).sum()))


def summarize(links,derivatives,tie):
    cols=('safe_half','safe_edge','fixed_half','fixed_unit','labelled_MLE','labelled_ridge',
          'all_human_MLE','all_human_ridge','frozen_teacher')
    names=('RADIANT','Swissmetro','combined')
    output=[]
    for site,budget in [(a,b) for a in names[:2] for b in (5,10)]+[(a,None) for a in names]:
        pairs=[row for key,row in links.items() if (site in (key[0],'combined') and budget in (key[2],None))]
        safe=[row for row in pairs if row['safe_half']['state']=='native']
        rescued=[row for row in safe if row['fixed_unit'].get('reason')=='noninterior_archive_support']
        methods={method:dict(counts=dict(Counter(r[method]['state'] for r in pairs)),
                            mean_logloss=float(np.mean([r[method]['loss'] for r in pairs if r[method]['loss'] is not None])),
                            mean_Brier=float(np.mean([r[method]['brier'] for r in pairs if r[method]['brier'] is not None])),
                            positive_cost=sum(r[method]['cost_positive'] is True for r in pairs))
                 for method in cols}
        all_pairs={(a+'__vs__'+b):compare(pairs,a,b,tie)
                   for a in ('safe_half','safe_edge') for b in ('labelled_ridge','fixed_half','fixed_unit',
                      'all_human_MLE','all_human_ridge','frozen_teacher')}
        both_lambda={a:compare(safe,a,'labelled_MLE',tie) for a in SAFETY}
        rescued_results={b:compare(rescued,'safe_half',b,tie)
                         for b in ('labelled_MLE','labelled_ridge','all_human_MLE','frozen_teacher')}
        subset=[(v,links[key]) for key,v in derivatives.items()
                if site in (key[0],'combined') and budget in (key[2],None)]
        slope=np.array([a['slope'] for a,b in subset])
        small=np.array([a['secant'] for a,b in subset])
        half=np.array([b['safe_half']['loss']-b['labelled_MLE']['loss'] for a,b in subset])
        output.append(dict(dataset=site,label_people=budget,pairs=len(pairs),methods=methods,
                           all_policy_comparisons=all_pairs,common_labelled_MLE=both_lambda,
                           rescued_at_unit=rescued_results,
                           derivative=dict(pairs=len(slope),negative_at_zero=int(sum(slope<0)),
                                           lower_at_half_cap=int(sum(half<-tie)),
                                           sign_agrees_with_small_secant=int(sum(np.sign(small)==np.sign(slope))),
                                           sign_agrees_with_half_cap_change=int(sum(np.sign(half)==np.sign(slope))),
                                           median_slope=float(np.median(slope)))))
    return output


def main(args):
    plan,archive,caps,states,panels=sources(args)
    recorded=read(args.results/'fits.json')
    slopes=read(args.results/'derivatives.json')
    record_key=lambda row:(row['dataset'],row['allocation'],row['label_people'],row['draw'])
    keyed={record_key(row)+(row['method'],):row for row in recorded}
    derivatives={record_key(row):row for row in slopes}
    assert len(keyed)==len(recorded)==3552 and len(derivatives)==len(slopes)==656
    tolerance=plan['tolerance']['score']
    eps=plan['tolerance']['proof_cap']
    pairs={}
    max_score=max_loss=max_secant=0
    numeric_fallback=0
    for panel in panels:
        for j in range(archive['label_allocations']):
            for budget in archive['label_budgets']:
                label,aux_draws=sets(panel,archive,j,budget)
                key=(panel.name,j,budget)
                group=key+(-1,)
                ordinary_support=states[key]['ordinary']
                k_caps=caps[key]
                baselines={name:keyed[group+(name,)] for name in (
                    'labelled_ridge','labelled_MLE','all_human_MLE','all_human_ridge','frozen_teacher')}
                for m,row in baselines.items():
                    if m=='frozen_teacher':
                        assert row['state']=='frozen_teacher' and row['beta'] is None
                    elif m=='labelled_MLE' and ordinary_support!='interior':
                        assert row['state']=='support_unavailable' and row['loss'] is None and row['beta'] is None
                        continue
                    else:
                        assert row['state']=='native'
                        b=np.asarray(row['beta'])
                        mask=panel.train|label if m.startswith('all_human') else label
                        ridge=.01 if m.endswith('ridge') else 0
                        rec=float(max(abs(original_score(panel,b,mask)+ridge*b)))
                        assert rec<tolerance and abs(rec-row['score'])<1e-10
                        assert bool(b[-1]>0)==row['cost_positive']
                        max_score=max(max_score,rec)
                    expected=check_metrics(panel,None if m=='frozen_teacher' else np.asarray(row['beta']))
                    error=max(abs(v-row[k]) for v,k in zip(expected,('loss','brier'),strict=True))
                    assert error<1e-10
                    max_loss=max(max_loss,error)
                if k_caps>eps:
                    assert ordinary_support=='interior'
                for draw,aux in enumerate(aux_draws):
                    obs=key+(draw,)
                    methods={m:keyed[obs+(m,)] for m in SAFETY+FIXED}
                    for m,row in methods.items():
                        if m in SAFETY:
                            power=k_caps/(2 if m=='safe_half' else 1)
                            eligibility=k_caps>eps
                        else:
                            power=.5 if m=='fixed_half' else 1.
                            eligibility=states[key]['rows'][draw,power][0]=='interior'
                        assert row['power']==power
                        b=np.asarray(row['beta'])
                        if row['state']=='labelled_ridge_fallback':
                            if eligibility:
                                assert row['reason']=='fit_failed_score'
                                numeric_fallback+=1
                            assert all(row[k]==baselines['labelled_ridge'][k] for k in ('beta','loss','brier','cost_positive'))
                        else:
                            assert eligibility and row['state']=='native'
                            score=original_score(panel,b,label)+power*(
                                original_score(panel,b,aux,True)-original_score(panel,b,label,True))
                            rec=float(max(abs(score)))
                            assert rec<tolerance and abs(rec-row['score'])<1e-10
                            assert bool(b[-1]>0)==row['cost_positive']
                            max_score=max(max_score,rec)
                        expected=check_metrics(panel,b)
                        error=max(abs(v-row[k]) for v,k in zip(expected,('loss','brier'),strict=True))
                        assert error<1e-10
                        max_loss=max(max_loss,error)
                    pairs[obs]=baselines|methods
                    if obs in derivatives:
                        assert baselines['labelled_MLE']['state']=='native' and k_caps>eps and derivatives[obs]['small_power']==.01*k_caps
                        err=verify_derivative(panel,np.asarray(baselines['labelled_MLE']['beta']),
                             label,aux,derivatives[obs],baselines['labelled_MLE'],tolerance)
                        max_secant=max(max_secant,err)
    assert len(pairs)==768 and numeric_fallback==0
    cells=summarize(pairs,derivatives,plan['tolerance']['tie'])
    receipt=dict(status='passed',pair_count=len(pairs),methods=len(plan['method']),fit_records=len(keyed),
                 zero_power_derivatives=len(derivatives),numeric_fallbacks=numeric_fallback,
                 largest_reconstructed_score=max_score,largest_forecast_reconciliation_error=max_loss,
                 largest_small_power_secant_difference=max_secant,
                 results=cells,hashes={n:digest(args.results/n) for n in ('fits.json','derivatives.json')},
                 verifier_sha256=digest(Path(__file__)),plan_sha256=digest(ROOT/'plan.json'))
    (args.results/'summary.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps(cells[-1]['all_policy_comparisons']['safe_half__vs__all_human_MLE']))
    print(json.dumps(cells[-1]['common_labelled_MLE']['safe_half']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('food-data','food-archive','swiss-data','swiss-archive','reference','certificate','results'):
        p.add_argument('--'+key,type=Path,required=True)
    with threadpool_limits(limits=1):
        main(p.parse_args())
