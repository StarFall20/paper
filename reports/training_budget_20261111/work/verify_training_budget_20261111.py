"""Independent reconciliation and separating-margin checks for the budget audit."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.special import expit, log_expit
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'work'))
from training_budget_diagnostic_20261111 import ridge
from power_tuned_choice_audit import WeightedLogit, objective, solve


def main():
    out = ROOT/'outputs/training_budget_diagnostic_20261111'
    plan = json.loads((ROOT/'work/training_budget_plan_20261111.json').read_text())
    status = json.loads((out/'training_budget_status.json').read_text())
    raw = pd.read_csv(out/'training_budget_trials.csv')
    assert status['state'] == 'complete' and len(raw) == 4800
    assert not raw.duplicated(['condition','trial','method']).any()
    assert set(raw.groupby(['condition','method']).size()) == {100}
    assert raw.groupby(['condition','trial']).size().eq(6).all()
    assert (raw.training_labels+raw.correction_labels).eq(raw.total_labels).all()
    assert raw.auxiliary_rows.eq(4*raw.correction_labels).all()
    returned = raw.status.eq('finite_root')
    assert returned.equals(raw.excess_log_loss.notna())
    assert raw.loc[returned, 'score_max'].max() < 1e-7
    assert raw.loc[~returned, 'coefficient_error'].isna().all()
    assert not raw.status.str.contains('unresolved').any()
    assert raw[raw.method.eq('all_labels_ridge_0.1')].status.eq('finite_root').all()
    pivot = raw.assign(finite=returned).pivot(index=['condition','trial'],columns='method',values='finite')
    assert (~pivot['corrected_power_1.0'] | pivot['corrected_power_0.5']).all()
    assert (~pivot['corrected_power_0.5'] | pivot['corrected_power_0.1']).all()
    summary = pd.read_csv(out/'training_budget_summary.csv')
    paired = pd.read_csv(out/'training_budget_paired.csv')
    max_summary = max_pair = max_mcse = 0.
    for _, row in summary.iterrows():
        part = raw[(raw.total_labels==row.total_labels)&(raw.slope_dimension==row.slope_dimension)&
                   (raw.training_labels==row.training_labels)&(raw.method==row.method)]
        assert len(part) == row.draws == 100
        assert part.excess_log_loss.notna().sum() == row.verified_fits
        if part.excess_log_loss.notna().any():
            max_summary = max(max_summary,abs(part.excess_log_loss.mean()-row.excess_log_loss))
    for _, row in paired.iterrows():
        part = raw[(raw.total_labels==row.total_labels)&(raw.slope_dimension==row.slope_dimension)&
                   (raw.training_labels==row.training_labels)]
        table = part.pivot(index='trial',columns='method',values='excess_log_loss')
        diff = (table[row.method]-table[row.baseline]).dropna().to_numpy()
        assert len(diff) == row.shared_verified_fits
        if len(diff): max_pair = max(max_pair,abs(np.mean(diff)-row.excess_loss_difference))
        if len(diff)>1: max_mcse = max(max_mcse,abs(np.std(diff,ddof=1)/np.sqrt(len(diff))-row.Monte_Carlo_SE))
    assert max(max_summary,max_pair,max_mcse) < 1e-10

    # Resolve every ordinary-logit boundary classification with a strictly
    # positive separating margin in an independently written L1-bounded LP.
    separators = []
    specs = [(b,round(b*r),round(b*f)) for b in plan['total_human_labels']
             for r in plan['slope_dimensions_per_total_label']
             for f in plan['independent_training_label_fractions']]
    risk_error = coef_error = 0.; regenerated = 0
    for condition,(budget,d,train) in enumerate(specs):
        n_aux = 4*(budget-train)
        truth = np.r_[0.,1.5,np.zeros(d-1)]
        for trial in range(100):
            part = raw[(raw.condition==condition)&(raw.trial==trial)]
            boundary = part[part.method.eq('all_labels_mle')].iloc[0].status != 'finite_root'
            if not boundary and trial != 0: continue
            rng = np.random.default_rng(np.random.SeedSequence([plan['seed'],condition,trial]))
            features = rng.normal(size=(budget+n_aux,d))
            x = np.c_[np.ones(len(features)),features]
            y = (rng.random(budget)<expit(x[:budget]@truth)).astype(float)
            if boundary:
                z = (2*y-1)[:,None]*x[:budget];p=x.shape[1]
                A = np.vstack([np.c_[-z,z,np.ones(budget)],np.r_[np.ones(2*p),0.]])
                lp = linprog(np.r_[np.zeros(2*p),-1.],A_ub=A,
                             b_ub=np.r_[np.zeros(budget),1.],
                             bounds=[(0,None)]*(2*p)+[(None,None)],method='highs')
                assert lp.success and lp.x[-1] > 1e-9,(condition,trial,lp.message)
                direction = lp.x[:p]-lp.x[p:2*p]
                margin = float(np.min(z@direction))
                assert abs(margin-lp.x[-1])<1e-7
                separators.append({'condition':condition,'trial':trial,'strict_separating_margin':margin})
            if trial != 0: continue
            m = budget-train
            btrain,_ = ridge(x[:train],y[:train]);ball,_ = ridge(x[:budget],y)
            ax = x[train:];ay=np.r_[y[train:],np.zeros(n_aux)];g=expit(ax@btrain)
            ordinary = WeightedLogit(x[:budget],np.full(budget,1/budget),x[:budget].T@y/budget)
            bmle,cmle = solve(ordinary,ball)
            cases=[('training_ridge_only',btrain),('all_labels_ridge_0.1',ball),('all_labels_mle',bmle)]
            for power in [.1,.5,1.]:
                beta,check=solve(objective(ax,ay,g,m,power),btrain)
                assert check['status']==part[part.method.eq(f'corrected_power_{power:.1f}')].iloc[0].status
                cases.append((f'corrected_power_{power:.1f}',beta))
            test=np.c_[np.ones(4096),rng.normal(size=(4096,d))]
            mu=expit(test@truth)
            entropy=-mu*log_expit(test@truth)-(1-mu)*log_expit(-test@truth)
            for method,beta in cases:
                row=part[part.method.eq(method)].iloc[0]
                assert (beta is not None)==(row.status=='finite_root')
                if beta is None: continue
                eta=test@beta
                kl=np.mean(-mu*log_expit(eta)-(1-mu)*log_expit(-eta)-entropy)
                risk_error=max(risk_error,abs(kl-row.excess_log_loss))
                coef_error=max(coef_error,abs(np.sum((beta-truth)**2)-row.coefficient_error))
                regenerated+=1
    assert len(separators)==397 and risk_error<1e-9 and coef_error<1e-6
    pd.DataFrame(separators).to_csv(out/'training_budget_mle_separation.csv',index=False)
    checks={'all_planned_rows_retained':4800,'planned_independent_model_draws':800,
            'all_summary_cells_reconciled':len(summary),'paired_comparisons_reconciled':len(paired),
            'maximum_summary_error':max_summary,'maximum_paired_error':max_pair,'maximum_MCSE_error':max_mcse,
            'nested_positive_power_fit_events_verified':800,
            'ordinary_boundary_cases_separately_certified_strictly_separated':len(separators),
            'minimum_ordinary_strict_separating_margin':min(z['strict_separating_margin'] for z in separators),
            'regenerated_designs':8,'regenerated_verified_fits':regenerated,
            'independent_log_probability_risk_formula_error':risk_error,
            'regenerated_coefficient_error_maximum':coef_error,
            'minimum_corrected_exterior_gap':float(raw.loc[raw.status.eq('outside_mean_support'),'separation_gap'].min()),
            'manuscript_package_20261110_modified':False,
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'training_budget_verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1): main()
