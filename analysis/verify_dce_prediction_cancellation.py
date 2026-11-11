"""Exact balanced-menu cancellation on the existing human DCE design.

AI pretests supply frozen menu-level predicted fractions,never human labels.
Only aggregate numerical identity checks are written; raw responses stay local.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
import dh36_observational_analysis as original
from application_support_models import build
from menu_corrected_logit import MenuLoss,support,save


def main():
    p=argparse.ArgumentParser()
    for name in ['input-dir','instrument','classification','out']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    role=json.loads(a.classification.read_text());assert role['primary_records']['total']==300 and role['synthetic_records']['total']==60
    spec=json.loads(a.instrument.read_text());people,provenance,*_=original.read_exports(a.input_dir,spec)
    expected={'V1':'dbe667ecc7511b69fcb5c36147f986499dbbbf23ec1d7f638eb137cfbb080e13','V2':'40b3cc7daee37b5ad0e1060b77c12cdccc9c5dff9cff1a57a75e32b8d5b7ce01'}
    assert {r['version']:r['sha256'] for r in provenance}==expected
    forms,*_=build(spec,people);x=forms['linear'][...,:5]
    y=people[[f'q{j}' for j in range(1,9)]].to_numpy(int)-1
    ai=people.ai_label.to_numpy();version=people.version2.to_numpy()
    assert (~ai).sum()==300 and ai.sum()==60
    rng=np.random.default_rng(2026111153);records=[];max_value=max_score=max_hessian=max_root=0.
    for group in [0,1]:
        human=(~ai)&(version==group);synthetic=ai&(version==group)
        assert human.sum()==150 and synthetic.sum()==30
        xh=x[human];yh=y[human];ys=y[synthetic]
        assert np.max(abs(xh-xh[:1]))<1e-12
        template=xh[0]-xh[0].mean(axis=1,keepdims=True)
        g=np.array([(np.bincount(ys[:,task],minlength=3)+.5)/31.5 for task in range(8)])
        for nlabel in [20,50,100]:
          for trial in range(20):
            order=rng.permutation(150);L=order[:nlabel];U=order[nlabel:]
            menus=np.concatenate([xh[L],xh[U]]).reshape(150*8,3,5)
            labels=np.r_[yh[L].ravel(),np.zeros(len(U)*8,dtype=int)]
            prediction=np.tile(g,(150,1));m=nlabel*8
            fractions=np.array([np.bincount(yh[L,task],minlength=3)/nlabel for task in range(8)])
            grouped=MenuLoss.__new__(MenuLoss);grouped.features=template;grouped.weights=np.full(8,1/8)
            grouped.moment=np.einsum('tjp,tj->p',template,fractions)/8
            state=support(grouped)['status'];beta=None
            if state=='interior':
                opt=minimize(grouped.value,np.zeros(5),jac=True,method='BFGS',options={'gtol':1e-10,'maxiter':1800})
                beta=opt.x
                for _ in range(6):
                    val,gradient=grouped.value(beta)
                    if np.max(abs(gradient))<1e-9:break
                    direction=np.linalg.solve(grouped.hessian(beta),gradient);step=1.
                    while step>1e-8 and grouped.value(beta-step*direction)[0]>val+1e-12:step/=2
                    beta=beta-step*direction
                residual=float(np.max(abs(grouped.value(beta)[1])))
                if residual>1e-8:state='fit_unresolved';beta=None
            anchors=[np.zeros(5),np.linspace(-.2,.3,5),rng.normal(size=5)/4]
            local_value=local_score=local_hessian=0.
            for power in [0.,.1,.5,1.]:
                corrected=MenuLoss(menus,labels,prediction,m,power)
                for anchor in anchors:
                    hv,hg=grouped.value(anchor);cv,cg=corrected.value(anchor)
                    local_value=max(local_value,abs(hv-cv));local_score=max(local_score,float(np.max(abs(hg-cg))))
                    local_hessian=max(local_hessian,float(np.max(abs(grouped.hessian(anchor)-corrected.hessian(anchor)))))
                if beta is not None:max_root=max(max_root,float(np.max(abs(corrected.value(beta)[1]))))
            max_value=max(max_value,local_value);max_score=max(max_score,local_score);max_hessian=max(max_hessian,local_hessian)
            records.append(dict(version=group+1,labelled_people=nlabel,labelled_choices=m,auxiliary_people=150-nlabel,
                                trial=trial,status=state,maximum_objective_difference=local_value,
                                maximum_score_difference=local_score,maximum_Hessian_difference=local_hessian))
    assert max(max_value,max_score,max_hessian)<1e-10
    assert all(r['status']!='fit_unresolved' for r in records)
    save(a.out/'balanced_dce_cancellation_checks.json',dict(
        human_records=300,AI_pretest_records_excluded_from_human_labels=60,versions=2,tasks_per_person=8,alternatives=3,
        whole_person_allocations=len(records),powers_per_allocation=4,parameter_anchors_per_power=3,
        maximum_objective_identity_error=max_value,maximum_score_identity_error=max_score,
        maximum_Hessian_identity_error=max_hessian,maximum_reused_root_score=max_root,
        states={s:sum(r['status']==s for r in records) for s in sorted({r['status'] for r in records})},
        predictions='Task/version AI-pretest fractions with fixed Jeffreys0.5 smoothing; same frozen vector in labelled and auxiliary menus',
        interpretation='Exact finite objective cancellation when both subsets repeat the same balanced menus and predictions depend only on those menus',
        allocation_differences_are_computational_checks_not_new_observations=True,raw_hashes=expected,
        participant_rows_or_answers_written=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        new_method_or_priority_claim=False,Gaussian_phase_law_empirically_validated=False))
    print((a.out/'balanced_dce_cancellation_checks.json').read_text(),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
