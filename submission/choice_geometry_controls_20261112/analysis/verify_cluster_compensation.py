"""Independent moment enumeration and original-objective support crosschecks."""
from pathlib import Path
import argparse,itertools,json,hashlib
import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
from power_tuned_choice_audit import objective,support


def write(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def enumerate_moments(cov,omega,epsilon):
    total=np.zeros((10,10));mass=0.
    for yt in itertools.product([0,1],repeat=3):
        y=np.array(yt,float);py=(1-omega)/8+omega/2*(len(set(yt))==1)
        for zt in itertools.product([0,1],repeat=3):
            z=np.array(zt,float);k=int(np.count_nonzero(y!=z));weight=py*epsilon**k*(1-epsilon)**(3-k)
            human=.5-y;pred=.5-(epsilon+(1-2*epsilon)*z)
            block=np.zeros((10,10))
            for left,res1 in enumerate([human,pred]):
                for right,res2 in enumerate([human,pred]):
                    block[5*left,5*right]=res1.sum()*res2.sum()/9
                    feature=float(res1@cov@res2/9)
                    for j in range(1,5):block[5*left+j,5*right+j]=feature
            total+=weight*block;mass+=weight
    assert abs(mass-1)<1e-12
    return total


def reconstruct(row,plan,derived):
    m=int(row.persons);max_q=max(max(v) for v in derived['noise_dimensions'][str(m)].values())
    generator=np.random.default_rng(np.random.SeedSequence([derived['seed'],m,int(row.replication)]))
    gaussian=generator.standard_normal((2*m,3,max_q+1))
    outcomes=generator.integers(0,2,size=(2*m,3)).astype(float)
    same=generator.uniform(size=2*m)<derived['outcome_dependence']
    common=generator.integers(0,2,size=2*m)
    outcomes[same,:]=np.repeat(common[same,None],3,axis=1)
    flips=generator.uniform(size=(2*m,3))<derived['epsilon']
    predictions=derived['epsilon']+(1-2*derived['epsilon'])*np.not_equal(outcomes,flips)
    covariance=next(np.array(x['covariance']) for x in plan['models'] if x['name']==row.model)
    root=np.asarray(sqrtm(covariance),float)
    features=np.matmul(root,gaussian).reshape(6*m,max_q+1)
    x=np.column_stack([np.ones(6*m),features[:,:int(row.noise_slopes)+1]])
    return x,outcomes.ravel(),predictions.ravel()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    plan=json.loads(a.plan.read_text());derived=json.loads((a.root/'phase/phase_plan.json').read_text())
    status=json.loads((a.root/'phase/run_status.json').read_text());assert status['complete']
    trials=pd.read_csv(a.root/'phase/phase_trials.csv');assert len(trials)==24000
    columns=['persons','model','power','noise_slopes','status']
    selected=trials.groupby(columns,sort=True).head(4)
    selected=pd.concat([selected,trials[trials.status=='unknown']]).drop_duplicates(['persons','replication','model','power','noise_slopes'])
    rows=[];witnesses=[]
    for i,row in enumerate(selected.itertuples(index=False)):
        x,y,g=reconstruct(row,plan,derived);loss=objective(x,y,g,3*int(row.persons),row.power);result=support(loss)
        expected='feasible' if result['status']=='interior' else 'outside' if result['status']=='outside_mean_support' else 'unknown'
        rows.append(dict(persons=row.persons,replication=row.replication,model=row.model,power=row.power,noise_slopes=row.noise_slopes,original_status=row.status,independent_status=result['status'],status_agrees=expected==row.status,interior_margin=result.get('interior_margin'),separation_gap=result.get('separation_gap')))
        if 'direction' in result:
            direction=np.array(result['direction']);cost=float(loss.weights@np.maximum(loss.x@direction,0)-loss.moment@direction)
            assert cost<-1e-8
            witnesses.append(dict(row_index=len(rows)-1,recession_cost=cost,direction=direction.tolist()))
        assert expected==row.status,(rows[-1],result)
        if (i+1)%40==0:print('independent support replays',i+1,'of',len(selected),flush=True)
    pd.DataFrame(rows).to_csv(a.out/'independent_support_replays.csv',index=False);write(a.out/'exterior_directions.json',witnesses)
    stored=json.loads((a.root/'population/score_population_moments.json').read_text());checks=[];maximum=0.
    for dependence in plan['outcome_dependences']:
        values={}
        for model in plan['models']:
            covariance=np.array(model['covariance']);matrix=enumerate_moments(covariance,dependence,plan['epsilon']);values[model['name']]=matrix
            old=next(x['moments'] for x in stored if x['model']==model['name'] and x['outcome_dependence']==dependence)
            for key,indices in [('intercept',[0,5]),('Gaussian_feature',[1,6])]:maximum=max(maximum,float(abs(matrix[np.ix_(indices,indices)]-np.array(old[key])).max()))
        matched=float(abs(values['equal_correlation']-values['concentrated_correlation']).max());assert matched<1e-14
        checks.append(dict(outcome_dependence=dependence,complete_joint_score_matrix_dimension=10,covariance_difference=matched))
    assert maximum<1e-14
    write(a.out/'verification.json',dict(date='2026-11-12',all_planned_phase_records_retained=len(trials),independent_support_replays=len(rows),verified_exterior_directions=len(witnesses),finite_interior_replays=sum(r['independent_status']=='interior' for r in rows),unknown_phase_records=int((trials.status=='unknown').sum()),independent_status_mismatches=0,all_discrete_states_enumerated=True,maximum_formula_vs_enumeration_error=maximum,matched_covariance_checks=checks,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),independent_theorem_priority_or_publication_certified=False))


if __name__=='__main__':main()
