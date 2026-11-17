"""Exact finite-design identities; standard library only, no generated cohorts."""
from fractions import Fraction as F
from pathlib import Path
import argparse,hashlib,json
P=Path(__file__).resolve().parent
def mean(xs):return sum(xs,F(0))/len(xs)
def covariance(rows):
    av=[mean([r[j] for r in rows]) for j in range(2)]
    return [[sum((r[j]-av[j])*(r[k]-av[k]) for r in rows)/(len(rows)-1) for k in range(2)] for j in range(2)]
def encode(v):
    if isinstance(v,F):return str(v)
    if isinstance(v,list):return [encode(x) for x in v]
    if isinstance(v,dict):return {k:encode(x) for k,x in v.items()}
    return v
def main(out):
    raw=json.loads((P/'design.json').read_text());a=F(raw['base_power'])
    L=[(F(r['x']),F(r['y']),F(r['g'])) for r in raw['labelled']]
    U=[(F(r['x']),F(r['g'])) for r in raw['auxiliary']]
    M=lambda rows:[[mean([(F(1),r[0])[j]*(F(1),r[0])[k] for r in rows]) for k in range(2)] for j in range(2)]
    assert M(L)==M(U)==[[1,0],[0,1]]
    records=[]
    for lam in map(F,raw['powers']):
        gl=[F(1,2)+a/lam*(g-F(1,2)) for x,y,g in L];gu=[F(1,2)+a/lam*(g-F(1,2)) for x,g in U]
        assert all(0<g<1 for g in gl+gu)
        sL=[[(1,x)[j]*(F(1,2)-y+lam*(g-F(1,2))) for j in range(2)] for (x,y,_),g in zip(L,gl)]
        sU=[[(1,x)[j]*lam*(F(1,2)-g) for j in range(2)] for (x,_),g in zip(U,gu)]
        grad=[mean([r[j] for r in sL])+mean([r[j] for r in sU]) for j in range(2)]
        b=[mean([(1,x)[j]*(y-lam*g) for (x,y,_),g in zip(L,gl)])+lam*mean([(1,x)[j]*g for (x,_),g in zip(U,gu)]) for j in range(2)]
        H=[[(1-lam)*M(L)[j][k]/4+lam*M(U)[j][k]/4 for k in range(2)] for j in range(2)]
        cL,cU=covariance(sL),covariance(sU)
        J=[[cL[j][k]/len(L)+cU[j][k]/len(U) for k in range(2)] for j in range(2)]
        sandwich=[[16*v for v in row] for row in J]
        upper=(1-lam)*mean([max(x,0) for x,y,g in L])+lam*mean([max(x,0) for x,g in U])
        gap=upper-b[1]
        assert b==[F(1,2),F(29,80)] and grad==[0,F(-29,80)] and H==[[F(1,4),0],[0,F(1,4)]]
        assert sandwich==[[F(4057,2800),0],[0,F(3,112)]]
        assert gap==F(11,80)-lam/4
        records.append(dict(power=lam,prediction_labelled=gl,prediction_auxiliary=gu,labelled_row_scores=sL,auxiliary_row_scores=sU,gradient=grad,corrected_moment=b,Hessian=H,score_middle=J,reference_sandwich=sandwich,one_step_from_reference=[-4*v for v in grad],slope_recession=gap,attainment='finite_unique' if gap>0 else 'boundary_no_finite_root' if gap==0 else 'unbounded_below'))
    for key in ['labelled_row_scores','auxiliary_row_scores','gradient','corrected_moment','Hessian','score_middle','reference_sandwich','one_step_from_reference']:
        assert all(r[key]==records[0][key] for r in records),key
    fourth=lambda lam:-((1-lam)*mean([x**4 for x,y,g in L])+lam*mean([x**4 for x,g in U]))/8
    fourth_difference=fourth(F(1))-fourth(a);assert fourth_difference==F(-3,16)
    data=dict(state='passed',reference_value='log(2)',third_derivative_tensor='identically zero at reference for every power',moments_labelled=M(L),moments_auxiliary=M(U),exact_critical_power=F(11,20),records=records,base_vs_unit_fourth_slope_derivative_difference=fourth_difference,new_deterministic_designs=1,new_independent_cohorts=0,new_human_or_AI_answers=0,variance_optimal_selection_claimed=False,valid_confidence_interval_claimed=False,global_originality_claimed=False,design_sha256=hashlib.sha256((P/'design.json').read_bytes()).hexdigest())
    out.mkdir(parents=True,exist_ok=True);(out/'exact_information.json').write_text(json.dumps(encode(data),indent=2)+'\n')
    print(json.dumps(dict(state='passed',powers=len(records),identical_reference_Hessian_and_full_sandwich=True,exact_critical_power='11/20',base_state=records[0]['attainment'],unit_state=records[-1]['attainment'])))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=P/'results');main(p.parse_args().out)
