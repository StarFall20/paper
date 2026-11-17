"""Reconstruct exact weights and interval bounds without importing the evaluator."""
from pathlib import Path
from fractions import Fraction as Q
import argparse,itertools,json,math,hashlib
from flint import arb,ctx
ctx.prec=160

def b(q):
    return arb(str(q.numerator)+'/'+str(q.denominator)) if isinstance(q,Q) else arb(q)
def upper(x):return math.nextafter(float(x.upper()),math.inf)
def lower(x):return math.nextafter(float(x.lower()),-math.inf)
def interval(x):
    lo=x.lower();hi=x.upper();lm,le=lo.man_exp();hm,he=hi.man_exp()
    return dict(lower=lower(x),upper=upper(x),exact_dyadic_lower=[str(lm),int(le)],exact_dyadic_upper=[str(hm),int(he)])

def ordinary(a,x):
    c=b('3/2')-a-x
    off=[(x*c-a)/((1-x*x)*(1-c*c)).sqrt(),(a*c-x)/((1-a*a)*(1-c*c)).sqrt(),(a*x-c)/((1-a*a)*(1-x*x)).sqrt()]
    return b('3/2')+(sum(y.asin() for y in off)-a.asin()-x.asin()-c.asin())/(2*arb.pi())

def check_model(m):
    a=b(m['first_correlation']);lo,hi=[b(x) for x in m['root_interval_fraction']]
    fl=ordinary(a,lo)-1;fh=ordinary(a,hi)-1
    assert (fl>0 and fh<0) or (fl<0 and fh>0)
    x=lo.union(hi);c=b('3/2')-a-x
    r=[[b(1),a,x],[a,b(1),c],[x,c,b(1)]]
    l22=(1-a*a).sqrt();l32=(c-a*x)/l22;l33=(1-x*x-l32*l32).sqrt()
    assert l22>0 and l33>0
    l=[[b(1),b(0),b(0)],[a,l22,b(0)],[x,l32,l33]]
    rf=[[b(f) for f in row] for row in m['covariance']]
    minor2=rf[0][0]*rf[1][1]-rf[0][1]*rf[0][1]
    det3=rf[0][0]*rf[1][1]*rf[2][2]+2*rf[0][1]*rf[0][2]*rf[1][2]-rf[0][0]*rf[1][2]**2-rf[1][1]*rf[0][2]**2-rf[2][2]*rf[0][1]**2
    assert rf[0][0]>0 and minor2>0 and det3>0
    re=sum(abs(v-b(f)).upper()**2 for row,fr in zip(r,m['covariance']) for v,f in zip(row,fr)).sqrt()
    le=sum(abs(v-b(f)).upper()**2 for row,fr in zip(l,m['cholesky']) for v,f in zip(row,fr)).sqrt()
    assert all(abs(v)<=1 for row in r for v in row)
    assert all(abs(f)<=1 for name in ['covariance','cholesky'] for row in m[name] for f in row)
    return dict(covariance_error=re,cholesky_error=le,ordinary=ordinary(a,x),root_endpoint_residuals=[interval(fl),interval(fh)],float_covariance_positive_principal_minors=[interval(rf[0][0]),interval(minor2),interval(det3)])

def quantizer(q):
    edges=[float.fromhex(x) for x in q['boundaries_hex']]
    assert edges[0]==-math.inf and edges[-1]==math.inf and len(edges)==q['bins']+1
    def phi(x):return b(0) if math.isinf(x) else (-(b(x)*b(x))/2).exp()/(2*arb.pi()).sqrt()
    def cdf(x):
        if x==-math.inf:return b(0)
        if x==math.inf:return b(1)
        return (1+(b(x)/b(2).sqrt()).erf())/2
    ps=[];mus=[]
    for l,u in zip(edges[:-1],edges[1:]):
        p=cdf(u)-cdf(l);assert p>0
        ps.append(p);mus.append((phi(l)-phi(u))/p)
    variance=1-sum(p*m*m for p,m in zip(ps,mus));assert variance>0
    dp=sum(abs(p-b(f)).upper() for p,f in zip(ps,q['probabilities']))
    dm=max(abs(m-b(f)).upper() for m,f in zip(mus,q['nodes']))
    assert dp<b('1e-12') and dm<b('1e-12') and max(abs(x) for x in q['nodes'])<8
    return variance,dp,dm

def exact_states():
    total={};second=[[Q(0) for _ in range(3)] for _ in range(3)]
    for y in itertools.product([0,1],repeat=3):
        py=Q(1,32)+(Q(3,8) if len(set(y))==1 else 0)
        for z in itertools.product([0,1],repeat=3):
            e=sum(a!=v for a,v in zip(y,z));p=py*Q(1,4)**e*Q(3,4)**(3-e)
            total[z]=total.get(z,Q(0))+p;r=[Q(1,4)+Q(v,2)-u for u,v in zip(y,z)]
            for j in range(3):
                for k in range(3):second[j][k]+=p*r[j]*r[k]
    assert sum(total.values())==1
    assert all(second[j][k]==(Q(3,16) if j==k else Q(27,256)) for j in range(3) for k in range(3))
    coefficient=3*Q(3,16)+3*Q(27,256);assert coefficient==Q(225,256)
    weights={z:total[z]+total[tuple(1-v for v in z)] for z in total if z[0]==0}
    assert sorted(weights.values())==[Q(13,64)]*3+[Q(25,64)]
    return weights,coefficient

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);ap.add_argument('--bins',type=int,default=128);a=ap.parse_args();r=a.root;d=json.loads((r/f'results/inputs{a.bins}.json').read_text())
    checks={k:check_model(v) for k,v in d['models'].items()};var,dp,dm=quantizer(d['quantizer']);weights,c=exact_states();n=a.bins**3
    unit=b(1)/(2**53);gamma=lambda k:k*unit/(1-k*unit)
    node_guard=b(1)/(2**30)
    assert gamma(1024)*1024<node_guard
    aggregate_guard=1024*gamma(8*n+256)
    mass_guard=256*3*dp*(1+dp)**2
    # ||C u||<=sqrt(trace(Sigma))*||u||<=9t/4; t<=42/64.
    centroid_guard=6*b(3).sqrt()*dm
    values={};details=[];nodes_checked=0;maxgap=0.
    for model,k in [('A',41),('B',40),('B',41),('B',42)]:
        raw=json.loads((r/f'results/node_{model}_{k}.json').read_text());assert raw['normal_bins']==a.bins and raw['scale_numerator']==k
        val=b(0)
        for state in raw['states']:
            z=tuple(state['prediction_bits']);p=weights[z]
            assert p==Q(state['probability_numerator'],state['probability_denominator'])
            assert state['node_count']==n and state['maximum_feasibility_violation']==0
            assert state['maximum_original_quadratic_dual_gap']<node_guard and state['maximum_primal_recalculation_disagreement']<node_guard
            assert state['maximum_normal_coordinate']<8 and state['maximum_raw_box_coordinate']<b('1/2') and state['maximum_raw_quadratic_residual']<32
            val+=b(p)*b(state['node_primal_sum']);nodes_checked+=n
            maxgap=max(maxgap,state['maximum_original_quadratic_dual_gap'])
        cm=checks[model]
        # Conservative uniform bounds: ||h||<14, ||u||<1.
        parameter_guard=28*cm['cholesky_error']+cm['covariance_error']
        guard=4*node_guard+aggregate_guard+mass_guard+centroid_guard+parameter_guard
        gap_guard=b(max(s['maximum_original_quadratic_dual_gap'] for s in raw['states']))
        t=b(Q(k,64));low=(val-guard-b(c)*t*t).lower();high=(val+guard+gap_guard+3*var-b(c)*t*t).upper()
        values[model+str(k)]=(low,high)
        details.append(dict(model=model,scale=str(Q(k,64)),objective_lower=interval(low),objective_upper=interval(high),guard_without_quantization=interval(guard),quantization_remainder=interval(3*var),parameter_guard=interval(parameter_guard),node_count_all_four_states=4*n))
    h=b(Q(1,64));dl=(values['B42'][0]-values['B41'][1])/h;du=(values['B41'][1]-values['B40'][0])/h
    derivative_abs=max(abs(dl).upper(),abs(du).upper())
    a_lower=values['A41'][0];b_upper=values['B41'][1]+derivative_abs**2/(4*b(c))
    witness=b(Q(719,1000));strict=bool(a_lower>b_upper);witness_separates=bool(a_lower>witness and b_upper<witness)
    out=dict(date='2026-11-18',state='separated' if strict else 'inconclusive',normal_bins=a.bins,product_nodes_per_state=n,original_quadratic_node_checks=nodes_checked,quantization_remainder=interval(3*var),node_arithmetic_guard=interval(node_guard),aggregation_guard=interval(aggregate_guard),maximum_observed_original_quadratic_dual_gap=maxgap,objective_enclosures=details,model_A_boundary_lower=interval(a_lower),model_B_boundary_upper=interval(b_upper),strict_gap_lower=interval(a_lower-b_upper),model_B_scale_derivative_bounds=[interval(dl),interval(du)],model_B_optimization_remainder=interval(derivative_abs**2/(4*b(c))),dimension_ratio_witness='719/1000',witness_separates=witness_separates,common_ordinary_boundary='9/8',labelled_quadratic_coefficient=str(c),exact_auxiliary_antipodal_weights={str(k):str(v) for k,v in weights.items()},ordinary_root_checks={k:dict(boundary=interval(v['ordinary']),endpoint_residuals=v['root_endpoint_residuals']) for k,v in checks.items()},scope='Deterministic enclosure conditional on stated IEEE-754 arithmetic model and source-audited operation bounds. Complete tail cells and all scale values are covered. No global originality, external theorem review, empirical prevalence or submission-readiness certificate.',floating_arithmetic_assumptions='Round-to-nearest IEEE-754 binary64 basic operations, standard bounded-length dot products, no overflow. Face inverses only propose feasible points; their errors do not enter the optimality claim. Original-quadratic primal/dual checks and explicit gamma bounds enclose evaluation and accumulation.',new_MC_or_fitted_models=0)
    names=['plan.json','enclosure_method.md','proof.md','floating_error_audit.md','requirements.txt','prepare_inputs.py','evaluate_nodes.py','verify_enclosure.py',f'results/inputs{a.bins}.json']+[f'results/node_{m}_{k}.json' for m,k in [('A',41),('B',40),('B',41),('B',42)]]
    out['float_covariance_positive_principal_minors']={k:v['float_covariance_positive_principal_minors'] for k,v in checks.items()}
    out['sources_sha256']={n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in names}
    (r/'results/enclosure.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:out[k] for k in ['state','original_quadratic_node_checks','model_A_boundary_lower','model_B_boundary_upper','strict_gap_lower','witness_separates','dimension_ratio_witness']}))

if __name__=='__main__':main()
