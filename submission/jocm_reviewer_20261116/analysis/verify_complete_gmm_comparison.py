"""Independent direct-array check; numerical returns are not global certificates."""
from pathlib import Path
import argparse,csv,hashlib,json
from fractions import Fraction
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from scipy import __version__ as scipy_version

def main():
    p=argparse.ArgumentParser();p.add_argument('--certificate',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    exact=json.loads(a.certificate.read_text());number=lambda x:float(Fraction(x))
    xl=np.array([2.,4.]);xu=np.ones(2);yl=np.ones(2)
    gl=np.full(2,25/44);gu=np.full(2,.5)
    sl=xl*(.5-yl);hl=xl*(.5-gl);hu=xu*(.5-gu)
    centered=np.column_stack((sl-sl.mean(),-(hl-hl.mean())))
    sigma=centered.T@centered/2
    sigma[1,1]+=np.mean((hu-hu.mean())**2)
    sigma+=np.eye(2)/4;weight=np.linalg.inv(sigma)
    expected=np.array([[number(exact['W']['entry11']),number(exact['W']['entry12'])],[number(exact['W']['entry12']),number(exact['W']['entry22'])]])
    weight_error=float(np.max(abs(weight-expected)));assert weight_error<2e-14
    def values(beta):
        pl=expit(xl*beta);pu=expit(xu*beta)
        moment=np.array([np.mean(xl*(pl-yl)),np.mean(xu*(pu-gu))-np.mean(xl*(pl-gl))])
        derivative=np.array([np.mean(xl**2*pl*(1-pl)),np.mean(xu**2*pu*(1-pu))-np.mean(xl**2*pl*(1-pl))])
        return float(moment@weight@moment/2),float(derivative@weight@moment),moment
    witness_beta=float(np.log(1.5));q,g,_=values(witness_beta)
    witness_error=abs(q-number(exact['witness']['criterion']));assert witness_error<2e-14
    plus=np.array([0.,-35/44]);minus=np.array([-3.,53/44])
    tail_error=max(abs(float(plus@weight@plus/2)-number(exact['positive_infinite_coefficient']['criterion'])),abs(float(minus@weight@minus/2)-number(exact['negative_infinite_coefficient']['criterion'])))
    assert tail_error<2e-14
    gradient_error=0.
    for beta in np.linspace(-2,2,21):
        step=1e-5
        fd=(values(beta+step)[0]-values(beta-step)[0])/(2*step)
        gradient_error=max(gradient_error,abs(fd-values(beta)[1]))
    assert gradient_error<2e-8
    # Check the candidate certificate's observable terms. This finite grid
    # only checks arithmetic; the outward-score proof gives the all-center
    # impossibility statement for a separated supervised likelihood.
    counts=dict(grid_pairs=0,positive_ar=0,positive_ar_and_Lr=0)
    for c in np.linspace(-4,4,81):
        pc=expit(xl*c);A=float(np.mean(xl**2*pc*(1-pc)))
        m=float(np.mean(xl*(pc-yl)));h=float(np.mean(abs(xl)**3))/4
        for radius in np.geomspace(.0001,2,61):
            ar=A-h*radius;Lr=ar*radius-abs(m)
            counts['grid_pairs']+=1;counts['positive_ar']+=int(ar>0)
            counts['positive_ar_and_Lr']+=int(ar>0 and Lr>0)
    assert counts['positive_ar_and_Lr']==0
    numerical=[]
    for start in [-4.,-1.,0.,.4,1.,4.,12.]:
        r=minimize(lambda b:values(float(b[0]))[0],np.array([start]),jac=lambda b:np.array([values(float(b[0]))[1]]),method='BFGS',options=dict(gtol=1e-11,maxiter=500))
        val,gradient,moment=values(float(r.x[0]))
        numerical.append(dict(initial_coefficient=start,returned_coefficient=float(r.x[0]),criterion=val,absolute_gradient=abs(gradient),optimizer_success=bool(r.success),moment1=float(moment[0]),moment2=float(moment[1]),global_optimality_certified=False))
    with (a.out/'numerical_returns.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(numerical[0]));w.writeheader();w.writerows(numerical)
    report=dict(date='2026-11-12',weight_matrix_error=weight_error,witness_error=witness_error,tail_error=tail_error,finite_difference_gradient_error=gradient_error,
        certificate_arithmetic_grid=counts,finite_global_attainment_evidence='Exact rational witness below both tails, not optimizer status or finite grid.',numerical_returns=len(numerical),new_sampling_or_risk_claim=False,
        numerical_returns_dominated_by_exact_finite_witness=sum(r['criterion']>q+1e-10 for r in numerical),
        all_numerical_returns_reported_success=all(r['optimizer_success'] for r in numerical),
        numpy_version=np.__version__,scipy_version=scipy_version,
        certificate_sha256=hashlib.sha256(a.certificate.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.out/'complete_gmm_independent_verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
