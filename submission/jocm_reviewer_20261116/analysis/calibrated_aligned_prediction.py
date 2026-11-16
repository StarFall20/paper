"""Same calibrated proxy marginals,external versus fitted Gaussian index."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from scipy.special import roots_legendre
from scipy.stats import norm
from threadpoolctl import threadpool_limits
from attribute_aligned_prediction import AlignedPopulation
from finite_ratio_signal_boundary import tail_derivative
from finite_ratio_choice_boundary import critical_ratio
from public_conjoint_optimizer_audit import save


class CalibratedAlignedPopulation(AlignedPopulation):
    def __init__(self,epsilon,tau,power,nodes=80):
        root,weight=roots_legendre(nodes)
        indices=[];weights=[];labels=[];proxies=[]
        for y in [0.,1.]:
            cutoff=norm.ppf(epsilon if y else 1-epsilon)
            for z,lower,upper in [(0.,-12.,cutoff),(1.,cutoff,12.)]:
                index=lower+(root+1)*(upper-lower)/2
                indices.extend(index);labels.extend(np.full(nodes,y));proxies.extend(np.full(nodes,z))
                weights.extend(weight*(upper-lower)/2*norm.pdf(index)/2)
        self.T=np.asarray(indices);self.weights=np.asarray(weights);self.y=np.asarray(labels);self.z=np.asarray(proxies)
        self.g=epsilon+(1-2*epsilon)*self.z;self.sign=1-2*self.y
        self.c=.5-epsilon;self.epsilon=epsilon;self.tau=tau;self.power=power
        r=np.where(self.y==0,self.g,1-self.g)
        self.aL=power*r;self.bL=1-power+power*r
        self.aU=-power*self.g/tau;self.bU=power*(1-self.g)/tau
        self.V=np.c_[np.ones(len(self.T)),self.T]
        assert abs(self.weights.sum()-1)<1e-12


def checks(out):
    rows=[];quadrature=scalar=derivative=gradient=symmetry=calibration=0.
    for epsilon in [.05,.1,.25]:
      for tau in [1.,4.]:
       for power in [1.,.5]:
        low=CalibratedAlignedPopulation(epsilon,tau,power,80);high=CalibratedAlignedPopulation(epsilon,tau,power,160)
        ext=low.critical(False);aligned=low.critical(True);fine_ext=high.critical(False);fine_aligned=high.critical(True)
        quadrature=max(quadrature,abs(ext['critical_ratio']-fine_ext['critical_ratio']),abs(aligned['critical_ratio']-fine_aligned['critical_ratio']))
        classic=critical_ratio(tau,epsilon,power)[0];scalar=max(scalar,abs(classic-fine_ext['critical_ratio']))
        eta=np.array([0.,fine_aligned['index_multiplier']]);eye=np.eye(2);step=1e-5
        fd=np.array([(high.maximize(eta+step*e)[0]-high.maximize(eta-step*e)[0])/(2*step) for e in eye])
        gradient=max(gradient,float(max(abs(fd-high.maximize(eta)[1]))))
        symmetry=max(symmetry,abs(high.maximize(eta+np.array([.3,0.]))[0]-high.maximize(eta-np.array([.3,0.]))[0]))
        for z in [0.,1.]:
            selection=high.z==z
            calibration=max(calibration,abs(high.weights[selection]@high.y[selection]/high.weights[selection].sum()-(epsilon+(1-2*epsilon)*z)))
        assert max(abs(high.maximize(eta)[1]))<1e-7
        assert fine_aligned['critical_ratio']<=fine_ext['critical_ratio']+1e-10
        row={'epsilon':epsilon,'tau':tau,'power':power,'external_critical_ratio':fine_ext['critical_ratio'],
             'aligned_critical_ratio':fine_aligned['critical_ratio'],'boundary_gap':fine_ext['critical_ratio']-fine_aligned['critical_ratio'],
             'aligned_index_multiplier':fine_aligned['index_multiplier'],'aligned_scale':fine_aligned['scale'],
             'external_scale':fine_ext['scale'],'Brier_risk_both':epsilon*(1-epsilon),
             'calibrated_predictor_both':True,'external_index_gradient':fine_ext['index_gradient']}
        if power==1:
            t=fine_ext['scale'];c=.5-epsilon
            analytic=2*norm.pdf(norm.ppf(1-epsilon))*(-2*t*c+tau*(tail_derivative(t*(1-epsilon)/tau)-tail_derivative(t*epsilon/tau)))
            derivative=max(derivative,abs(analytic-fine_ext['index_gradient']))
            assert analytic<0 and row['boundary_gap']>0
            row['analytic_negative_index_derivative']=analytic
        rows.append(row)
    pd.DataFrame(rows).to_csv(out/'calibrated_aligned_boundaries.csv',index=False)
    result={'population_conditions':12,'80_160_node_max_error':quadrature,'existing_balanced_scalar_formula_max_error':scalar,
            'negative_unit_derivative_identity_error':derivative,'envelope_gradient_finite_difference_error':gradient,
            'intercept_symmetry_error':symmetry,'conditional_calibration_probability_error':calibration,
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'population_helper_sha256':hashlib.sha256((Path(__file__).parent/'attribute_aligned_prediction.py').read_bytes()).hexdigest(),
            'matched_complete_Y_Z_g_law':True,'new_human_or_AI_responses':0,'priority_or_submission_readiness_proven':False}
    save(out/'calibrated_aligned_population_attempt.json',result)
    assert max(quadrature,scalar,derivative,gradient,calibration)<1e-7 and symmetry<1e-9,result
    save(out/'calibrated_aligned_numeric_checks.json',result)
    print(json.dumps(result),flush=True)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=1):checks(a.out)
