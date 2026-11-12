"""Exploratory fixed-cluster compensation boundary and matched-moment audit.

The conic transformation and Gaussian kinematic tools are established. This
program evaluates a proposed block extension; numerical agreement is not an
independent theorem or originality review. No respondent data are required.
"""
from pathlib import Path
import argparse,hashlib,itertools,json
import numpy as np
import pandas as pd
from scipy.optimize import minimize,minimize_scalar
from scipy.special import ndtri
from scipy.stats import qmc
from finite_ratio_choice_boundary import critical_ratio


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def symmetric_root(covariance):
    values,vectors=np.linalg.eigh(covariance)
    if values.min()<=0:raise ValueError('The row covariance must be positive definite.')
    return (vectors*np.sqrt(values))@vectors.T


class BoxProjection:
    """Enumerate every face of a small linearly transformed box."""
    def __init__(self,covariance,normal):
        self.cov=np.asarray(covariance,float)
        self.root=symmetric_root(self.cov)
        self.normal=np.asarray(normal,float)
        self.linear=self.normal@self.root
        self.dimension=len(self.cov)
        self.faces=[]
        for pattern in itertools.product([-1,0,1],repeat=self.dimension):
            fixed=np.flatnonzero(np.array(pattern)!=0)
            free=np.flatnonzero(np.array(pattern)==0)
            inverse=np.linalg.inv(self.cov[np.ix_(free,free)]) if len(free) else None
            self.faces.append((pattern,fixed,free,inverse))

    def value(self,lower,upper,return_projection=False):
        lower=np.asarray(lower,float);upper=np.asarray(upper,float)
        if np.any(lower>upper):raise ValueError('Reversed box endpoints.')
        if np.array_equal(lower,upper):
            r=np.broadcast_to(lower,self.linear.shape)
            value=2*(self.linear@lower)-lower@self.cov@lower
            return (value,r@self.root) if return_projection else value
        best=np.full(len(self.normal),-np.inf)
        chosen=np.zeros_like(self.normal) if return_projection else None
        for pattern,fixed,free,inverse in self.faces:
            point=np.empty_like(self.normal)
            if len(fixed):point[:,fixed]=[lower[j] if pattern[j]<0 else upper[j] for j in fixed]
            if len(free):
                rhs=self.linear[:,free].copy()
                if len(fixed):rhs-=point[:,fixed]@self.cov[np.ix_(fixed,free)]
                point[:,free]=rhs@inverse
                feasible=np.all((point[:,free]>=lower[free]-1e-12)&(point[:,free]<=upper[free]+1e-12),axis=1)
            else:feasible=np.ones(len(point),bool)
            value=2*np.sum(self.linear*point,axis=1)-np.sum((point@self.cov)*point,axis=1)
            value=np.where(feasible,value,-np.inf)
            better=value>best
            if return_projection:chosen[better]=(point[better]@self.root)
            best=np.maximum(best,value)
        assert np.isfinite(best).all()
        return (best,chosen) if return_projection else best


def cluster_patterns(size,dependence,epsilon):
    """With probability dependence use one shared fair outcome; else independent.

    Independent proxy flips then preserve rowwise calibration. Antipodal pairs
    can be combined because the numerical Gaussian rule is exactly antithetic.
    """
    labelled=[];unlabelled={}
    for yt in itertools.product([0,1],repeat=size):
        y=np.array(yt,float)
        probability=(1-dependence)/2**size+dependence/2*(len(set(yt))==1)
        for zt in itertools.product([0,1],repeat=size):
            z=np.array(zt,float);errors=np.count_nonzero(y!=z)
            mass=probability*epsilon**errors*(1-epsilon)**(size-errors)
            if not mass:continue
            g=epsilon+(1-2*epsilon)*z
            if yt[0]==0:labelled.append((y,g,2*mass))
            if zt[0]==0:unlabelled[zt]=unlabelled.get(zt,0.)+2*mass
    auxiliary=[(epsilon+(1-2*epsilon)*np.array(z,float),mass) for z,mass in unlabelled.items()]
    assert abs(sum(p for _,_,p in labelled)-1)<1e-12
    assert abs(sum(p for _,p in auxiliary)-1)<1e-12
    return labelled,auxiliary


def normal_rule(size,exponent,seed):
    u=qmc.Sobol(size,scramble=True,seed=seed).random_base2(exponent-1)
    h=ndtri(u)
    return np.vstack([h,-h])


def evaluate_boundary(covariance,dependence,epsilon,power,tau,exponent,seed):
    size=len(covariance);h=normal_rule(size,exponent,seed)
    projection=BoxProjection(covariance,h)
    labelled,auxiliary=cluster_patterns(size,dependence,epsilon)
    def objective(scale):
        total=0.
        for y,g,probability in labelled:
            lower=scale*(g-y/power)
            upper=lower+scale*(1-power)/power
            total+=probability*projection.value(lower,upper).mean()
        for g,probability in auxiliary:
            total+=tau*probability*projection.value(-scale*g/tau,scale*(1-g)/tau).mean()
        return float(total)
    upper=1.
    while objective(2*upper)>objective(upper):
        upper*=2
        if upper>1024:raise ArithmeticError('Could not bracket the conic scale.')
    fit=minimize_scalar(lambda v:-objective(v),bounds=(0.,2*upper),method='bounded',options={'xatol':1e-9})
    assert fit.success and 0<fit.x<2*upper
    return dict(critical_slopes_per_labelled_person=float(-fit.fun),critical_slopes_per_labelled_choice=float(-fit.fun/size),scaled_minimizer=float(fit.x),normal_points=len(h),exponent=exponent,seed=seed)


def ordinary_three_task_boundary(covariance,dependence):
    """Classical solid-angle/statistical-dimension reduction, not new theory."""
    size=len(covariance);assert size==3
    inv=np.linalg.inv(covariance)
    corr=inv/np.sqrt(np.outer(np.diag(inv),np.diag(inv)))
    upper=np.triu_indices(3,1)
    same=1.5+(np.arcsin(corr[upper]).sum()-np.arcsin(covariance[upper]).sum())/(2*np.pi)
    return float(dependence*same+(1-dependence)*1.5)


def score_moments(covariance,dependence,epsilon):
    """Joint covariance of human and prediction average cluster score at beta=0.

    One intercept and one representative independent Gaussian feature suffice:
    all fitted Gaussian coordinates have the same block and zero cross moments.
    """
    size=len(covariance);a=1-2*epsilon
    cy=np.full((size,size),dependence/4);np.fill_diagonal(cy,.25)
    cp=a*a*cy
    pp=a**4*cy;np.fill_diagonal(pp,a*a/4)
    matrices={}
    for name,weight in [('intercept',np.ones((size,size))),('Gaussian_feature',covariance)]:
        matrices[name]=np.array([[np.sum(weight*cy),np.sum(weight*cp)],[np.sum(weight*cp),np.sum(weight*pp)]])/size**2
    return {key:value.tolist() for key,value in matrices.items()}


def numerical_checks(plan):
    rng=np.random.default_rng(9505093);errors=[];moment_errors=[]
    for model in plan['models']:
        cov=np.array(model['covariance']);root=symmetric_root(cov)
        h=rng.normal(size=(30,3));p=BoxProjection(cov,h)
        for point in range(len(h)):
            lower=rng.uniform(-1.5,.2,3);upper=lower+rng.uniform(.05,2.,3)
            values,proj=BoxProjection(cov,h[point:point+1]).value(lower,upper,True)
            linear=h[point]@root
            check=minimize(lambda r:np.sum((h[point]-r@root)**2),np.clip(linear@np.linalg.inv(cov),lower,upper),jac=lambda r:2*(r@cov-linear),bounds=list(zip(lower,upper)),method='L-BFGS-B',options={'ftol':1e-14,'gtol':1e-11,'maxiter':1000})
            expected=np.sum(h[point]**2)-check.fun
            errors.append(abs(values[0]-expected))
            assert check.success or np.linalg.norm(check.jac)<1e-6
    for dependence in plan['outcome_dependences']:
        a=score_moments(np.array(plan['models'][1]['covariance']),dependence,plan['epsilon'])
        b=score_moments(np.array(plan['models'][2]['covariance']),dependence,plan['epsilon'])
        moment_errors.extend(abs(np.array(a[k])-np.array(b[k])).max() for k in a)
    assert max(errors)<1e-9,max(errors)
    assert max(moment_errors)<1e-14,max(moment_errors)
    return dict(independent_box_projection_comparisons=len(errors),maximum_projection_value_error=max(errors),matched_population_score_covariance_error=max(moment_errors),prior_ordinary_boundary_is_not_a_novelty_claim=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text());args.out.mkdir(parents=True,exist_ok=True)
    save(args.out/'numerical_checks.json',numerical_checks(plan))
    rows=[];ordinary=[];moments=[]
    for model in plan['models']:
        cov=np.array(model['covariance'],float)
        for dependence in plan['outcome_dependences']:
            ordinary.append(dict(model=model['name'],outcome_dependence=dependence,ordinary_slopes_per_person=ordinary_three_task_boundary(cov,dependence)))
            moments.append(dict(model=model['name'],outcome_dependence=dependence,moments=score_moments(cov,dependence,plan['epsilon'])))
            for power in plan['powers']:
                for exponent in plan['normal_exponents']:
                    for seed in plan['normal_seeds']:
                        value=evaluate_boundary(cov,dependence,plan['epsilon'],power,plan['tau'],exponent,seed)
                        rows.append(dict(model=model['name'],outcome_dependence=dependence,power=power,**value))
                pd.DataFrame(rows).to_csv(args.out/'population_boundaries.csv',index=False)
                print(model['name'],dependence,power,'population cells',len(rows),flush=True)
    pd.DataFrame(ordinary).to_csv(args.out/'ordinary_classical_boundaries.csv',index=False)
    save(args.out/'score_population_moments.json',moments)
    save(args.out/'source_hashes.json',dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(),candidate_extension_not_independently_proof_reviewed=True))


if __name__=='__main__':main()
