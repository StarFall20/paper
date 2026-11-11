"""Population cone calculations for the same corrected menu-level logit.

Gaussian quadrature here is scrambled Sobol integration in orthonormal menu
contrasts. It is numerical integration,not additional choice respondents.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from scipy.linalg import helmert
from scipy.optimize import brentq
from scipy.special import ndtri,softmax
from scipy.stats import qmc
from threadpoolctl import threadpool_limits
from menu_corrected_logit import simplex_projection,vertex_cone_projection,save


class MenuPopulation:
    def __init__(self,J,gamma,epsilon,tau,power,bits=12,seed=2026111149):
        self.J=J;self.gamma=gamma;self.tau=tau;self.power=power
        r=J-1;basis=helmert(J,full=False).T
        unit=qmc.Sobol(2*r,scramble=True,seed=seed).random_base2(bits)
        normal=ndtri(np.clip(unit,1e-15,1-1e-15));normal=np.vstack([normal,-normal])
        self.S=normal[:,:r]@basis.T;self.H=normal[:,r:]@basis.T
        self.mu=softmax(gamma*self.S,axis=1)
        transition=np.full((J,J),epsilon/(J-1));np.fill_diagonal(transition,1-epsilon)
        self.joint=self.mu[:,None,:]*transition[None,:,:]
        self.pZ=self.joint.sum(axis=2)
        self.g=self.joint/self.pZ[:,:,None]
        self.eye=np.eye(J);self.n=len(normal);self.seed=seed;self.bits=bits
        assert np.max(abs(self.joint.sum(axis=(1,2))-1))<1e-12
        assert np.max(abs(self.S.sum(axis=1)))<1e-12

    def _mean(self,label,aux):
        return float(np.mean(np.sum(self.joint*label,axis=(1,2))+self.tau*np.sum(self.pZ*aux,axis=1)))

    def value(self,eta,scale,limiting=False):
        w=self.H-eta*self.S;norm=np.sum(w*w,axis=1)
        if limiting:
            c=self.g[:,:,None,:]-self.eye[None,None,:,:]
            shifted=w[:,None,None,:]-scale*c
            label_projection=scale*c+vertex_cone_projection(shifted,np.arange(self.J)[None,None,:])
            aux_shift=self.g/self.tau
            aux_projection=simplex_projection(w[:,None,:]+scale*aux_shift,scale/self.tau)-scale*aux_shift
        else:
            lam=self.power
            shift=self.eye[None,None,:,:]-lam*self.g[:,:,None,:]
            label_projection=simplex_projection(w[:,None,None,:]+scale*shift,scale*(1-lam))-scale*shift
            aux_shift=lam*self.g/self.tau
            aux_projection=simplex_projection(w[:,None,:]+scale*aux_shift,scale*lam/self.tau)-scale*aux_shift
        label_residual=w[:,None,None,:]-label_projection
        aux_residual=w[:,None,:]-aux_projection
        label=norm[:,None,None]-np.sum(label_residual**2,axis=-1)
        aux=norm[:,None]-np.sum(aux_residual**2,axis=-1)
        value=self._mean(label,aux)
        gradient=-2*self._mean(np.sum(self.S[:,None,None,:]*label_projection,axis=-1),
                               np.sum(self.S[:,None,:]*aux_projection,axis=-1))
        if limiting:
            dt_label=2*np.sum(label_residual*c,axis=-1)
            if scale:
                dt_aux=2*np.sum(aux_residual*aux_projection,axis=-1)/scale
            else:
                dt_aux=2*(np.max(w,axis=1)[:,None]-np.sum(self.g*w[:,None,:],axis=-1))/self.tau
            dt=self._mean(dt_label,dt_aux)
        elif scale:
            dt=2*self._mean(np.sum(label_residual*label_projection,axis=-1),np.sum(aux_residual*aux_projection,axis=-1))/scale
        else:
            dt=2*float(np.mean(np.max(w,axis=1)-np.sum(self.mu*w,axis=1)))
        return value,gradient,dt

    def maximize(self,eta,limiting=False):
        start=self.value(eta,0.,limiting)
        if start[2]<=0:return start[0],start[1],0.
        upper=1.
        while self.value(eta,upper,limiting)[2]>=0:
            upper*=2
            if upper>1e5:raise RuntimeError('Menu cone scale not bracketed')
        scale=brentq(lambda t:self.value(eta,t,limiting)[2],0.,upper,xtol=1e-9)
        value,gradient,_=self.value(eta,scale,limiting)
        return value,gradient,scale

    def ordinary_value(self,eta):
        w=self.H-eta*self.S
        projection=vertex_cone_projection(w[:,None,:]+np.zeros((1,self.J,1)),np.arange(self.J)[None,:])
        value=float(np.mean(np.sum(self.mu*np.sum(projection**2,axis=-1),axis=1)))
        gradient=-2*float(np.mean(np.sum(self.mu*np.sum(self.S[:,None,:]*projection,axis=-1),axis=1)))
        return value,gradient

    def critical(self,limiting=False,ordinary=False):
        if ordinary:
            fn=lambda e:(*self.ordinary_value(e),0.)
        else:fn=lambda e:self.maximize(e,limiting)
        if self.gamma==0:
            eta=0.
        else:
            lower,upper=-1.,1.
            while fn(lower)[1]>=0:lower*=2
            while fn(upper)[1]<=0:upper*=2
            eta=brentq(lambda e:fn(e)[1],lower,upper,xtol=1e-8)
        value,gradient,scale=fn(eta)
        return dict(critical_ratio=value,signal_multiplier=eta,scale=scale,gradient=gradient,
                    scale_derivative=0. if ordinary else self.value(eta,scale,limiting)[2])

    def zero_scale_identity(self,eta):
        w=self.H-eta*self.S
        projection=vertex_cone_projection(w[:,None,:]+np.zeros((1,self.J,1)),np.arange(self.J)[None,:])
        maximum=np.max(projection,axis=-1)[:,None,:]
        dot=np.einsum('nzk,nyk->nzy',self.g,projection)
        expected=2*float(np.mean(np.sum(self.joint*(maximum-dot),axis=(1,2))))
        return self.value(eta,0.,True)[2],expected


def checks(out,bits):
    out.mkdir(parents=True,exist_ok=True);rows=[];derivative=identity=ordinary_limit=0.
    for J in [2,3,4]:
      for gamma in [0.,1.5]:
        model=MenuPopulation(J,gamma,.25,1.,.5,bits)
        eta=.3;t=.8;step=1e-5
        value,grad,dt=model.value(eta,t)
        fd=(model.value(eta+step,t)[0]-model.value(eta-step,t)[0])/(2*step)
        ft=(model.value(eta,t+step)[0]-model.value(eta,t-step)[0])/(2*step)
        derivative=max(derivative,abs(fd-grad),abs(ft-dt))
        for e in [-.4,0.,.7]:
            one,two=model.zero_scale_identity(e);identity=max(identity,abs(one-two));assert two>0
            ordinary_limit=max(ordinary_limit,abs(model.value(e,0.,True)[0]-model.ordinary_value(e)[0]))
        corrected=model.critical();ordinary=model.critical(ordinary=True);limit=model.critical(limiting=True)
        assert limit['critical_ratio']>ordinary['critical_ratio'] and limit['critical_ratio']>=corrected['critical_ratio']-1e-9
        rows.append(dict(alternatives=J,gamma=gamma,tau=1.,epsilon=.25,power=.5,bits=bits,nodes=model.n,
                         critical_ratio=corrected['critical_ratio'],ordinary_ratio=ordinary['critical_ratio'],
                         positive_power_limit=limit['critical_ratio'],signal_multiplier=corrected['signal_multiplier']))
        pd.DataFrame(rows).to_csv(out/'menu_population_pilot.csv',index=False)
        print(json.dumps(rows[-1]),flush=True)
    assert derivative<1e-6 and identity<1e-9 and ordinary_limit<1e-9
    save(out/'menu_population_pilot_checks.json',dict(maximum_gradient_or_scale_derivative_error=derivative,
          maximum_strict_gain_derivative_identity_error=identity,maximum_ordinary_zero_scale_identity_error=ordinary_limit,
          source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),bits=bits,
          stage='Integration/derivation pilot; not finite phase data or a novelty certificate'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--bits',type=int,default=10);a=p.parse_args()
    with threadpool_limits(limits=1):checks(a.out,a.bits)
