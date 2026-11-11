"""The existing corrected objective for whole-menu conditional logit.

Convex support and simplex projection are classical. This implementation keeps
each menu's probability-sum constraint; it claims no new general LP diagnostic.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from scipy import sparse
from scipy.optimize import linprog,minimize,nnls
from scipy.special import logsumexp,softmax
from threadpoolctl import threadpool_limits
from power_tuned_choice_audit import objective as binary_objective,support as binary_support


def simplex_projection(v,mass):
    """Euclidean projection onto nonnegative vectors summing to mass."""
    v=np.asarray(v,float)
    if mass==0:return np.zeros_like(v)
    assert mass>0
    ordered=np.sort(v,axis=-1)[...,::-1]
    sums=np.cumsum(ordered,axis=-1)-mass
    indices=np.arange(1,v.shape[-1]+1)
    active=(ordered-sums/indices)>0
    rho=np.sum(active,axis=-1)-1
    theta=np.take_along_axis(sums,np.expand_dims(rho,-1),axis=-1)[...,0]/(rho+1)
    return np.maximum(v-theta[...,None],0.)


def vertex_cone_projection(w,label):
    """Project onto cone{e_j-e_label:j!=label} in the zero-sum subspace."""
    w=np.asarray(w,float);label=np.broadcast_to(label,w.shape[:-1])
    y=np.take_along_axis(w,label[...,None],axis=-1)[...,0]
    theta=y.copy();mask=np.arange(w.shape[-1])!=label[...,None]
    for _ in range(w.shape[-1]):
        selected=(w>theta[...,None])&mask
        theta=(y+np.sum(np.where(selected,w,0.),axis=-1))/(1+np.sum(selected,axis=-1))
    z=np.minimum(w,theta[...,None])
    np.put_along_axis(z,label[...,None],theta[...,None],axis=-1)
    return w-z


class MenuLoss:
    def __init__(self,features,y,g,m,power):
        self.features=np.asarray(features,float)
        self.y=np.asarray(y,int);self.g=np.asarray(g,float);self.m=m;self.power=power
        n,J,p=self.features.shape
        assert self.g.shape==(n,J) and len(y)==n and 0<m<n and 0<=power<=1
        assert np.max(abs(self.g.sum(1)-1))<1e-10 and np.min(self.g)>=0
        self.weights=np.r_[np.full(m,(1-power)/m),np.full(n-m,power/(n-m))]
        labels=np.eye(J)[self.y[:m]]
        self.moment=(np.einsum('njp,nj->p',self.features[:m],labels-power*self.g[:m])/m
                     +power*np.einsum('njp,nj->p',self.features[m:],self.g[m:])/(n-m))

    def value(self,beta):
        utility=self.features@beta;p=softmax(utility,axis=1)
        return (float(self.weights@logsumexp(utility,axis=1)-self.moment@beta),
                np.einsum('n,nj,njp->p',self.weights,p,self.features)-self.moment)

    def hessian(self,beta):
        p=softmax(self.features@beta,axis=1)
        mean=np.einsum('nj,njp->np',p,self.features)
        return (np.einsum('n,nj,nja,njb->ab',self.weights,p,self.features,self.features)
                -np.einsum('n,na,nb->ab',self.weights,mean,mean))


def menu_features(attributes):
    n,J,d=attributes.shape
    constants=np.broadcast_to(np.eye(J)[:,:-1],(n,J,J-1))
    features=np.concatenate([constants,attributes],axis=2)
    return features-features.mean(axis=1,keepdims=True)


def support(loss):
    keep=loss.weights>0;f=loss.features[keep];w=loss.weights[keep]
    n,J,p=f.shape;contrasts=(f[:,:-1]-f[:,-1:]).reshape(-1,p)
    if np.linalg.matrix_rank(contrasts)<p:return {'status':'rank_deficient'}
    scale=np.maximum(abs(f).max(axis=(0,1)),1e-12)
    weighted=(w[:,None,None]*f/scale).reshape(n*J,p)
    size=n*J;eye=sparse.eye(size,format='csr');one=sparse.csr_matrix(np.ones((size,1)))
    fit=linprog(np.r_[np.zeros(size),-1.],A_ub=sparse.hstack([-eye,one]),b_ub=np.zeros(size),
                A_eq=sparse.vstack([sparse.hstack([sparse.kron(sparse.eye(n),np.ones((1,J))),sparse.csr_matrix((n,1))]),
                                   sparse.hstack([sparse.csr_matrix(weighted.T),sparse.csr_matrix((p,1))])]),
                b_eq=np.r_[np.ones(n),loss.moment/scale],
                bounds=[(0.,1.)]*size+[(0.,1/J)],method='highs')
    if fit.status==2:
        # A separate recession program; each menu has one maximum-utility slack.
        centered=f-f.mean(axis=1,keepdims=True)
        flat=centered.reshape(size,p)
        select=sparse.kron(sparse.eye(n),np.ones((J,1)),format='csr')
        ray=linprog(np.r_[-loss.moment,loss.moment,w],
                    A_ub=sparse.vstack([sparse.hstack([sparse.csr_matrix(flat),sparse.csr_matrix(-flat),-select]),
                                       sparse.csr_matrix(np.r_[np.ones(2*p),np.zeros(n)][None,:])]),
                    b_ub=np.r_[np.zeros(size),1.],bounds=(0.,None),method='highs')
        if not ray.success:return {'status':'unresolved'}
        direction=ray.x[:p]-ray.x[p:2*p]
        gap=float(loss.moment@direction-w@np.max(centered@direction,axis=1))
        if gap<=1e-8:return {'status':'unresolved','separation_gap':gap}
        value=loss.value(1e6*direction)[0]
        assert -1e-7<=value+1e6*gap<=np.log(J)+1e-7
        return {'status':'outside','separation_gap':gap,'direction':direction.tolist()}
    if not fit.success:return {'status':'unresolved'}
    probs=fit.x[:-1].reshape(n,J)
    residual=max(float(np.max(abs(probs.sum(1)-1))),float(np.max(abs(weighted.T@fit.x[:-1]-loss.moment/scale))))
    if residual>1e-7:return {'status':'unresolved','equality_residual':residual}
    margin=float(fit.x[-1])
    return {'status':'interior' if margin>1e-8 else 'boundary_or_near_boundary','margin':margin,'equality_residual':residual}


def residual_support(features,y,g,m,power):
    """Independent zero-sum residual formulation,with per-menu lower bounds."""
    n,J,p=features.shape;tau=(n-m)/m
    lower=np.vstack([power*g[:m]-np.eye(J)[y[:m]],-power*g[m:]/tau])
    if power<1:lower[:m]-=0.  # The positive simplex mass follows from sum zero.
    eq=sparse.vstack([sparse.kron(sparse.eye(n),np.ones((1,J))),sparse.csr_matrix(features.reshape(n*J,p).T)])
    fit=linprog(np.zeros(n*J),A_eq=eq,b_eq=np.zeros(n+p),bounds=[(a,None) for a in lower.ravel()],method='highs')
    if fit.status==2:return 'outside'
    if not fit.success:return 'unresolved'
    return 'feasible' if np.max(abs(eq@fit.x))<1e-7 else 'unresolved'


def save(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def diagnostics(out):
    out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(2026111147)
    gradient=hessian=identity=projection=cone_error=shift_error=0.;matches=binary_matches=0
    states={};records=[]
    for trial in range(120):
        J=2+trial%3;m=14+trial%7;N=m*(1+(trial//3)%3);d=2+trial%8
        f=menu_features(rng.normal(size=(m+N,J,d)));y=rng.integers(J,size=m+N)
        z=(y+rng.integers(1,J,size=m+N)*(rng.random(m+N)<.25))%J
        g=np.full((m+N,J),.25/(J-1));g[np.arange(m+N),z]=.75
        current=[]
        for power in [1.,.5,.1]:
            loss=MenuLoss(f,y,g,m,power);original=support(loss);separate=residual_support(f,y,g,m,power)
            assert original['status']!='unresolved' and separate!='unresolved',(trial,power,original,separate)
            assert (original['status']=='interior' and separate=='feasible') or original['status']==separate,(trial,power,original,separate)
            matches+=1;current.append(original['status']);key=f'J{J}_{original["status"]}';states[key]=states.get(key,0)+1
            records.append({'trial':trial,'alternatives':J,'labels':m,'auxiliary':N,'profile_slopes':d,'power':power,'status':original['status'],'residual_status':separate})
            if J==2:
                bx=f[:,1]-f[:,0];old=binary_objective(bx,(y==1).astype(float),g[:,1],m,power)
                bs=binary_support(old)['status'];expected='interior' if bs=='interior' else 'outside' if bs=='outside_mean_support' else bs
                assert expected==original['status'];binary_matches+=1
                beta=rng.normal(size=f.shape[-1])/4
                identity=max(identity,abs(loss.value(beta)[0]-old.value(beta)[0]),float(np.max(abs(loss.value(beta)[1]-old.value(beta)[1]))))
            if trial<9:
                beta=rng.normal(size=f.shape[-1])/4;step=1e-5;eye=np.eye(len(beta))
                fd=np.array([(loss.value(beta+e*step)[0]-loss.value(beta-e*step)[0])/(2*step) for e in eye])
                fh=np.column_stack([(loss.value(beta+e*step)[1]-loss.value(beta-e*step)[1])/(2*step) for e in eye])
                gradient=max(gradient,float(np.max(abs(fd-loss.value(beta)[1]))));hessian=max(hessian,float(np.max(abs(fh-loss.hessian(beta)))))
                common=rng.normal(size=(m+N,1,f.shape[-1]))
                shifted=MenuLoss(f+common,y,g,m,power)
                shift_error=max(shift_error,abs(shifted.value(beta)[0]-loss.value(beta)[0]),float(np.max(abs(shifted.value(beta)[1]-loss.value(beta)[1]))))
        assert not(current[0]=='interior' and current[1]!='interior') and not(current[1]=='interior' and current[2]!='interior')
        if trial<24:
            v=rng.normal(size=J);mass=.2+rng.random()
            projected=simplex_projection(v,mass)
            qp=minimize(lambda x:(float(np.sum((x-v)**2)),2*(x-v)),np.full(J,mass/J),jac=True,
                        constraints={'type':'eq','fun':lambda x:x.sum()-mass,'jac':lambda x:np.ones(J)},bounds=[(0,None)]*J,
                        method='SLSQP',options={'ftol':1e-12,'maxiter':1000})
            assert qp.success;projection=max(projection,float(np.max(abs(projected-qp.x))))
            w=v-v.mean();label=trial%J;direct=vertex_cone_projection(w,label)
            generators=(np.eye(J)-np.eye(J)[label])[:,np.arange(J)!=label]
            coefficients,_=nnls(generators,w);cone_error=max(cone_error,float(np.max(abs(direct-generators@coefficients))))
            assert abs(np.sum(direct))<1e-10 and abs(direct@(w-direct))<1e-10
    # Intercept-only menu,interior target equals uniform and has root zero.
    f=menu_features(np.zeros((6,3,0)));y=np.array([0,1,2,0,1,2]);g=np.full((6,3),1/3)
    inside=MenuLoss(f,y,g,3,1.);assert support(inside)['status']=='interior' and max(abs(inside.value(np.zeros(2))[1]))<1e-12
    y[:3]=0;boundary=MenuLoss(f,y,g,3,1.);assert support(boundary)['status']=='boundary_or_near_boundary'
    g[:3]=[.1,.45,.45];g[3:]=[.8,.1,.1];assert support(MenuLoss(f,y,g,3,1.))['status']=='outside'
    assert support(MenuLoss(menu_features(np.zeros((6,3,1))),y,g,3,.5))['status']=='rank_deficient'
    assert max(gradient,hessian,identity,projection,cone_error,shift_error)<1e-7
    save(out/'menu_diagnostic_records.json',records)
    result={'finite_menu_designs':120,'original_moment_residual_LP_matches':matches,'binary_objective_support_matches':binary_matches,
            'states':states,'maximum_gradient_error':gradient,'maximum_Hessian_error':hessian,
            'maximum_binary_objective_score_identity_error':identity,'maximum_simplex_independent_QP_error':projection,
            'maximum_vertex_cone_NNLS_error':cone_error,'maximum_common_menu_shift_error':shift_error,
            'analytic_uniform_root_boundary_exterior_rank_checks':True,'new_human_or_LLM_responses':0,
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'novelty_or_submission_readiness_established':False}
    save(out/'menu_diagnostic_checks.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    with threadpool_limits(limits=1):diagnostics(args.out)
