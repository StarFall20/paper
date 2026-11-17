"""Respondent-separated audit of an existing compensated menu loss.

Public derived results only. The source workbook is read, never edited or copied.
No downloaded research code is executed. Support and trace tuning are prior tools.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from scipy.optimize import linprog,minimize
from scipy.special import logsumexp,softmax
from threadpoolctl import threadpool_limits

BASE=Path(__file__).resolve().parents[1]
NAMES=['option1','option2','bere','nutrition_medium','nutrition_high','quality_label','access_speciality','access_grocery','origin_elsewhere_Scotland','origin_outside_Scotland','cost_GBP10']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')

def load(path,plan):
    assert sha(path)==plan['source_sha256']
    c=pd.read_excel(path,'choices');s=pd.read_excel(path,'survey').set_index('unique_id')
    assert not c.duplicated(['unique_id','choice_card','option']).any()
    groups=c.groupby(['unique_id','choice_card']);assert groups.size().eq(3).all() and groups.choice.sum().eq(1).all()
    counts=groups.size().groupby('unique_id').size();ids=np.sort(counts[counts.eq(6)].index.to_numpy());assert len(ids)==298
    c=c[c.unique_id.isin(ids)].sort_values(['unique_id','choice_card','option'])
    x=lambda name:c[name].to_numpy().reshape(298,6,3)
    assert np.array_equal(x('option'),np.broadcast_to([1,2,3],(298,6,3)))
    features=np.stack([x('option')==1,x('option')==2,x('ingr')==1,x('nutr')==2,x('nutr')==3,x('label')==2,x('access')==2,x('access')==3,x('prox')==2,x('prox')==3,x('cost')/10],axis=-1).astype(float)
    features-=features.mean(axis=2,keepdims=True)
    y=x('choice').argmax(axis=2);card=x('choice_card')[:,:,0]
    demographics=s.loc[ids,['age','education']].to_numpy(float)
    order=np.random.default_rng(plan['partition_seed']).permutation(len(ids));train=order[:100];test=order[100:160];pool=order[160:]
    for j in range(2):demographics[np.isnan(demographics[:,j]),j]=np.nanmedian(demographics[train,j])
    z=np.column_stack([(demographics[:,0]-4)/3,(demographics[:,1]-3)/2])
    teacher=np.concatenate([features,features*z[:,0,None,None,None],features*z[:,1,None,None,None]],axis=-1)
    assert len(set(train)&set(test))==len(set(train)&set(pool))==len(set(test)&set(pool))==0
    # Exact compression is by the design vector, independent of observed choices.
    design,inv=np.unique(features.reshape(298*6,-1),axis=0,return_inverse=True)
    assert len(design)==34  # Two repeated design pairs among the 36 card numbers.
    return dict(f=features,y=y,card=card,teacher=teacher,train=train,test=test,pool=pool,design=design.reshape(-1,3,11),design_index=inv.reshape(298,6),ids=ids)

class Loss:
    def __init__(self,f,w,b,ridge=0.):self.f=f;self.w=w;self.b=b;self.ridge=ridge
    def value(self,beta):
        u=self.f@beta;p=softmax(u,axis=1)
        return (float(self.w@logsumexp(u,axis=1)-self.b@beta+self.ridge*(beta@beta)/2),np.einsum('n,nj,njp->p',self.w,p,self.f)-self.b+self.ridge*beta)
    def hessian(self,beta):
        p=softmax(self.f@beta,axis=1);mean=np.einsum('nj,njp->np',p,self.f)
        return np.einsum('n,nj,nja,njb->ab',self.w,p,self.f,self.f)-np.einsum('n,na,nb->ab',self.w,mean,mean)+self.ridge*np.eye(len(beta))

def build(data,people,target,ridge=0.):
    f=data['f'][people].reshape(-1,3,11);weights=np.full(len(f),1/len(f))
    b=np.einsum('njp,nj->p',f,target.reshape(-1,3))/len(f)
    w=np.bincount(data['design_index'][people].ravel(),weights,minlength=len(data['design']))
    return Loss(data['design'],w,b,ridge)

def corrected(data,labelled,aux,g,power):
    y=np.eye(3)[data['y'][labelled]];one=build(data,labelled,y);prL=build(data,labelled,g[labelled]);prU=build(data,aux,g[aux])
    return Loss(data['design'],(1-power)*one.w+power*prU.w,one.b-power*prL.b+power*prU.b)

def support(loss):
    keep=loss.w>0;f=loss.f[keep];w=loss.w[keep];n,J,p=f.shape
    rank=int(np.linalg.matrix_rank((f[:,:-1]-f[:,-1:]).reshape(-1,p)))
    if rank<p:return dict(support='rank_deficient',rank=rank)
    weighted=(w[:,None,None]*f).reshape(n*J,p)
    eq=np.r_[np.c_[np.kron(np.eye(n),np.ones((1,J))),np.zeros(n)],np.c_[weighted.T,np.zeros(p)]]
    ub=np.c_[-np.eye(n*J),np.ones(n*J)]
    fit=linprog(np.r_[np.zeros(n*J),-1.],A_ub=ub,b_ub=np.zeros(n*J),A_eq=eq,b_eq=np.r_[np.ones(n),loss.b],bounds=[(0,1)]*(n*J)+[(0,1/J)],method='highs')
    if fit.status==2:
        # Independently interpretable unit-L1 negative original recession ray.
        select=np.kron(np.eye(n),np.ones((J,1)));flat=f.reshape(n*J,p)
        ray=linprog(np.r_[-loss.b,loss.b,w],A_ub=np.r_[np.c_[flat,-flat,-select],np.r_[np.ones(2*p),np.zeros(n)][None,:]],b_ub=np.r_[np.zeros(n*J),1.],bounds=[(0,None)]*(2*p)+[(None,None)]*n,method='highs')
        if not ray.success:return dict(support='unresolved',rank=rank)
        d=ray.x[:p]-ray.x[p:2*p];gap=float(loss.b@d-w@np.max(f@d,axis=1))
        if gap<=1e-8:return dict(support='unresolved',rank=rank,separation_gap=gap)
        return dict(support='outside',rank=rank,separation_gap=gap,direction=d.tolist())
    if not fit.success:return dict(support='unresolved',rank=rank)
    resid=float(np.max(abs(eq@fit.x-np.r_[np.ones(n),loss.b])));margin=float(fit.x[-1])
    return dict(support='interior' if margin>1e-8 else 'boundary_or_near_boundary',rank=rank,interior_margin=margin,equality_residual=resid)

def fit_loss(loss,start):
    fit=minimize(loss.value,start,jac=True,method='BFGS',options={'gtol':1e-10,'maxiter':2000});beta=fit.x
    for _ in range(15):
        val,grad=loss.value(beta)
        if max(abs(grad))<=1e-9:break
        try:step=np.linalg.solve(loss.hessian(beta),grad)
        except np.linalg.LinAlgError:break
        a=1.
        while a>1e-8 and loss.value(beta-a*step)[0]>val-1e-4*a*float(grad@step):a/=2
        if a<=1e-8:break
        beta-=a*step
    gradient=float(max(abs(loss.value(beta)[1])))
    ok=np.isfinite(beta).all() and gradient<=1e-7
    return beta,dict(fit='passed_score' if ok else 'failed_score',gradient_max=gradient,optimizer_success=bool(fit.success))

def cluster_power(data,labelled,aux,g,pilot):
    f=data['f'];prob=softmax(f@pilot,axis=2);ys=np.eye(3)[data['y'][labelled]]
    h=np.einsum('ntjp,ntj->np',f[labelled],prob[labelled]-ys)/6
    dL=np.einsum('ntjp,ntj->np',f[labelled],prob[labelled]-g[labelled])/6
    dU=np.einsum('ntjp,ntj->np',f[aux],prob[aux]-g[aux])/6
    m=len(labelled);N=len(aux);hc=h-h.mean(0);dc=dL-dL.mean(0)
    cross=(hc.T@dc+dc.T@hc)/(2*(m-1)*m)
    V=np.cov(dL,rowvar=False)/m+np.cov(dU,rowvar=False)/N
    pool=np.r_[labelled,aux];dummy=np.full((len(pool),6,3),1/3)
    H=build(data,pool,dummy).hessian(pilot);inverse=np.linalg.inv(H)
    top=float(np.trace(inverse@cross@inverse));bottom=float(np.trace(inverse@V@inverse))
    assert bottom>0
    raw=top/bottom
    return float(np.clip(raw,0,1)),raw

def metrics(f,y,beta):
    u=f@beta;logp=u-logsumexp(u,axis=2,keepdims=True);prob=np.exp(logp)
    losses=-np.take_along_axis(logp,y[...,None],axis=2)[...,0]
    return dict(heldout_logloss=float(losses.mean()),heldout_Brier=float(np.sum((prob-np.eye(3)[y])**2,axis=2).mean()),heldout_accuracy=float((prob.argmax(2)==y).mean()),coefficient_norm=float(np.linalg.norm(beta)),cost_positive=bool(beta[-1]>0))

def main(path,out):
    plan=json.loads((BASE/'plan.json').read_text());data=load(path,plan);out.mkdir(parents=True,exist_ok=True)
    if (out/'status.json').exists():raise RuntimeError('Use a new output directory; never overwrite a partially observed run.')
    train,test,pool=[data[k] for k in ['train','test','pool']];t=data['teacher'][train].reshape(-1,3,33);ty=data['y'][train].ravel()
    teacher_loss=Loss(t,np.full(len(t),1/len(t)),np.mean(t[np.arange(len(t)),ty],axis=0),plan['ridge_penalty'])
    betaT,check=fit_loss(teacher_loss,np.zeros(33));assert check['fit']=='passed_score'
    g=softmax(data['teacher']@betaT,axis=2)
    identity=lambda ix:hashlib.sha256(np.asarray(data['ids'][ix],dtype='<i8').tobytes()).hexdigest()
    write(out/'preflight.json',dict(source_sha256=sha(path),plan_sha256=sha(BASE/'plan.json'),complete_respondents=298,excluded_partial_respondents=1,training_people=100,evaluation_people=60,pool_people=138,target_parameters=11,teacher_parameters=33,distinct_menu_designs=len(data['design']),split_sha256={k:identity(data[k]) for k in ['train','test','pool']},teacher_check=check,teacher_coefficients=betaT.tolist(),feature_names=NAMES,scope='Observed human choices; fixed-pool allocation audit. No generated respondents or Gaussian-boundary validation.'))
    records=[];certificates=[];betas=[];rng=np.random.default_rng(plan['allocation_seed']);eye=np.eye(3)
    for trial in range(plan['allocations']):
        order=rng.permutation(pool)
        for m in plan['labelled_respondent_budgets']:
            L=order[:m];U=order[m:];human=build(data,L,eye[data['y'][L]]);ridge=Loss(human.f,human.w,human.b,plan['ridge_penalty'])
            pilot,pilotcheck=fit_loss(ridge,np.zeros(11));assert pilotcheck['fit']=='passed_score'
            power,raw=cluster_power(data,L,U,g,pilot)
            all_people=np.r_[train,L];all_human=build(data,all_people,eye[data['y'][all_people]])
            fullridge=Loss(all_human.f,all_human.w,all_human.b,plan['ridge_penalty'])
            pooled=build(data,pool,np.where(np.isin(pool,L)[:,None,None],eye[data['y'][pool]],g[pool]))
            losses={'labelled_MLE':human,'labelled_ridge':ridge,'all_human_MLE':all_human,'all_human_ridge':fullridge,'corrected_unit':corrected(data,L,U,g,1.),'corrected_half':corrected(data,L,U,g,.5),'corrected_cluster_power':corrected(data,L,U,g,power),'pooling':pooled}
            for method,loss in losses.items():
                sid=len(records);state=dict(support='ridge_coercive') if loss.ridge else support(loss)
                direction=state.pop('direction',None)
                row=dict(record=sid,allocation=trial,labelled_people=m,auxiliary_people=138-m,labelled_menus=6*m,total_human_labels_with_training=100+m,method=method,power=1. if method=='corrected_unit' else .5 if method=='corrected_half' else power if method=='corrected_cluster_power' else None,raw_cluster_power=raw if method=='corrected_cluster_power' else None,**state)
                if direction is not None:certificates.append(dict(record=sid,direction=direction))
                if method=='labelled_ridge':beta=pilot;fitcheck=pilotcheck
                elif state['support'] in ['interior','ridge_coercive']:beta,fitcheck=fit_loss(loss,pilot)
                else:beta=None;fitcheck=dict(fit='unavailable_by_support')
                row.update(fitcheck)
                if beta is not None and fitcheck['fit']=='passed_score':
                    row.update(metrics(data['f'][test],data['y'][test],beta));betas.append(dict(record=sid,beta=beta.tolist()))
                records.append(row)
        if (trial+1)%25==0:
            write(out/'status.json',dict(state='running',allocations_complete=trial+1,planned=plan['allocations'],records=len(records)))
            print(json.dumps(dict(allocations_complete=trial+1,records=len(records))),flush=True)
    frame=pd.DataFrame(records);frame.to_csv(out/'method_records.csv',index=False)
    write(out/'coefficients.json',betas);write(out/'exterior_certificates.json',certificates)
    summaries=[];paired=[];fallback=[]
    for (m,method),part in frame.groupby(['labelled_people','method'],sort=True):
        good=part[part.fit.eq('passed_score')]
        summaries.append(dict(labelled_people=int(m),method=method,allocations=len(part),available=len(good),support_counts=part.support.value_counts().to_dict(),failed_score=int(part.fit.eq('failed_score').sum()),mean_logloss=float(good.heldout_logloss.mean()) if len(good) else None,median_logloss=float(good.heldout_logloss.median()) if len(good) else None,mean_Brier=float(good.heldout_Brier.mean()) if len(good) else None,positive_cost_fits=int(good.cost_positive.sum()) if len(good) else 0))
    for m,part in frame.groupby('labelled_people'):
        pivot=part.pivot(index='allocation',columns='method',values='heldout_logloss')
        for method in plan['methods']:
            for baseline in ['labelled_ridge','all_human_ridge']:
                if method==baseline:continue
                diff=(pivot[method]-pivot[baseline]).dropna()
                paired.append(dict(labelled_people=int(m),method=method,baseline=baseline,shared_allocations=len(diff),mean_difference=float(diff.mean()) if len(diff) else None,allocation_MCSE=float(diff.std(ddof=1)/np.sqrt(len(diff))) if len(diff)>1 else None))
        for method in ['corrected_unit','corrected_half','corrected_cluster_power']:
            value=pivot[method].fillna(pivot['labelled_ridge']);diff=value-pivot['labelled_ridge']
            fallback.append(dict(labelled_people=int(m),method=method,fallback_allocations=int(pivot[method].isna().sum()),all_allocations=len(value),mean_logloss=float(value.mean()),paired_excess_over_labelled_ridge=float(diff.mean()),allocation_MCSE=float(diff.std(ddof=1)/np.sqrt(len(diff)))))
    write(out/'summary.json',summaries);pd.DataFrame(paired).to_csv(out/'paired_loss.csv',index=False);pd.DataFrame(fallback).to_csv(out/'fallback_loss.csv',index=False)
    assert len(frame)==500*4*8
    write(out/'status.json',dict(state='complete',allocations_complete=500,method_records=len(frame),exterior_certificates=len(certificates),coefficient_records=len(betas),source_sha256=sha(path),plan_sha256=sha(BASE/'plan.json'),analysis_sha256=sha(Path(__file__))))
    print(json.dumps(dict(state='complete',method_records=len(frame),exterior_certificates=len(certificates),unresolved=int(frame.support.eq('unresolved').sum()))),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    with threadpool_limits(limits=1):main(args.input,args.out)
