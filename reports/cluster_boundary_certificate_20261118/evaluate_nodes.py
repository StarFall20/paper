"""Deterministic product-normal node values with original-quadratic dual gaps."""
from pathlib import Path
import argparse,concurrent.futures,itertools,json,time,os
import numpy as np

def one_job(inputs,model,k,out):
    started=time.time();d=json.loads(Path(inputs).read_text());quant=d['quantizer'];n=quant['bins'];total=n**3
    nodes=np.array(quant['nodes']);probs=np.array(quant['probabilities']);m=d['models'][model]
    cov=np.array(m['covariance']);chol=np.array(m['cholesky']);scale=k/64
    faces=[]
    for pat in itertools.product([-1,0,1],repeat=3):
        fix=np.flatnonzero(np.array(pat));free=np.flatnonzero(np.array(pat)==0)
        inv=np.linalg.inv(cov[np.ix_(free,free)]) if len(free) else None
        faces.append((pat,fix,free,inv))
    records=[]
    for state in itertools.product([0,1],repeat=2):
        zstate=np.array([0,*state]);g=.25+.5*zstate;lo=-scale*g;hi=scale*(1-g)
        weighted=[];gapmax=0.;disagreement=0.;feasmax=0.;hmax=0.;umax=0.;rmax=0.
        for first in range(0,total,131072):
            ix=np.arange(first,min(first+131072,total));i=ix//(n*n);j=(ix//n)%n;kx=ix%n
            h=np.column_stack([nodes[i],nodes[j],nodes[kx]]);weight=probs[i]*probs[j]*probs[kx]
            linear=h@chol.T;best=np.zeros(len(ix));chosen=np.zeros_like(h)
            for pat,fix,free,inv in faces:
                u=np.zeros_like(h)
                if len(fix):u[:,fix]=[lo[t] if pat[t]<0 else hi[t] for t in fix]
                if len(free):
                    rhs=linear[:,free].copy()
                    if len(fix):rhs-=u[:,fix]@cov[np.ix_(fix,free)]
                    u[:,free]=rhs@inv
                np.clip(u,lo,hi,out=u)
                value=2*np.sum(linear*u,axis=1)-np.sum((u@cov)*u,axis=1)
                better=value>best;best[better]=value[better];chosen[better]=u[better]
            u0,u1,u2=chosen.T;z0,z1,z2=linear.T
            # Recompute the primal and dual without the face formula.
            primal=2*(z0*u0+z1*u1+z2*u2)-(cov[0,0]*u0*u0+cov[1,1]*u1*u1+cov[2,2]*u2*u2+2*cov[0,1]*u0*u1+2*cov[0,2]*u0*u2+2*cov[1,2]*u1*u2)
            residual=np.column_stack([z0-cov[0,0]*u0-cov[0,1]*u1-cov[0,2]*u2,z1-cov[1,0]*u0-cov[1,1]*u1-cov[1,2]*u2,z2-cov[2,0]*u0-cov[2,1]*u1-cov[2,2]*u2])
            gap=2*np.sum(np.maximum(residual*(lo-chosen),residual*(hi-chosen)),axis=1)
            gapmax=max(gapmax,float(gap.max()));disagreement=max(disagreement,float(np.max(abs(primal-best))))
            feasmax=max(feasmax,float(np.max(lo-chosen)),float(np.max(chosen-hi)))
            hmax=max(hmax,float(abs(h).max()));umax=max(umax,float(abs(chosen).max()));rmax=max(rmax,float(abs(residual).max()))
            weighted.append(float(np.dot(weight,primal)))
        assert feasmax==0 and gapmax<2**-36 and disagreement<2**-36
        assert hmax<8 and umax<.5 and rmax<32
        records.append(dict(prediction_bits=zstate.tolist(),probability_numerator=25 if not zstate.any() else 13,probability_denominator=64,node_primal_sum=sum(weighted),node_count=total,maximum_original_quadratic_dual_gap=gapmax,maximum_primal_recalculation_disagreement=disagreement,maximum_feasibility_violation=feasmax,maximum_normal_coordinate=hmax,maximum_raw_box_coordinate=umax,maximum_raw_quadratic_residual=rmax))
        print(model,k,'state',zstate.tolist(),'nodes',total,'dual_gap',gapmax,flush=True)
    value=sum(x['probability_numerator']/64*x['node_primal_sum'] for x in records)
    result=dict(model=model,scale_numerator=k,scale_denominator=64,scale=scale,normal_bins=n,auxiliary_node_value=value,states=records,seconds=time.time()-started,implementation_scope='All product-normal nodes; feasible primal and original-quadratic first-order dual bound, no fitted estimator or MC replication.')
    path=Path(out)/f'node_{model}_{k}.json';path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return dict(model=model,scale_numerator=k,auxiliary_node_value=value,seconds=result['seconds'])

def main():
    a=argparse.ArgumentParser();a.add_argument('--inputs',type=Path,required=True);a.add_argument('--out',type=Path,required=True);a.add_argument('--workers',type=int,default=4);p=a.parse_args();p.out.mkdir(parents=True,exist_ok=True)
    jobs=[('A',41),('B',40),('B',41),('B',42)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=p.workers) as ex:
        fs=[ex.submit(one_job,str(p.inputs),m,k,str(p.out)) for m,k in jobs]
        for f in concurrent.futures.as_completed(fs):print(json.dumps(f.result()),flush=True)

if __name__=='__main__':main()
