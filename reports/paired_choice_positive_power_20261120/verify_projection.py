"""Numerical check of the polygon expectation using Gaussian quasi-Monte Carlo."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc
from threadpoolctl import threadpool_limits


def point_distances(points, vertices):
    edge = np.roll(vertices,-1,axis=0)-vertices
    rel = points[:,None,:]-vertices[None,:,:]
    cross = edge[None,:,0]*rel[:,:,1]-edge[None,:,1]*rel[:,:,0]
    inside = np.all(cross >= 0,axis=1)
    t = np.clip(np.sum(rel*edge[None,:,:],axis=2)/np.sum(edge*edge,axis=1),0.,1.)
    dist = np.min(np.sum((rel-t[:,:,None]*edge[None,:,:])**2,axis=2),axis=1)
    dist[inside] = 0.
    return np.mean(dist)


def score(points,row):
    rho, weight, tau, u = (row[n] for n in ("correlation","power","auxiliary_ratio","maximizing_scale"))
    vals, vecs = np.linalg.eigh(np.array([[1,rho],[rho,1]]))
    root = (vecs*np.sqrt(vals))@vecs.T
    square = np.array([[-1,-1],[1,-1],[1,1],[-1,1]],dtype=float)@root
    center = np.array([[1,1],[1,-1]])@root
    if weight == 1.:
        label_dist = np.mean([np.mean(np.sum((points-u*s)**2,axis=1)) for s in center])
    else:
        label_dist = np.mean([point_distances(points,u*((1-weight)*square+s)) for s in center])
    u_dist = point_distances(points,u*weight/tau*square)
    return 2-label_dist+tau*(2-u_dist)


def main(path,out):
    rows=json.loads(path.read_text())
    points=ndtri(qmc.Sobol(d=2,scramble=True,seed=20261120).random_base2(m=17))
    results=[]
    for row in rows:
        q=float(score(points,row))
        exact=row["critical_slopes_per_labelled_person"]
        results.append(dict(power=row["power"],correlation=row["correlation"],
                            analytic_integration=exact,qmc_value=q,absolute_difference=abs(q-exact)))
    max_diff=max(r["absolute_difference"] for r in results)
    assert max_diff<2e-4
    report=dict(state="passed",grid_points=len(rows),quadrature_inputs=len(points),
                max_abs_difference=max_diff,scope="Independent projection evaluation at the selected deterministic scale",
                comparisons=results)
    out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"grid_points":len(rows),"max_abs_difference":max_diff}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundaries",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        main(args.boundaries,args.out)
