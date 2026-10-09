"""Audit a published L9 array and generated locally D-efficient binary DCEs.

The raw menu has (time_A, cost_A, time_B, cost_B), each at levels 0,1,2.
The complete candidate vector is (time_A+cost_A, time_B+cost_B).
Time levels can be read as 20,30,40 min and fares as 2,3,4 currency units,
with a predeclared conversion of 10 min/unit. No field-data claim is made.
Exposure spectra use a unit response metric; rank is the exact finite
lack-of-fit dimension. D-efficiency uses a separate full raw-linear logit.
"""
from __future__ import annotations
import argparse, csv, itertools, os
import numpy as np
from scipy.special import expit
from support_complete_fibre_design import fibre_contrast_basis

# NIST/SEMATECH e-Handbook, Table L9, zero-based level coding.
# https://www.itl.nist.gov/div898/handbook/pri/section3/pri33a.htm
L9 = np.asarray([[0,0,0,0],[0,1,1,1],[0,2,2,2],
                 [1,0,1,2],[1,1,2,0],[1,2,0,1],
                 [2,0,2,1],[2,1,0,2],[2,2,1,0]])
MENUS = np.asarray(list(itertools.product(range(3), repeat=4)))
PHI = np.column_stack((MENUS[:,0]+MENUS[:,1], MENUS[:,2]+MENUS[:,3]))
H = (MENUS[:,0]-MENUS[:,1])-(MENUS[:,2]-MENUS[:,3])
DX = np.column_stack((np.ones(len(MENUS)), MENUS[:,0]-MENUS[:,2],
                      MENUS[:,1]-MENUS[:,3]))
LOOKUP = {tuple(x):i for i,x in enumerate(MENUS)}
GROUPS = [np.flatnonzero(np.all(PHI == z,axis=1)) for z in np.unique(PHI,axis=0)]


def audit(counts):
    w=counts/counts.sum(); spectra=[]; formula_rank=0; spread=0.; supported_fibres=0
    for ids in GROUPS:
        mass=w[ids].sum()
        if mass<=0: continue
        m=int(np.count_nonzero(w[ids])); supported_fibres+=1; formula_rank+=m-1
        cond=w[ids]/mass
        q=fibre_contrast_basis(len(ids))
        block=mass*q.T@(np.diag(cond)-np.outer(cond,cond))@q
        spectra.extend(np.linalg.eigvalsh(block))
        mean=cond@H[ids]
        spread+=mass*(cond@(H[ids]-mean)**2)
    eig=np.asarray(spectra); rank=int(np.sum(eig>1e-10))
    assert rank==formula_rank
    pos=eig[eig>1e-10]
    return dict(unique_menus=int(np.count_nonzero(counts)),supported_fibres=supported_fibres,
                exposure_rank=rank,full_universe_dimension=56,
                minimum_positive_eigenvalue=float(pos.min()) if len(pos) else 0.,
                relative_time_cost_departure_exposure=float(spread))


def information(counts,beta):
    p=expit(DX@beta); return DX.T@((counts/counts.sum()*p*(1-p))[:,None]*DX)


def d_exchange(beta,n,seed,rounds=100):
    """Multistart row-exchange local search; no global optimum assertion."""
    rng=np.random.default_rng(seed); rows=rng.integers(0,len(MENUS),n)
    info_by_menu=expit(DX@beta)*(1-expit(DX@beta))
    atoms=np.einsum('ni,nj,n->nij',DX,DX,info_by_menu)
    for _ in range(rounds):
        changed=False
        for row in rng.permutation(n):
            total=atoms[rows].sum(axis=0)-atoms[rows[row]]
            sign,ld=np.linalg.slogdet(total[None,:,:]+atoms)
            ld=np.where(sign>0,ld,-np.inf); chosen=int(np.argmax(ld))
            old=float(ld[rows[row]])
            if ld[chosen]>old+1e-10:
                rows[row]=chosen;changed=True
        if not changed:break
    return np.bincount(rows,minlength=len(MENUS))


def repair(counts):
    """Add one feasible menu maximizing exposure increase per added task."""
    candidates=[]
    for j in range(len(MENUS)):
        trial=counts.copy();trial[j]+=1; a=audit(trial)
        candidates.append((a['exposure_rank'],a['relative_time_cost_departure_exposure'],j))
    _,_,j=max(candidates);trial=counts.copy();trial[j]+=1
    return trial,j


def run(out='results/common_dce_design_exposure_audit.csv',menus_out='results/common_dce_design_menus.csv',starts=50):
    # Every pair of L9 columns contains each of the 9 level pairs once.
    for a,b in itertools.combinations(range(4),2):
        assert len(set(map(tuple,L9[:,[a,b]])))==9
    records=[]; allocations=[]
    priors=[np.array([0.,0.,0.]),np.array([0.,-.3,-.3]),np.array([0.,-.6,-.2])]
    def add(name,counts,beta,seed='',kind='generated'):
        rec=dict(design=name,kind=kind,seed=seed,prior=';'.join(map(str,beta)),tasks=int(counts.sum()),**audit(counts))
        I=information(counts,beta);rec['raw_linear_logdet']=float(np.linalg.slogdet(I)[1]);rec['raw_linear_information_rank']=int(np.linalg.matrix_rank(I))
        records.append(rec)
        for j in np.flatnonzero(counts):
            allocations.append(dict(design=name,menu_id=int(j),time_A=int(MENUS[j,0]),cost_A=int(MENUS[j,1]),time_B=int(MENUS[j,2]),cost_B=int(MENUS[j,3]),candidate_A=int(PHI[j,0]),candidate_B=int(PHI[j,1]),count=int(counts[j])))
    zero=priors[0]
    oa=np.bincount([LOOKUP[tuple(x)] for x in L9],minlength=len(MENUS))
    add('NIST_L9',oa,zero,kind='published_array_mapping')
    fixed,j=repair(oa);add('NIST_L9_plus_one',fixed,zero,kind='generated_repair')
    # Audit all 24 factor-to-menu column mappings, with fixed physical level order.
    for k,perm in enumerate(itertools.permutations(range(4))):
        counts=np.bincount([LOOKUP[tuple(x)] for x in L9[:,perm]],minlength=len(MENUS))
        add(f'L9_mapping_{k:02d}',counts,zero,kind='column_mapping_sensitivity')
    for k,beta in enumerate(priors):
        best=None
        for s in range(starts):
            seed=2026100900+100*k+s;counts=d_exchange(beta,12,seed)
            add(f'D_exchange_prior{k}_start{s:02d}',counts,beta,seed=seed)
            criterion=records[-1]['raw_linear_logdet']
            if best is None or criterion>best[0]:best=(criterion,counts,seed)
        fixed,j=repair(best[1]);add(f'D_exchange_prior{k}_best_plus_one',fixed,beta,seed=best[2],kind='generated_repair')
    add('full_3x3_menu_factorial',np.ones(len(MENUS),dtype=int),zero,kind='full_universe_reference')
    for path,rows in [(out,records),(menus_out,allocations)]:
        os.makedirs(os.path.dirname(path) or '.',exist_ok=True)
        with open(path,'w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    for k in range(3):
        subset=[r for r in records if r['design'].startswith(f'D_exchange_prior{k}_start')]
        print('prior',k,'starts',len(subset),'rank_range',min(r['exposure_rank'] for r in subset),max(r['exposure_rank'] for r in subset),'blind_fraction',np.mean([r['exposure_rank']==0 for r in subset]))
    for r in records:
        if r['design'] in ('NIST_L9','NIST_L9_plus_one') or r['kind']=='generated_repair':print(r)
    oa_rows=[r for r in records if r['kind']=='column_mapping_sensitivity']
    print('L9 mappings rank range',min(r['exposure_rank'] for r in oa_rows),max(r['exposure_rank'] for r in oa_rows))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',default='results/common_dce_design_exposure_audit.csv');parser.add_argument('--menus-out',default='results/common_dce_design_menus.csv');parser.add_argument('--starts',type=int,default=50);run(**vars(parser.parse_args()))
