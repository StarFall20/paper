"""Independent numerical checks of saturated conditional score and LR."""
import numpy as np
from scipy.special import xlogy
from fibre_baseline_comparison import table_statistics, conditional_tests
from support_complete_fibre_benchmark import generate

rng=np.random.default_rng(1234)
for k in (2,3,7):
    for _ in range(100):
        n=rng.integers(1,60,k); a=np.array([rng.integers(0,j+1) for j in n])
        N=n.sum(); M=a.sum()
        sc,lr=table_statistics(n,a)
        if M in (0,N):
            assert sc[0]==0 and lr[0]==0;continue
        w=n/N; u=a-n*M/N
        covariance=M*(N-M)/(N-1)*(np.diag(w)-np.outer(w,w))
        expected_score=u @ np.linalg.pinv(covariance) @ u
        p0=M/N; phat=a/n
        alternative_ll=np.sum(xlogy(a,phat)+xlogy(n-a,1-phat))
        null_ll=np.sum(xlogy(a,p0)+xlogy(n-a,1-p0))
        assert np.allclose(sc[0],expected_score,rtol=1e-10,atol=1e-10)
        assert np.allclose(lr[0],2*(alternative_ll-null_ll),rtol=1e-10,atol=1e-10)
        perm=rng.permutation(k)
        sc2,lr2=table_statistics(n[perm],a[perm])
        assert np.allclose(sc,sc2) and np.allclose(lr,lr2)
for seed in range(5):
    pts,g,profile,y=generate(seed,200,'null',0)
    p=conditional_tests(pts,g,profile,y,199,seed)
    assert p['support_complete']==p['generalized_cmh']
    assert all(0<q<=1 for q in p.values())
print('300 independent covariance/likelihood identities and relabelling checks passed; score/CMH p-values identical')
