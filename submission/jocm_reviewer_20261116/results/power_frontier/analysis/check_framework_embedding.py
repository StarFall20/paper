"""Exact coefficient comparison with the published pooled-score formula."""
from fractions import Fraction as F
from pathlib import Path
import json

rows=[]
for m,N in [(2,7),(4,4),(7,2)]:
    for lam in [F(1,4),F(1,2),F(3,4),F(1)]:
        alpha=(1+F(m,N))*lam
        pooled_label=alpha*(F(1,m+N)-F(1,m))
        pooled_aux=alpha*F(1,m+N)
        assert pooled_label==-lam/m and pooled_aux==lam/N
        rows.append(dict(m=m,N=N,power=str(lam),human_label_coefficient=str(F(1,m)),
                         prediction_label_coefficient=str(pooled_label),prediction_aux_coefficient=str(pooled_aux)))
out=Path(__file__).resolve().parents[1]/'results/framework_embedding.json'
assert not out.exists()
out.write_text(json.dumps(dict(state='passed',cases=rows,
    meaning='Exact scalar rescaling embeds the corrected score in the pooled-score framework for every coefficient and every common-coefficient menu design.',
    exclusions='No comparison of their learned estimator, no growing-dimension extension of their theorem, and no claim that categorical probability-vector use is new.'),indent=2)+'\n')
print(json.dumps(dict(state='passed',exact_fraction_cases=len(rows))))
