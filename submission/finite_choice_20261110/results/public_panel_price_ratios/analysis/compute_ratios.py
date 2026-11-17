"""Describe all saved DCE coefficients; no refitting or response filtering."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def summary(values):
    values=np.asarray(values,float);good=np.isfinite(values);z=values[good]
    r=dict(ratios_attempted=len(values),finite_numeric_ratios=len(z),nonfinite_numeric_ratios=int((~good).sum()))
    if len(z):r.update(mean=float(z.mean()),**{f'q{int(q*100):02}':float(np.quantile(z,q,method='linear')) for q in [0,.05,.25,.5,.75,.95,1]},absolute_ratio_gt_price_span=int((abs(z)>12.5).sum()))
    return r
def main(root,out):
    out.mkdir(parents=True,exist_ok=True);plan=json.loads((root/'plan.json').read_text());ref=root/'reference'
    assert sha(ref/'method_records.csv')==plan['input_records_sha256'] and sha(ref/'coefficients.json')==plan['input_coefficients_sha256']
    assert sha(ref/'verification.json')==plan['independent_primary_verification_sha256']
    a=pd.read_csv(ref/'method_records.csv');b=json.loads((ref/'coefficients.json').read_text());assert len(a)==16000 and len(b)==15418
    mapping={r['record']:np.array(r['beta']) for r in b};assert len(mapping)==15418
    assert set(mapping)==set(a.loc[a.fit.eq('passed_score'),'record'])
    assert all(v.shape==(11,) and np.isfinite(v).all() for v in mapping.values())
    a['price_beta']=[mapping[i][10] if i in mapping else np.nan for i in a.record];a['bere_beta']=[mapping[i][2] if i in mapping else np.nan for i in a.record]
    a['ratio_eligible']=a.fit.eq('passed_score')&a.price_beta.lt(0)
    with np.errstate(divide='ignore',invalid='ignore',over='ignore'):a['Bere_algebraic_ratio']=-10*a.bere_beta/a.price_beta
    a.loc[a.price_beta.eq(0),'Bere_algebraic_ratio']=np.nan
    a[['record','allocation','labelled_people','method','support','fit','price_beta','bere_beta','ratio_eligible','Bere_algebraic_ratio']].to_csv(out/'price_and_Bere_records.csv',index=False)
    availability=[];negative=[];all_signed=[];pairs=[]
    for m in plan['budgets']:
        for method in plan['methods']:
            g=a[(a.labelled_people==m)&a.method.eq(method)];assert len(g)==500
            ids=g.loc[g.fit.eq('passed_score'),'record'].to_numpy();beta=np.array([mapping[i] for i in ids]);cost=beta[:,10];neg=cost<0;nonzero=cost!=0
            availability.append(dict(labelled_people=m,method=method,allocations=500,available_fits=len(ids),unavailable_fits=500-len(ids),negative_price=int(neg.sum()),positive_price=int((cost>0).sum()),exact_zero_price=int((cost==0).sum()),negative_price_available_fraction=float(neg.sum()/500)))
            for name,j in plan['attributes'].items():
                with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
                    nr=-10*beta[neg,j]/cost[neg];ar=-10*beta[nonzero,j]/cost[nonzero]
                base=dict(labelled_people=m,method=method,attribute=name,allocations=500,available_fits=len(ids),negative_price_count=int(neg.sum()))
                negative.append(dict(**base,**summary(nr)));all_signed.append(dict(**base,nonzero_price_count=int(nonzero.sum()),**summary(ar)))
        for comp in plan['paired_comparators']:
            baseline=a[(a.labelled_people==m)&a.method.eq(comp)].set_index('allocation')
            for method in plan['methods']:
                other=a[(a.labelled_people==m)&a.method.eq(method)].set_index('allocation');keep=baseline.ratio_eligible&other.ratio_eligible
                diff=other.loc[keep,'Bere_algebraic_ratio']-baseline.loc[keep,'Bere_algebraic_ratio'];assert np.isfinite(diff).all()
                pairs.append(dict(labelled_people=m,method=method,comparator=comp,allocations=500,paired_negative_price_fits=int(keep.sum()),**summary(diff.to_numpy())))
    pd.DataFrame(availability).to_csv(out/'availability.csv',index=False);pd.DataFrame(negative).to_csv(out/'negative_price_attribute_ratios.csv',index=False);pd.DataFrame(all_signed).to_csv(out/'all_signed_algebraic_ratios.csv',index=False);pd.DataFrame(pairs).to_csv(out/'paired_Bere_differences.csv',index=False)
    overview=dict(state='complete',original_method_records=16000,verified_finite_coefficients=15418,availability_rows=len(availability),negative_price_attribute_rows=len(negative),all_signed_attribute_rows=len(all_signed),paired_Bere_rows=len(pairs),formula='-10 * beta_attribute / beta_cost_GBP10',unit='GBP per 250g package',experimental_product_price_span=12.5,no_refitting=True,no_trim_or_winsorization=True,allocation_and_price_sign_denominators_preserved=True,descriptive_secondary_analysis=True,population_WTP_or_estimator_superiority_claim=False,plan_sha256=sha(root/'plan.json'),source_sha256=sha(Path(__file__)))
    (out/'status.json').write_text(json.dumps(overview,indent=2)+'\n');print(json.dumps(overview))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.root,a.out)
