"""Independent scalar/Decimal reconstruction; no primary code or fitting import."""
from pathlib import Path
from decimal import Decimal,localcontext
from math import fsum,isfinite,floor
import argparse,csv,hashlib,json
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return list(csv.DictReader(p.open()))
def quantile(x,numerator):
    # Exact rational probability numerator/20 with ordinary order interpolation.
    a=sorted(x);k,rem=divmod((len(a)-1)*numerator,20)
    return a[k] if rem==0 else a[k]*(1-rem/20)+a[k+1]*(rem/20)
def stats(x):
    z=[v for v in x if isfinite(v)];r=dict(ratios_attempted=len(x),finite_numeric_ratios=len(z),nonfinite_numeric_ratios=len(x)-len(z))
    if z:r.update(mean=fsum(z)/len(z),**{name:quantile(z,q) for name,q in [('q00',0),('q05',1),('q25',5),('q50',10),('q75',15),('q95',19),('q100',20)]},absolute_ratio_gt_price_span=sum(abs(v)>12.5 for v in z))
    return r
def check_stats(row,expected,errors):
    for key,value in expected.items():
        if isinstance(value,int):assert int(row[key])==value,(key,row,value)
        else:
            error=abs(float(row[key])-value)/(1+abs(value));errors.append(error);assert error<3e-12,(key,row[key],value)
def main(root,out):
    out.mkdir(parents=True,exist_ok=True);plan=json.loads((root/'plan.json').read_text());ref=root/'reference';res=root/'results'
    for n,key in [('method_records.csv','input_records_sha256'),('coefficients.json','input_coefficients_sha256'),('verification.json','independent_primary_verification_sha256')]:assert sha(ref/n)==plan[key]
    old=read(ref/'method_records.csv');coef=json.loads((ref/'coefficients.json').read_text());beta={r['record']:r['beta'] for r in coef};assert len(old)==16000 and len(beta)==15418
    assert set(beta)=={int(r['record']) for r in old if r['fit']=='passed_score'}
    indexed={};price={};ratios={};max_compensation=0.
    with localcontext() as ctx:
        ctx.prec=60
        for r in old:
            i=int(r['record']);indexed[(int(r['labelled_people']),r['method'],int(r['allocation']))]=i
            if i not in beta:continue
            b=beta[i];cost=Decimal(str(b[10]))/Decimal(10);price[i]=float(cost)
            assert (r['cost_positive']=='True')==(cost>0)
            for name,j in plan['attributes'].items():
                if cost==0:continue
                ratio=-Decimal(str(b[j]))/cost;ratios[(i,name)]=float(ratio)
                residual=abs(Decimal(str(b[j]))+cost*ratio);max_compensation=max(max_compensation,float(residual))
    assert len(indexed)==16000
    errors=[];record_rows=read(res/'price_and_Bere_records.csv');assert len(record_rows)==16000
    for row in record_rows:
        i=int(row['record']);assert (row['ratio_eligible']=='True')==(i in price and price[i]<0)
        if i in price and price[i]!=0:
            val=ratios[(i,'bere')];e=abs(float(row['Bere_algebraic_ratio'])-val)/(1+abs(val));errors.append(e);assert e<3e-12
    availability=read(res/'availability.csv');assert len(availability)==32
    for row in availability:
        m=int(row['labelled_people']);method=row['method'];ids=[indexed[(m,method,t)] for t in range(500)];valid=[i for i in ids if i in price]
        counts=dict(allocations=500,available_fits=len(valid),unavailable_fits=500-len(valid),negative_price=sum(price[i]<0 for i in valid),positive_price=sum(price[i]>0 for i in valid),exact_zero_price=sum(price[i]==0 for i in valid))
        for k,v in counts.items():assert int(row[k])==v,(k,row)
    checked=0
    for file,negative in [('negative_price_attribute_ratios.csv',True),('all_signed_algebraic_ratios.csv',False)]:
        tables=read(res/file);assert len(tables)==256
        for row in tables:
            m=int(row['labelled_people']);method=row['method'];attr=row['attribute'];ids=[indexed[(m,method,t)] for t in range(500)];ids=[i for i in ids if i in price and (price[i]<0 if negative else price[i]!=0)]
            check_stats(row,stats([ratios[(i,attr)] for i in ids]),errors);checked+=1
    paired=read(res/'paired_Bere_differences.csv');assert len(paired)==64
    for row in paired:
        m=int(row['labelled_people']);vals=[]
        for t in range(500):
            i=indexed[(m,row['method'],t)];j=indexed[(m,row['comparator'],t)]
            if i in price and j in price and price[i]<0 and price[j]<0:vals.append(ratios[(i,'bere')]-ratios[(j,'bere')])
        assert int(row['paired_negative_price_fits'])==len(vals);check_stats(row,stats(vals),errors)
    result=dict(state='passed',input_method_records_checked=16000,finite_coefficient_vectors_checked=15418,scalar_attribute_ratios_checked=len(ratios),availability_rows_checked=32,attribute_summary_rows_checked=checked,paired_summary_rows_checked=64,max_scaled_numeric_discrepancy=max(errors),max_60_digit_compensation_identity_residual=max_compensation,primary_code_or_model_routine_imported=False,independent_calculation='Decimal conversion to per-GBP price slope, scalar compensation equation, math.fsum means and exact rational order-statistic interpolation.',source_hashes={str(p.relative_to(root)):sha(p) for p in [root/'plan.json',root/'analysis/compute_ratios.py',Path(__file__),*sorted(res.glob('*.csv')),res/'status.json']})
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_hashes'}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.root,a.out)
