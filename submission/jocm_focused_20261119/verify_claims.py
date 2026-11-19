"""Recompute the compact counterexample and headline application counts."""
from argparse import ArgumentParser
from collections import Counter
import csv
from fractions import Fraction as F
from hashlib import sha256
import json
from pathlib import Path
from statistics import median


def rows(path):
    with path.open(newline='') as file:
        return list(csv.DictReader(file))


def exact_example():
    profile_counts = [(F(0), -1, 2), (F(0), 1, 6), (F(-1,2), 0, 1), (F(1,2), 0, 7)]
    labels = [(x,z,y) for x,z,count in profile_counts for y in [0]*(8-count)+[1]*count]
    anchor = [(F(0),z) for z in (-1,1) for _ in range(8)]
    designs = {
        'A': {F(-1,2):8, F(1,2):8},
        'B': {F(-1,2):4, F(1,2):4, F(-1):1, F(1):1, F(0):6},
        'C': {F(-1):2, F(1):2, F(0):12},
    }
    g = lambda z: F(1,2)+F(z,4)
    means, scores, hessians, margins = [], [], [], []
    for shape in designs.values():
        auxiliary = anchor+[(x,0) for x,n in shape.items() for _ in range(n)]
        assert len(labels)==len(auxiliary)==32
        corrected = [sum((y-g(z))*part(x) for x,z,y in labels)/32 +
                     sum(g(z)*part(x) for x,z in auxiliary)/32 for part in (lambda x:F(1),lambda x:x)]
        means.append(corrected)
        score = (F(1,2)-corrected[0],sum(x/2 for x,_ in auxiliary)/32-corrected[1])
        sandwich_rows = sorted((F(1,2)-g(z),(F(1,2)-g(z))*x) for x,z in auxiliary)
        scores.append((score,sandwich_rows))
        hessians.append([F(1,4),sum(x for x,_ in auxiliary)/128,sum(x*x for x,_ in auxiliary)/128])
        positive = sum(max(F(0),x) for x,_ in auxiliary)/32-corrected[1]
        negative = sum(max(F(0),-x) for x,_ in auxiliary)/32+corrected[1]
        margins.append((positive,negative))
    assert all(m==[F(1,2),F(3,32)] for m in means)
    assert scores[0]==scores[1]==scores[2] and scores[0][0]==(0,F(-3,32))
    assert all(h==[F(1,4),F(0),F(1,32)] for h in hessians)
    assert [a for a,b in margins]==[F(1,32),F(0),F(-1,32)] and min(b for a,b in margins)>0
    chosen = sum(y*x for x,_,y in labels)/32
    probability_score_a = (8*F(-1,2)*F(1,8)+8*F(1,2)*F(7,8))/32
    assert chosen == probability_score_a == means[0][1]
    assert all(0<n<8 for _,_,n in profile_counts)
    return {'labelled':32,'auxiliary_each':32,'positive_recession':[str(a) for a,b in margins],
            'score':['0','-3/32'],'hessian_diagonal':['1/4','1/32'],
            'third_derivatives_zero_at_reference':True,'same_reference_score_rows':True,'finite_slope':'2 log 7'}


def heldout_paths(records, identity, score, scale, a, b):
    grouped = {}
    for r in records:
        f = float(r[scale])
        if f in (a,b):
            grouped.setdefault(r[identity],{})[f] = float(r[score])
    assert all(set(v)=={a,b} for v in grouped.values())
    first,second=zip(*[(v[a],v[b]) for v in grouped.values()],strict=True)
    return [len(first),median(first),median(second),sum(y>x for x,y in zip(first,second,strict=True))]


def verify(root, source):
    evidence=set()

    def artifact(rel):
        p=root/rel; evidence.add(rel); return p

    panel=rows(artifact('results/public_panel_dce/results/method_records.csv'))
    results={}
    for budget in (5,10,20,40):
        task=[r for r in panel if int(r['labelled_people'])==budget]
        ordinary={r['allocation']:r['support'] for r in task if r['method']=='labelled_MLE'}
        unit={r['allocation']:r['support'] for r in task if r['method']=='corrected_unit'}
        assert len(unit)==len(ordinary)==500 and set(unit)==set(ordinary)
        results[budget]=[sum(x=='interior' for x in unit.values()),sum(x=='outside' for x in unit.values()),
                         sum(v=='outside' and ordinary[k]=='interior' for k,v in unit.items())]
    assert [results[m] for m in (5,10,20,40)]==[[356,144,40],[479,21,18],[500,0,0],[500,0,0]]
    food=rows(artifact('results/dce_frontier_economic_path/results/paths.csv'))
    grouped={}
    for group in ('uncapped_positive','capped_low_budget'):
        a=[r for r in food if r['group']==group and int(r['labelled_people'])==5]
        n, lo, hi, up=heldout_paths(a,'frontier_record','heldout_logloss','relative_distance',.5,.1)
        grouped[group]=[n,round(lo,4),round(hi,4),up]
    assert grouped['uncapped_positive'][:3]==[87,1.3075,2.2208]
    assert grouped['capped_low_budget'][:3]==[356,.9004,.9264]

    swiss='results/swissmetro_corrected_path/results/'
    cases=rows(artifact(swiss+'allocations.csv'))
    paths=rows(artifact(swiss+'paths.csv'))
    positive=[r for r in cases if r['endpoint']=='positive_uncapped']
    assert len(cases)==576 and len(positive)==93 and sum(r['labelled_support']=='interior' for r in positive)==59
    swiss_table=[]
    for model in ('canonical_4','mode_specific_8'):
        for n in (2,5,10):
            ids={r['record'] for r in positive if r['specification']==model and int(r['labelled_people'])==n}
            result=heldout_paths([r for r in paths if r['record'] in ids], 'record','holdout_loss','fraction',.5,.9)
            swiss_table.append([model,n]+[result[0],round(result[1],3),round(result[2],3),result[3]])
    expected=[[25,1.967,3.002,24],[13,1.657,3.315,13],[3,.994,2.439,3],[20,2.160,3.704,18],[24,1.970,4.635,24],[8,1.389,2.922,8]]
    assert [r[2:] for r in swiss_table]==expected
    assert sum(r[-1] for r in swiss_table)==90
    ids={r['record'] for r in cases if r['specification']=='mode_specific_8' and r['labelled_people']=='5' and r['endpoint']=='unit_interior'}
    control=heldout_paths([r for r in paths if r['record'] in ids],'record','holdout_loss','fraction',.5,.9)
    assert control[0]==64 and [round(x,3) for x in control[1:3]]==[1.150,1.162]
    cost=rows(artifact(swiss+'label_cost/records.csv'))
    assert len(cost)==1152 and sum(r['corrected_available']=='False' for r in cost)==172
    assert all(int(r['individual_labels'])==600+int(r['budget']) for r in cost)
    assert all(float(r['matched_direct_loss'])<float(r['corrected_or_ridge_loss']) for r in cost)

    menu=json.loads(artifact('results/menu_distribution_transport/results.json').read_text())
    method=('unit','poststratified','composition_only'); menu_table=[]
    for n in (15,5):
        group=[r for r in menu if r['label_people']==n]
        menu_table.append([n,len(group)]+[sum(r[m]['state']=='interior' for r in group) for m in method])
    assert menu_table==[[15,120,115,114,120],[5,360,268,215,295]]
    main=[r for r in menu if r['label_people']==15]
    stress=Counter((r['unit']['state'],r['composition_only']['state']) for r in menu if r['label_people']==5)
    assert stress['outside','interior']==43 and stress['interior','outside']==16
    assert sum(r['unit']['state']=='outside' and r['poststratified']['state']=='outside' for r in main)==5
    difference=[round(sum(r[m]['loss']-r['ridge'] for r in main)/len(main),4) for m in method[1:]]
    assert difference==[.0640,.0827]

    tab=artifact('manuscript/tableB1.md').read_text()
    assert '| 160 | 0.4500 | 0.056 | 0.924 | 0.012 | 0.814 |' in tab
    reference=json.loads(artifact('results/fixed_power_identical_scores/verification.json').read_text())
    example=exact_example()
    assert reference['max_order']==33 and reference['design_cases']==30
    assert [v['recession_positive'] for v in reference['compact_case']['outcomes'].values()]==example['positive_recession']
    for n in ('0.924','0.012','90 of 93','21 of 500','all 1152','32 labelled and 32 auxiliary','0.0640 and 0.0827'):
        assert n in source,n
    return dict(status='passed',fixed_example=example,food_availability=results,food_loss_paths=grouped,
                swissmetro_table=swiss_table,swissmetro_control=control,swissmetro_cost_pairs=len(cost),
                language_model_table=menu_table,language_model_excess_loss=difference,
                evidence_hashes={n:sha256((root/n).read_bytes()).hexdigest() for n in sorted(evidence)})


if __name__=='__main__':
    cli=ArgumentParser(description=__doc__)
    for name in ('research','manuscript','out'):
        cli.add_argument('--'+name,required=True,type=Path)
    args=cli.parse_args()
    result=verify(args.research,args.manuscript.read_text())
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(f'Checked claims against {len(result["evidence_hashes"])} archived numerical sources and an exact reconstruction.')
