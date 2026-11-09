"""Classic finite-support lack-of-fit audit for declared DCE menu representations.

No choice outcomes are read. Rank is structural visibility on the supplied
support, not evidence of sufficiency, power, feasibility or valid recruitment.
"""
from __future__ import annotations
import argparse,hashlib,json
from collections import defaultdict
from fractions import Fraction
from pathlib import Path


def rational_rows(rows):
    values=tuple(tuple(Fraction(str(x)) for x in row) for row in rows)
    if not values or not values[0] or any(len(r)!=len(values[0]) for r in values): raise ValueError('ragged or empty menu')
    return values


def audit(spec):
    mode=spec['candidate_representation']
    if mode not in ['absolute_menu_summary','utility_difference_vector']: raise ValueError('unknown representation')
    J=int(spec['choice_categories']); unique={}; totals=defaultdict(Fraction); dimensions=None
    if J<2: raise ValueError('at least two categories required')
    for entry in spec['menus']:
        raw=rational_rows(entry['raw_menu']); candidate=rational_rows(entry['candidate_menu']); w=Fraction(str(entry.get('weight',1)))
        if len(raw)!=J or len(candidate)!=J: raise ValueError('every complete menu must include all J alternatives')
        dims=(len(raw[0]),len(candidate[0]))
        if dimensions is None: dimensions=dims
        if dims!=dimensions: raise ValueError('inconsistent attribute/basis dimensions')
        if w<0: raise ValueError('negative design weight')
        key=tuple(x for row in candidate for x in row) if mode=='absolute_menu_summary' else tuple(x-y for row in candidate[:-1] for x,y in zip(row,candidate[-1]))
        if raw in unique and unique[raw]!=key: raise ValueError('candidate map not deterministic on repeated raw menu')
        unique[raw]=key; totals[raw]+=w
    groups=defaultdict(list)
    for raw,key in unique.items():
        if totals[raw]>0: groups[key].append(raw)
    if not groups: raise ValueError('no positive mass')
    M=sum(map(len,groups.values())); G=len(groups); menu_rank=M-G
    return {'positive_mass_raw_menus':M,'candidate_fibres':G,'non_singleton_fibres':sum(len(g)>1 for g in groups.values()),'menu_contrast_rank':menu_rank,'saturated_probability_contrast_df':menu_rank*(J-1),'null_probability_dimensions':G*(J-1),'full_probability_dimensions':M*(J-1),'candidate_representation':mode,'within_fibre_structural_contrasts_available':menu_rank>0,'candidate_sufficiency_verified':False,'statistical_power_verified':False,'feasible_original_design_universe_verified':False,'interpretation':'classic finite-support lack-of-fit dimensions for the declared map and positive-mass support','input_menu_count':len(spec['menus'])}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); args=ap.parse_args()
    spec=json.loads(args.input.read_text()); result=audit(spec); result['input_sha256']=hashlib.sha256(args.input.read_bytes()).hexdigest(); args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__': main()
