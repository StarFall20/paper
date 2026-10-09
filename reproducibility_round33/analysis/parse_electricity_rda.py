"""Convert the public mlogit Electricity RDA to a flat CSV.

Raw RDA files stay outside the repository.  The conversion is deterministic
and records the package source and checksum in the provenance note.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pyreadr

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('rda'); ap.add_argument('--out',required=True)
    args=ap.parse_args(); result=pyreadr.read_r(args.rda)
    if 'Electricity' not in result: raise ValueError(f'expected Electricity object, found {list(result)}')
    df=result['Electricity']; Path(args.out).parent.mkdir(parents=True,exist_ok=True); df.to_csv(args.out,index=False)
    print(f'wrote {len(df)} rows and {len(df.columns)} columns to {args.out}')
if __name__=='__main__': main()
