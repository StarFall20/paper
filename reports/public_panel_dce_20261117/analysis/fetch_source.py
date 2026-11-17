"""Fetch one explicitly licensed, pinned source workbook; never execute it."""
from pathlib import Path
import argparse,hashlib,urllib.request
URL='https://zenodo.org/api/records/15420092/files/COSCBB.xlsx/content'
SHA='447cb59fab8500639d9f5478dad4aeeeb64bc09f1d0d0848ae0219e1c42c88a0'
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raw=a.out.read_bytes()
    else:
        with urllib.request.urlopen(URL,timeout=60) as r:raw=r.read()
    assert len(raw)==541767 and hashlib.sha256(raw).hexdigest()==SHA
    assert hashlib.md5(raw).hexdigest()=='d1c07bc41035afab588ecd03acdc5abc'
    if not a.out.exists():a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_bytes(raw)
    print('Pinned CC-BY-4.0 source verified. Raw respondent records stay local.')
