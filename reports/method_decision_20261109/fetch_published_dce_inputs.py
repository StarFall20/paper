"""Fetch official public inputs, verify exact archived bytes, omit private DH36."""
from pathlib import Path
from urllib.request import urlopen
import argparse,hashlib,io,zipfile


def read(url):
    with urlopen(url,timeout=90) as r: return r.read()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out-dir',type=Path,required=True); args=ap.parse_args(); args.out_dir.mkdir(parents=True,exist_ok=True)
    route=read('https://apollochoicemodelling.com/files/examples/data/apollo_swissRouteChoiceData.csv')
    if hashlib.sha256(route).hexdigest()!='80ed1a1fd213d2baafbe946e2b1cd4740ce1f73374edca12f4d562cdd70bd028': raise ValueError('Official route input changed; review source version before analysis')
    archive=read('https://eml.berkeley.edu/Software/source_code/train_mxlmsl_06.zip')
    with zipfile.ZipFile(io.BytesIO(archive)) as z: vehicle=z.read('data.txt')
    if hashlib.sha256(vehicle).hexdigest()!='81d87eb6008bab4b96ed6384038aaa7db9a3cb93aeba69ac15fd91e56f80575f': raise ValueError('Official vehicle input changed; review source version before analysis')
    (args.out_dir/'apollo_swissRouteChoiceData.csv').write_bytes(route); (args.out_dir/'vehicle_data.txt').write_bytes(vehicle)
    print('Official public inputs verified; source usage and citation conditions apply.')

if __name__=='__main__': main()
