"""Download the two attributed public archives to a separate local directory."""
from pathlib import Path
from urllib.request import Request, urlopen
import argparse
from audit_released_workflow import CORE, OUTPUT, HASHES, digest

URLS = {
    CORE: 'https://zenodo.org/api/records/23140920/files/Estudo2_protocolo_e_codigo.zip/content',
    OUTPUT: 'https://zenodo.org/api/records/23143623/files/saida_confirmatorio_qwen3-30b-a3b-instruct-2507-q4_k_m.zip/content',
}

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    for name,url in URLS.items():
        target=a.out/name
        if target.exists():
            assert digest(target.read_bytes())==HASHES[name],target
            continue
        with urlopen(Request(url,headers={'User-Agent':'Mozilla/5.0 (research replication)'}),timeout=60) as r:
            data=r.read(4000001)
        assert len(data)<4000001 and digest(data)==HASHES[name],name
        target.write_bytes(data)
        print(name,len(data))
