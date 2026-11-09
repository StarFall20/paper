"""Fetch the eight frozen, openly licensed files listed in the preflight.

Downloads are never executed. Cached files are rehashed before use.
"""
from pathlib import Path
import argparse, concurrent.futures, hashlib, json, urllib.request


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    files=json.loads(a.manifest.read_text())['file_checks']
    for record in sorted({z['record'] for z in files}):
        assert record in ['15420092','23143985']
        target=a.out/(record+'_metadata.json')
        if not target.exists():
            with urllib.request.urlopen('https://zenodo.org/api/records/'+record,timeout=25) as response:
                metadata=json.load(response)
            target.write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
        metadata=json.loads(target.read_text())
        assert metadata['metadata']['license']['id']=='cc-by-4.0'
    def download(row):
        target=a.out/row['record']/row['file'];target.parent.mkdir(exist_ok=True)
        assert Path(row['file']).name==row['file']
        assert row['url'].startswith('https://zenodo.org/api/records/'+row['record']+'/files/')
        if target.exists():
            raw=target.read_bytes()
        else:
            with urllib.request.urlopen(row['url'],timeout=25) as response:
                raw=response.read()
        assert len(raw)==row['bytes']
        assert hashlib.md5(raw).hexdigest()==row['publisher_md5'].split(':')[1]
        assert hashlib.sha256(raw).hexdigest()==row['sha256']
        if not target.exists():target.write_bytes(raw)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download,files))
    (a.out/'download_receipt.json').write_text(json.dumps(files,indent=2)+'\n')
    print(json.dumps({'files_verified':len(files),'downloaded_code_executed':False}))


if __name__=='__main__':
    main()
