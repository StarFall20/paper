"""Download the pinned public official package; verify every file before use."""
from pathlib import Path
import argparse,hashlib,json,urllib.request
def main(out):
    root=Path(__file__).resolve().parents[1];m=json.loads((root/'upstream_source_manifest.json').read_text());out.mkdir(parents=True,exist_ok=True)
    for r in m['files']:
        rel=Path(r['path']);assert not rel.is_absolute() and '..' not in rel.parts and rel.parts[0]=='ppi_py'
        target=out/rel;url=f"https://raw.githubusercontent.com/{m['repository']}/{m['commit']}/{r['path']}"
        if target.exists():b=target.read_bytes()
        else:
            with urllib.request.urlopen(url,timeout=60) as response:b=response.read()
        assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
        assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==r['git_blob_sha1']
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b)
    print(json.dumps(dict(verified_files=len(m['files']),commit=m['commit'],downloaded_code_executed_by_this_downloader=False)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
