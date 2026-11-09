"""Download pinned, openly released numeric data; verify each Git blob hash.

Lu, Wang, Zhang and Zhang, generative-augmented-inference. Code MIT; data
CC-BY-4.0 according to their pinned README. No downloaded script is executed.
"""
from pathlib import Path
import argparse, hashlib, json, urllib.request


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).resolve().parents[1]/'reports/public_conjoint_audit_20261109/source_manifest.json')
    args=parser.parse_args();manifest=json.loads(args.manifest.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    for item in manifest['files']:
        path=Path(item['path'])
        assert not path.is_absolute() and '..' not in path.parts
        url=item['url']
        assert url.startswith('https://raw.githubusercontent.com/'+manifest['repository']+'/'+manifest['commit']+'/')
        destination=args.out/path
        if destination.exists(): raw=destination.read_bytes()
        else:
            request=urllib.request.Request(url, headers={'User-Agent':'public-choice-reanalysis/1.0'})
            with urllib.request.urlopen(request,timeout=90) as response: raw=response.read()
        assert len(raw)==item['size'], path
        digest=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        assert digest==item['git_blob_sha1'], path
        destination.parent.mkdir(parents=True,exist_ok=True)
        if not destination.exists(): destination.write_bytes(raw)
    (args.out/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'verified_files':len(manifest['files']), 'executed_downloaded_code':False}))


if __name__=='__main__': main()
