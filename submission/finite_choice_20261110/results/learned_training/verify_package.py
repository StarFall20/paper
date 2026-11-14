"""Check the complete supplied inventory and immutable scientific sources."""
from pathlib import Path
import hashlib,json,ast,gzip

ROOT=Path(__file__).resolve().parent
listed=set()
for line in (ROOT/'MANIFEST.sha256').read_text().splitlines():
    sha,name=line.split('  ',1);p=ROOT/name
    assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==sha,name
    listed.add(name)
actual={str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and p.name!='.DS_Store' and p!=ROOT/'MANIFEST.sha256' and not {'__pycache__','regenerated','regenerated_full'}.intersection(p.relative_to(ROOT).parts)}
assert actual==listed,dict(extra=sorted(actual-listed),missing=sorted(listed-actual))
sources=list((ROOT/'analysis').glob('*.py'));assert len(sources)==15
current=json.loads((ROOT/'source_record.json').read_text())
for f in sources:
    ast.parse(f.read_text())
    assert hashlib.sha256(f.read_bytes()).hexdigest()==current['sources'][f.name]
for case in ['attributes_only','proxy']:
    base=ROOT/'evidence'/case
    for name,sha in json.loads((base/'primary_source_hashes.json').read_text()).items():
        assert hashlib.sha256((ROOT/'analysis'/name).read_bytes()).hexdigest()==sha,name
    for index in base.rglob('directions.index.json'):
        info=json.loads(index.read_text());count=0
        for row in info['files']:
            raw=(index.parent/row['path']).read_bytes()
            assert hashlib.sha256(raw).hexdigest()==row['sha256']
            count+=len(json.loads(gzip.decompress(raw)))
        assert count==info['record_count']
print(json.dumps(dict(manifest_entries=len(listed),scientific_sources=len(sources),
    compressed_direction_shards_verified=True,raw_questionnaire_or_downloaded_inputs_present=False)))
