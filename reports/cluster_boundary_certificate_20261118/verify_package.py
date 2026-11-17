"""Data-free inventory/source binding; the arithmetic verifier is separate."""
from pathlib import Path
import hashlib,json,re
R=Path(__file__).resolve().parent
listed=set()
for line in (R/'MANIFEST.sha256').read_text().splitlines():
    h,n=line.split('  ',1);assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n;listed.add(n)
actual={str(f.relative_to(R)) for f in R.rglob('*') if f.is_file() and f.name not in ['MANIFEST.sha256','.DS_Store'] and '__pycache__' not in f.parts}
assert actual==listed
d=json.loads((R/'results/enclosure.json').read_text());assert d['state']=='separated' and d['witness_separates']
assert d['model_A_boundary_lower']['lower']>.719 and d['model_B_boundary_upper']['upper']<.719
assert d['original_quadratic_node_checks']==33554432 and d['new_MC_or_fitted_models']==0
for n,h in d['sources_sha256'].items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n
for f in R.rglob('*.md'):
    for target in re.findall(r'\]\(([^)]+)\)',f.read_text()):
        if target.startswith(('http:','https:','#')):continue
        assert (f.parent/target.split('#',1)[0]).exists(),(str(f),target)
print(json.dumps(dict(state='passed',manifest_entries=len(listed),source_bindings=len(d['sources_sha256']),boundary_witness='719/1000',node_checks=d['original_quadratic_node_checks'],inventory_check_only=True)))
