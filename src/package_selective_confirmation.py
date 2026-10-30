"""Pack immutable result folders without changing archived source bytes."""
import argparse,hashlib,json,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('stage',choices=['training','evaluation']);p.add_argument('seed',type=int);a=p.parse_args()
root=Path('results/rq3_selective_confirmation_20261030_v1');folder=root/a.stage/f'seed_{a.seed}';manifest=json.loads((folder/'manifest.json').read_text())
for rel,h in manifest.items():assert hashlib.sha256((folder/rel).read_bytes()).hexdigest()==h
out=Path('releases/confirmation_20261030')/f'{a.stage}_seed_{a.seed}.zip'
with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for f in sorted(folder.rglob('*')):
        if f.is_file():z.write(f,str(f.relative_to(root)))
with zipfile.ZipFile(out) as z:
    assert z.testzip() is None
    for rel,h in manifest.items():assert hashlib.sha256(z.read(str(folder.relative_to(root)/rel))).hexdigest()==h
b=out.read_bytes();desc=dict(archive=out.name,stage=a.stage,seed=a.seed,files=len(manifest)+1,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),git_blob=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),root_directory=root.name,extraction='python -m zipfile -e ARCHIVE results/rq3_selective_confirmation_20261030_v1',contains='all checkpoint arrays, predictions, histories, provenance and local manifest' if a.stage=='training' else 'visible inputs, separate evaluation-only truth/targets, all predictions, events, diagnostics and local manifest')
out.with_suffix('.json').write_text(json.dumps(desc,indent=2)+'\n');print(json.dumps(desc))
