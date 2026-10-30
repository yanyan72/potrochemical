"""Restore published ZIP contents and verify every archived SHA256 manifest."""
import argparse,hashlib,json,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('result_root',type=Path);a=p.parse_args();root=a.result_root
for archive in sorted((root/'archives').glob('*.zip')):
    info=json.loads(archive.with_suffix('.json').read_text());assert hashlib.sha256(archive.read_bytes()).hexdigest()==info['sha256']
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            rel=Path(entry.filename);assert not rel.is_absolute() and '..' not in rel.parts
            dest=root/rel;content=z.read(entry.filename)
            if dest.exists():assert dest.read_bytes()==content,str(dest)
            else:dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(content)
    folder=root/info['stage']/f"seed_{info['seed']}"
    for rel,h in json.loads((folder/'manifest.json').read_text()).items():assert hashlib.sha256((folder/rel).read_bytes()).hexdigest()==h
    print(archive.name,'verified')
