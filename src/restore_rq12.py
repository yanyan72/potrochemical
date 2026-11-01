"""Restore all 18 published research archives and verify extracted file hashes."""
from __future__ import annotations
import argparse,hashlib,json,zipfile
from pathlib import Path


def restore(archives: Path,output: Path) -> dict:
    """Recover a unique directory from a checked archive index, without overwrite."""
    index=json.loads((archives/'archive_index.json').read_text());output.mkdir(parents=True,exist_ok=False);files=0
    for item in index:
        p=archives/Path(item['path']).name
        if hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('archive hash mismatch: '+str(p))
        with zipfile.ZipFile(p) as z:
            assert z.testzip() is None
            for name in z.namelist():
                path=Path(name)
                if path.is_absolute() or '..' in path.parts:raise ValueError('invalid archive path')
                if (output/path).exists():raise FileExistsError(output/path)
            files+=len(z.namelist());z.extractall(output)
    checked=0
    for p in output.glob('*/manifest.json'):
        for rel,h in json.loads(p.read_text()).items():
            assert hashlib.sha256((p.parent/rel).read_bytes()).hexdigest()==h,rel;checked+=1
    result={'archives':len(index),'files':files,'manifest_items_verified':checked,'all_passed':True}
    print(json.dumps(result));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archives',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();restore(a.archives,a.output)
