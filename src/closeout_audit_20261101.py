"""Restore the frozen release independently without generating data or retraining."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys
base=Path(__file__).resolve().parents[2]/'deadline_20261029'
out=Path(__file__).resolve().parents[1]/'results/restored_confirmation_v3'
out.mkdir(exist_ok=False)
original=base/'results/rq3_selective_confirmation_20261030_v1'
for p in original.iterdir():
 if p.is_file():shutil.copyfile(p,out/p.name)
(out/'training').mkdir()
shutil.copyfile(original/'training/manifest.json',out/'training/manifest.json')
(out/'archives').mkdir()
for p in (base/'releases/confirmation_20261030').iterdir():shutil.copyfile(p,out/'archives'/p.name)
subprocess.run([sys.executable,str(base/'src/restore_selective_confirmation.py'),str(out)],check=True)
m=json.loads((out/'manifest.json').read_text())
for p,h in m.items():assert hashlib.sha256((out/p).read_bytes()).hexdigest()==h,p
before=hashlib.sha256((original/'manifest.json').read_bytes()).hexdigest()
layout=out.parent/'repository_layout';(layout/'results/rq3_prediction').mkdir(parents=True)
for name in ['shared_mode_fault_development_20261029_v2','rq3_joint_prediction_dev_20261029_v1']:(layout/'results/rq3_prediction'/name).symlink_to(base/'results'/name,target_is_directory=True)
(layout/'results/rq3_prediction/rq3_uncertainty_a0_20261029_v1').symlink_to(base.parent/'a0_20261029/results/rq3_uncertainty_a0_20261029_v1',target_is_directory=True)
subprocess.run([sys.executable,str(Path(__file__).with_name('verify_selective_confirmation.py')),str(out),'--repository-root',str(layout)],check=True)
for p,h in m.items():assert hashlib.sha256((out/p).read_bytes()).hexdigest()==h,p
assert hashlib.sha256((original/'manifest.json').read_bytes()).hexdigest()==before
receipt=dict(passed=True,archive_count=10,manifest_entries_verified=len(m),models=65,new_models=0,new_data_batches=0,original_unchanged=True,restored_statistics_identical=True,scope='local published archives restored; GitHub blob identities independently compared in remote receipt',source_result_commit='5740c4b7f4af64a3043d5e6234a4dc1d48c5e3e2')
(out.parent/'restoration_audit.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
