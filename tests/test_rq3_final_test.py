"""Final-test authorization, independent-unit arithmetic and frozen inference checks."""
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.rq3_final_test import (aligned_labels, endpoint_summary, evaluate_arrays,
    predict_saved, replay_validation, run_seed, seed_interval, summarize)
from src.rq3_paired_data import write_dataset
from src.rq3_prediction_pilot import run_pilot
from src.rq3_lstm_pilot import run_seed as run_lstm_seed, summarize as summarize_lstm
from src.rq3_release_audit import sha256


def config():
    return yaml.safe_load(Path('configs/rq3_final_test_20261021.yaml').read_text())


def test_explicit_authorization_before_any_test_io(tmp_path):
    c = config()
    with pytest.raises(ValueError, match='requires explicit'):
        predict_saved(tmp_path, tmp_path, tmp_path, c, 42, 'test')
    with pytest.raises(ValueError, match='explicit'):
        aligned_labels(tmp_path, 42, np.array([]), c['targets'])
    with pytest.raises(ValueError, match='explicit'):
        run_seed(tmp_path, tmp_path, tmp_path, c, tmp_path / 'out', '0' * 40, '0' * 40, 42)
    assert not (tmp_path / 'out').exists()


def test_seed_interval_hand_calculation_and_duplicate_guard():
    actual = seed_interval([1., 2., 3., 4., 5.], .95)
    # mean=3, sample variance=2.5, t(.975,4)=2.776445105.
    half = 2.776445105 * np.sqrt(.5)
    assert abs(actual['ci_low'] - (3 - half)) < 1e-8
    assert abs(actual['ci_high'] - (3 + half)) < 1e-8
    assert actual['data_repeats'] == 5 and actual['df'] == 4
    c = config()
    rows = [dict(seed=s, architecture='lstm', comparison='B3-B2', target='arrival_proxy', scope='suite_equal_weight', mae_difference=float(s - 41)) for s in c['seeds']]
    summary = endpoint_summary(rows, c)[0]
    assert summary['primary'] is True and summary['data_repeats'] == 5
    with pytest.raises(ValueError, match='duplicate or missing'):
        endpoint_summary(rows + [rows[0]], c)


def test_paired_views_are_averaged_inside_seed():
    c = config()
    dc = yaml.safe_load(Path('configs/rq3_paired_data.yaml').read_text())
    ids = np.array(['a', 'b'])
    y = np.zeros((2, 2))
    pred = np.ones((2, 10, 7, 2, 2))
    # B3 errors exceed B2 by 0,1,...,9 within each seed.
    pred[:, :, 3] += np.arange(10)[None, :, None, None]
    events = [dict(sequence_id=i, channel='0', start='0', end_exclusive='4') for i in ids]
    rows, paired, endpoints = evaluate_arrays(pred, y, ids, np.ones((2, 118), dtype=int), events, c, dc, 42)
    selected = [r for r in endpoints if r['architecture']=='lstm' and r['comparison']=='B3-B2' and r['target']=='arrival_proxy']
    assert {r['scope']: r['mae_difference'] for r in selected} == {'original': 0., 'pressure_equal_weight': 5., 'suite_equal_weight': 4.5}
    assert all(r['batches'] == 2 for r in selected)
    assert len(selected) == 3


def test_target_ids_must_align(tmp_path):
    p = tmp_path / 'seed_42/test'
    p.mkdir(parents=True)
    (p / 'targets.csv').write_text('sequence_id,arrival_proxy,null_proxy\na,1,2\n')
    with pytest.raises(ValueError, match='ID mismatch'):
        aligned_labels(tmp_path, 42, np.array(['wrong']), ['arrival_proxy', 'null_proxy'], execute_frozen_test=True)


def test_isolated_final_integration_uses_frozen_models_only(tmp_path, monkeypatch):
    dc = yaml.safe_load(Path('configs/rq3_paired_data.yaml').read_text())
    dc.update(seeds=[42, 43], dataset_id='isolated_final_test_fixture')
    dc['counts'] = dict.fromkeys(dc['counts'], 16)
    data, ridge, lstm, output = [tmp_path / n for n in ('data', 'ridge', 'lstm', 'final')]
    write_dataset(dc, data)
    rc = yaml.safe_load(Path('configs/rq3_ridge_pilot_20261019.yaml').read_text())
    rc.update(seeds=dc['seeds'], dataset_id=dc['dataset_id'], data_manifest_sha256=sha256(data / 'manifest.json'))
    run_pilot(data, rc, ridge, '0' * 40)
    lc = yaml.safe_load(Path('configs/rq3_lstm_pilot_20261020.yaml').read_text())
    lc.update(seeds=dc['seeds'], dataset_id=dc['dataset_id'], data_manifest_sha256=sha256(data / 'manifest.json'),
              ridge_manifest_sha256=sha256(ridge / 'manifest.json'), hidden_size=4, epochs=2, batch_size=8)
    for seed in dc['seeds']:
        run_lstm_seed(data, ridge, lc, lstm, '0' * 40, seed)
    summarize_lstm(lstm, lc)
    c = config()
    c.update(seeds=dc['seeds'], dataset_id=dc['dataset_id'], data_manifest_sha256=sha256(data/'manifest.json'),
             ridge_manifest_sha256=sha256(ridge/'manifest.json'), lstm_manifest_sha256=sha256(lstm/'manifest.json'))
    import src.rq3_prediction_pilot as rp
    import src.rq3_lstm_pilot as lp
    import src.rq3_final_test as final
    def forbidden(*args, **kwargs):
        raise AssertionError('fitting forbidden during final evaluation')
    for name in ('fit_preprocessor', 'fit_score', 'ridge_path', 'select_model'):
        monkeypatch.setattr(rp, name, forbidden)
    monkeypatch.setattr(lp, 'train_model', forbidden)
    loaded = []
    original_loader = final.load_inputs
    def tracked(root, seed, split, **kwargs):
        loaded.append(split)
        return original_loader(root, seed, split, **kwargs)
    monkeypatch.setattr(final, 'load_inputs', tracked)
    report = replay_validation(data, ridge, lstm, c, tmp_path/'replay.json')
    assert report['status'] == 'pass' and 'test' not in loaded
    original_labels = final.aligned_labels
    def after_prediction(root, seed, *args, **kwargs):
        assert (output / f'seed_{seed}/predictions.npz').exists()
        return original_labels(root, seed, *args, **kwargs)
    monkeypatch.setattr(final, 'aligned_labels', after_prediction)
    for seed in c['seeds']:
        result = run_seed(data, ridge, lstm, c, output, '0'*40, '0'*40, seed, execute_frozen_test=True)
        assert result['models_refitted'] == 0 and result['source_files_unchanged']
    meta = summarize(output, c)
    assert meta['unique_test_batches'] == 32 and meta['test_status'] == 'used'
    assert meta['primary_endpoint']['data_repeats'] == 2
    with pytest.raises(FileExistsError):
        run_seed(data, ridge, lstm, c, output, '0'*40, '0'*40, 42, execute_frozen_test=True)
    manifest = json.loads((output/'manifest.json').read_text())
    assert all(sha256(output/k) == h for k,h in manifest.items())
    from scripts.verify_rq3_final_test import verify
    assert verify(output, data, ridge, lstm)['status'] == 'pass'
