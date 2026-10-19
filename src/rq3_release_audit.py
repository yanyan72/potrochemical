"""Supplement frozen RQ3 integrity checks with provenance and null-label checks."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .rq3_paired_data import SPLITS, audit_dataset


def sha256(path: Path) -> str:
    """Hash file bytes without modifying the artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_audit(root: Path, *, expected_dataset_id: str,
                 expected_source_sha256: str,
                 expected_manifest_sha256: str | None = None) -> dict[str, Any]:
    """Check external identities and replay independent null-label scalar streams.

    A caller must supply the expected source identity from a trusted record.
    Manifest pinning is optional for isolated semantic tests, required by the
    formal prediction configuration. This audit never computes predictions.
    """
    if expected_manifest_sha256 is not None and sha256(root / "manifest.json") != expected_manifest_sha256:
        raise ValueError("external manifest identity mismatch")
    c = yaml.safe_load((root / "config.yaml").read_text())
    meta = json.loads((root / "metadata.json").read_text())
    if meta["dataset_id"] != c["dataset_id"] or c["dataset_id"] != expected_dataset_id:
        raise ValueError("metadata dataset identity mismatch")
    if meta["data_source"] != "synthetic" or meta["source_sha256"] != expected_source_sha256:
        raise ValueError("metadata source identity mismatch")
    if meta.get("target_unit") != "arbitrary_proxy_unit" or meta.get("time_unit") != "abstract_step":
        raise ValueError("metadata units mismatch")
    report = audit_dataset(root)
    checked = 0
    for seed in c["seeds"]:
        for split in ("train", "val", "test"):
            with (root / f"seed_{seed}" / split / "targets.csv").open(newline="") as f:
                rows = list(csv.DictReader(f))
            for batch, row in enumerate(rows):
                rng = np.random.default_rng(np.random.SeedSequence(
                    [c["namespace"], seed, SPLITS.index(split), batch, 6]))
                expected = rng.normal(c["proxy"]["null_mean"], c["proxy"]["null_std"])
                if float(row["null_proxy"]) != expected:
                    raise ValueError(f"null label mismatch: seed={seed}, split={split}, batch={batch}")
                checked += 1
    return {**report, "null_labels_replayed": checked,
            "external_manifest_pinned": expected_manifest_sha256 is not None,
            "metadata_identity_checked": True,
            "expected_source_sha256": expected_source_sha256,
            "original_source_commit_field": meta.get("source_commit"),
            "audit_scope": "integrity, identities, structural rules and label construction; no predictive evaluation"}
