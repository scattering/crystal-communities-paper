"""Prevent projection across incompatible versions of the crystal encoder."""
from __future__ import annotations

import json
from pathlib import Path

from crystal_neighbors import FEATURE_VERSION


def require_feature_version(feature_path: str | Path) -> dict:
    """Require current encoder metadata beside the actual raw feature file.

    Production runs write ``summary.json``; a separately packaged feature
    matrix may instead carry ``feature_metadata.json``. Either can establish
    the version, but explicit conflicting versions are never ignored. Follow
    symlinks so an unrelated sidecar cannot relabel a legacy feature matrix.
    """
    feature_path = Path(feature_path).resolve()
    metadata_paths = [
        feature_path.parent / "summary.json",
        feature_path.parent / "feature_metadata.json",
    ]
    matching = None
    remedy = (
        "Regenerate the ICSD features with the current encoder and use their "
        "regenerated basis; do not reuse legacy features or PCA."
    )
    for path in metadata_paths:
        if not path.exists():
            continue
        try:
            metadata = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot read feature provenance from {path}: {exc}. {remedy}") from exc
        if not isinstance(metadata, dict):
            raise ValueError(f"Feature provenance in {path} must be a JSON object. {remedy}")
        version = metadata.get("feature_version")
        if version is None:
            continue
        if version != FEATURE_VERSION:
            raise ValueError(
                f"Incompatible ICSD feature version {version!r} in {path}; "
                f"the encoder requires {FEATURE_VERSION!r}. {remedy}"
            )
        matching = metadata
    if matching is None:
        raise ValueError(
            f"Missing feature_version provenance for {feature_path}; expected "
            f"{FEATURE_VERSION!r} in summary.json or feature_metadata.json "
            f"beside the feature file. {remedy}"
        )
    return matching
