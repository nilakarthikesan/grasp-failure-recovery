#!/usr/bin/env python3
"""Verify the pinned DexGrasp Anything source and checkpoint before GPU setup."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


SOURCE_COMMIT = "86353d710ca71d0f547f0d8b15cc8ca959c78163"
CHECKPOINT_SHA256 = "f7675d8eb93b3978b2c2d72ec9f54ead6b936d1dd307e07605daa0c02bcac14f"
CHECKPOINT_SIZE = 110_490_637
MODEL_REPOSITORY_REVISION = "b24637ce366af4dd75cff723d2b853179155ec29"


def verify(source: Path, checkpoint: Path) -> dict:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source.resolve(), text=True
    ).strip()
    if revision != SOURCE_COMMIT:
        raise ValueError(f"DexGrasp Anything checkout must be {SOURCE_COMMIT}, got {revision}")
    size = checkpoint.stat().st_size
    if size != CHECKPOINT_SIZE:
        raise ValueError(f"checkpoint size mismatch: expected {CHECKPOINT_SIZE}, got {size}")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != CHECKPOINT_SHA256:
        raise ValueError(f"checkpoint SHA-256 mismatch: expected {CHECKPOINT_SHA256}, got {digest}")
    return {
        "source_commit": revision,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_size_bytes": size,
        "checkpoint_sha256": f"sha256:{digest}",
        "huggingface_repository_revision": MODEL_REPOSITORY_REVISION,
        "verified": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.source, args.checkpoint), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
