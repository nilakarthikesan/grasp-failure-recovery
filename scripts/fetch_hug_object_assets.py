"""Fetch geometry only from the pinned 1M-HuGs revision; never grasp labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from urllib.parse import quote
from urllib.request import urlopen

DATASET = "kevinywu/1m-hugs"
REVISION = "56ea3f100b1a4ec38bb35e986c0dd14d38977283"


def fetch_geometry(object_path: str, output: Path) -> dict:
    base = PurePosixPath(object_path)
    if base.is_absolute() or ".." in base.parts or not base.parts or base.parts[0] != "val":
        raise ValueError("declare a relative validation object path")
    if output.exists():
        raise FileExistsError("asset downloads require a fresh directory")
    prefix = str(base / "sim_assets")
    api = f"https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}/{quote(prefix)}?recursive=true&limit=1000"
    with urlopen(api, timeout=30) as response:
        if response.headers.get("Link"):
            raise ValueError("asset listing exceeds bounded single-page quota")
        listing = json.load(response)
    rows = [row for row in listing if row["type"] == "file"]
    if not rows or len(rows) > 200 or sum(row.get("size", 0) for row in rows) > 50_000_000:
        raise ValueError("asset closure exceeds declared file/byte quota")
    output.mkdir(parents=True)
    files = []
    for row in rows:
        path = PurePosixPath(row["path"])
        relative = path.relative_to(prefix)
        if ".." in relative.parts or relative.is_absolute():
            raise ValueError("invalid remote asset path")
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{quote(str(path))}"
        with urlopen(url, timeout=30) as response:
            content = response.read(50_000_001)
        if len(content) != row["size"] or len(content) > 50_000_000:
            raise ValueError("download length differs from pinned catalog")
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        files.append(dict(dataset_path=str(path), local_path=str(relative), url=url,
                          size=len(content), sha256=hashlib.sha256(content).hexdigest()))
    manifest = dict(dataset=DATASET, revision=REVISION, license="cc-by-4.0",
                    attribution="1M-HuGs / HUG authors; original object geometry at linked pinned paths",
                    usage="geometry only; no grasp labels or trajectory annotations", files=files)
    (output / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = fetch_geometry(args.object_path, args.output)
    print(json.dumps(dict(files=len(result["files"]), bytes=sum(row["size"] for row in result["files"]))))


if __name__ == "__main__":
    main()
