from pathlib import Path
import gzip, hashlib
base=Path(__file__).resolve().parent
specs=[(base/'geometry/geometry_cause_report.json',Path('/private/tmp/hug_geometry_causes_2026-10-14/geometry_cause_report.json'),False), (base/'reference/hug_reference_comparison_full.json.gz',Path('/private/tmp/hug_reference_compare_2026-10-14/hug_reference_comparison.json'),True)]
for src,dst,compressed in specs:
    raw=src.read_bytes()
    if compressed: raw=gzip.decompress(raw)
    if dst.exists(): assert dst.read_bytes()==raw, f'Existing diagnostic differs: {dst}'
    else:
        dst.parent.mkdir(parents=True,exist_ok=True)
        dst.write_bytes(raw)
    print(dst,hashlib.sha256(raw).hexdigest())
