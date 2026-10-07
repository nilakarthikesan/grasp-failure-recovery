"""Offline RGB-D preflight. Passing is not proof of HUG inference or grasp success."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def validate_arrays(rgb, depth, K, mask, point, *, registered, calibration_matches):
    errors, warnings = [], []
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        errors.append('RGB must be uint8 HxWx3')
    if depth.ndim != 2 or depth.shape != rgb.shape[:2]:
        errors.append('Depth must match RGB height and width')
    if not np.issubdtype(depth.dtype, np.floating):
        errors.append('Depth must explicitly be floating-point meters')
    if not np.isfinite(depth).all() or np.any(depth < 0):
        errors.append('Depth must be finite and nonnegative; zero means invalid')
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any(): errors.append('No valid depth')
    if K.shape != (3, 3) or not np.isfinite(K).all():
        errors.append('Intrinsics must be a finite 3x3 matrix')
    elif not np.allclose(K[2], [0, 0, 1]) or K[0,0] <= 0 or K[1,1] <= 0 or abs(np.linalg.det(K)) < 1e-12:
        errors.append('Invalid pinhole intrinsics')
    elif not (0 <= K[0,2] < rgb.shape[1] and 0 <= K[1,2] < rgb.shape[0]):
        warnings.append('Principal point outside image; review calibration/cropping')
    if not registered: errors.append('RGB-depth registration not declared')
    if not calibration_matches: errors.append('Calibration at this image resolution not declared')
    if mask.shape != depth.shape or not np.isin(mask, [0, 1, 255]).all() or not np.any(mask):
        errors.append('Object mask must be nonempty, binary, and match image dimensions')
    if len(point) != 2 or not np.isfinite(point).all():
        errors.append('Selection must be finite [u,v]')
    else:
        u,v = np.rint(point).astype(int)
        if not (0 <= v < depth.shape[0] and 0 <= u < depth.shape[1]):
            errors.append('Selection outside image')
        elif depth.shape == mask.shape:
            if not mask[v,u]: errors.append('Selection outside object mask')
            if not valid[v,u]: errors.append('Selection has invalid depth; no silent fallback')
    if mask.shape == depth.shape and np.any(mask):
        fraction = float(valid[mask > 0].mean())
        if fraction == 0: errors.append('Object has no valid depth')
        elif fraction < .95: warnings.append(f'Object valid-depth coverage only {fraction:.1%}')
    warnings.append('Alignment, object identity, and physical scale require visual review')
    return {'preflight_passed': not errors, 'errors': errors, 'warnings': warnings,
            'hug_preprocessing_tested': False, 'hug_inference_tested': False,
            'physical_grasp_tested': False}


def check_manifest(path, output):
    from PIL import Image, ImageDraw
    path, output = Path(path), Path(output)
    manifest = json.loads(path.read_text()); root = path.parent
    names = ('rgb_path','depth_m_path','intrinsics_path','mask_path')
    paths = {k: root / manifest[k] for k in names}
    rgb = np.asarray(Image.open(paths['rgb_path']).convert('RGB'))
    depth = np.load(paths['depth_m_path'], allow_pickle=False)
    K = np.load(paths['intrinsics_path'], allow_pickle=False)
    mask = np.asarray(Image.open(paths['mask_path']).convert('L'))
    if manifest.get('depth_units') != 'meters': raise ValueError('Explicit depth_units=meters required')
    if not manifest.get('source'): raise ValueError('Source provenance required')
    result = validate_arrays(rgb, depth, K, mask, manifest['selection_uv'],
        registered=manifest.get('registered_to_rgb') is True,
        calibration_matches=manifest.get('intrinsics_at_rgb_resolution') is True)
    result['source'] = manifest['source']
    result['file_sha256'] = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()}
    result['manifest_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    if rgb.shape[:2] == mask.shape:
        overlay = rgb.copy(); selected = mask > 0
        overlay[selected] = (.5*overlay[selected] + .5*np.array([255,0,0])).astype('uint8')
        im = Image.fromarray(overlay); draw = ImageDraw.Draw(im)
        u,v = manifest['selection_uv']; draw.ellipse((u-4,v-4,u+4,v+4),outline='lime',width=2)
        im.save(output/'selection_overlay.png')
    if depth.ndim == 2 and np.any(depth > 0):
        valid = np.isfinite(depth) & (depth > 0); lo,hi = np.percentile(depth[valid],[2,98])
        preview = np.zeros(depth.shape,dtype='uint8')
        preview[valid] = (255*np.clip((depth[valid]-lo)/max(hi-lo,1e-6),0,1)).astype('uint8')
        Image.fromarray(preview).save(output/'depth_preview.png')
        result['depth_preview_range_m'] = [float(lo),float(hi)]
    (output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest'); parser.add_argument('--output',required=True)
    args=parser.parse_args()
    try: result=check_manifest(args.manifest,args.output)
    except (ValueError,KeyError,OSError) as exc:
        print(f'Observation rejected: {exc}'); return 1
    print(json.dumps(result,indent=2)); return 0 if result['preflight_passed'] else 1

if __name__ == '__main__': raise SystemExit(main())
