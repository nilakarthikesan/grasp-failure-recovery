"""Show source-native mapping correction using sealed saved readbacks only."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = Path(__file__).resolve().parents[2]
NEW = ROOT/'research_artifacts/manosim_native_source_pipeline_2026-10-15_v1/run_v1'
OLD = ROOT/'research_artifacts/manosim_skin_engine_bridge_2026-10-15_v2/run_v1'
OUT = ROOT/'runs/manosim_native_source_evidence_2026-10-15_v1'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    OUT.mkdir(parents=True, exist_ok=False)
    fig = plt.figure(figsize=(13, 9), layout='constrained')
    sources, metrics, old_normals, new_normals = [], [], [], []
    chart = fig.add_subplot(2, 2, 3)
    for column, case in enumerate((20, 28)):
        path = NEW/str(case)/'result.json'
        report = json.loads(path.read_text())
        artifact = next(a for a in report['artifacts'] if a['path'].endswith('all109_native_source_engine_readbacks.npz'))
        assert sha(artifact['path']) == artifact['sha256']
        with np.load(artifact['path'], allow_pickle=False) as z:
            engine = z['compiled_vertices_world_m'].copy()
            source = z['source_vertices_world'].copy()
            faces = z['faces'].copy()
            assert engine.shape == source.shape == (109, 778, 3)
            ids = [row['pose_id'] for row in json.loads((NEW.parent/'protocol.json').read_text())['motion_grid']['poses']]
            pose = ids.index('coupled_+1_5deg')
            # This is displacement in the finite static-state grid, not velocity.
            moved = np.linalg.norm(engine-engine[0], axis=2).max(axis=1)*1000
            chart.plot(moved, label=f'Case {case}')
        ax = fig.add_subplot(2, 2, column+1, projection='3d')
        origin = engine[0].mean(axis=0)
        for k, color, alpha in ((0, '#949ca2', .22), (pose, '#357ca2', .65)):
            vertices = (engine[k]-origin)*1000
            ax.add_collection3d(Poly3DCollection(vertices[faces], facecolors=color,
                edgecolors=color, linewidths=.1, alpha=alpha))
        points = (engine[[0, pose]]-origin)*1000
        middle = (points.min(axis=(0, 1))+points.max(axis=(0, 1)))/2
        radius = np.ptp(points, axis=(0, 1)).max()*.56
        ax.set(xlim=(middle[0]-radius, middle[0]+radius), ylim=(middle[1]-radius, middle[1]+radius),
               zlim=(middle[2]-radius, middle[2]+radius), xlabel='x (mm)', ylabel='y (mm)', zlabel='z (mm)')
        ax.set_box_aspect((1, 1, 1)); ax.view_init(25, -60)
        ax.set_title(f'Case {case}: actual engine surface\nGrey baseline; blue coupled +5° static pose')
        old_path = OLD/str(case)/'result.json'
        prior = json.loads(old_path.read_text())
        old_normals.append(prior['metrics']['original_native_baseline_normal_component_error'])
        new_normals.append(report['metrics']['original_native_baseline_normal_component_error'])
        assert all(report['checks'].values()) and report['new_collection_or_training_allowed'] is False
        metrics.append(dict(case=case, checks=report['checks'], metrics=report['metrics'],
                            maximum_static_displacement_mm=float(moved.max())))
        sources.extend([dict(path=str(path), sha256=sha(path)), dict(path=artifact['path'], sha256=artifact['sha256']),
                        dict(path=str(old_path), sha256=sha(old_path))])
    chart.set(xlabel='Declared static-state index (109 per case)', ylabel='Maximum vertex displacement from baseline (mm)',
              title='The engine surface follows finger, wrist and slide changes')
    chart.grid(alpha=.2); chart.legend()
    error = fig.add_subplot(2, 2, 4)
    x = np.arange(2)
    error.bar(x-.18, old_normals, .36, label='Earlier GS64 / world64 path', color='#bc6748')
    error.bar(x+.18, new_normals, .36, label='Corrected source AA32 / camera32 path', color='#3b8298')
    error.axhline(1e-6, color='#a52828', linestyle='--', label='Unchanged original normal limit')
    error.set(yscale='log', xticks=x, xticklabels=['Case 20', 'Case 28'], ylabel='Maximum face-normal component error',
              title='Starting pose vs the unchanged original HUG surface')
    error.grid(axis='y', alpha=.2); error.legend(fontsize=8, loc='center right')
    fig.suptitle('Corrected HUG → MANOSim source transfer: all 218 static poses pass', fontsize=16)
    fig.text(.5, -.035, 'New source-initialized state and precision domain; previous FAIL retained. Contacts and actuation disabled.\n'
             'Roundoff geometry agreement does not qualify dynamic finger tracking, native contact, collection or training.', ha='center', fontsize=10)
    target = OUT/'faithful_finger_transfer.png'
    fig.savefig(target, dpi=160, bbox_inches='tight')
    (OUT/'receipt.json').write_text(json.dumps(dict(schema='saved_native_source_pipeline_illustration_v1',
        source=dict(path=str(Path(__file__)), sha256=sha(__file__)), inputs=sources, measurements=metrics,
        output=dict(path=str(target), sha256=sha(target)), new_models_skin_queries_or_physics=0), indent=2)+'\n')
    print(target)

if __name__ == '__main__':
    main()
