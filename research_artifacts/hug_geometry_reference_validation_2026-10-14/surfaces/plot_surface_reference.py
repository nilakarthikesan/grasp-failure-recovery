from pathlib import Path
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

folder = Path(__file__).parent
report_path = folder / 'surface_reference_comparison.json'
report = json.loads(report_path.read_text())
methods = ['stored_current', 'dexpilot_shipped', 'dexpilot_metric_static']
names = ['Current vector', 'Released DexPilot style', 'Metric/static sensitivity']
colors = ['#256ca6', '#b85d05', '#22834b']
cases = report['cases']
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), layout='constrained')
x = np.arange(len(cases))
for i, method in enumerate(methods):
    values = [c['resolutions']['2']['methods'][method]['bands']['2']['whole_hand_unsigned_near']['jaccard'] for c in cases]
    axes[0].bar(x + (i-1)*.23, values, width=.22, color=colors[i], label=names[i])
axes[0].set_xticks(x, ['Apple A', 'Apple B', 'Milk A', 'Milk B', 'Mug A', 'Mug B'])
axes[0].set_ylim(0, .65)
axes[0].set_ylabel('Unsigned proximity-region IoU')
axes[0].set_title('Same saved poses, visible object query sites, 2 mm band')
axes[0].grid(axis='y', alpha=.18)
axes[0].set_axisbelow(True)
parts = ['thumb', 'index', 'middle', 'ring', 'little']
x = np.arange(len(parts))
counts = []
for part in parts:
    counts.append(sum(c['resolutions']['2']['methods']['stored_current']['bands']['2']['independent_per_part_unsigned'][part]['native_area_mm2'] > 0 for c in cases))
for i, method in enumerate(methods):
    values = []
    for part in parts:
        regions = [c['resolutions']['2']['methods'][method]['bands']['2']['independent_per_part_unsigned'][part] for c in cases]
        available = [r['native_retention'] for r in regions if r['native_retention'] is not None]
        values.append(np.mean(available) if available else np.nan)
    axes[1].bar(x + (i-1)*.23, values, width=.22, color=colors[i])
axes[1].set_xticks(x, [f'{part}\nn={n}/6' for part,n in zip(parts,counts)])
axes[1].set_ylim(0, 1)
axes[1].set_ylabel('Mean retained fraction of native proximity area')
axes[1].set_title('Per-pose means; native-empty parts remain undefined')
axes[1].grid(axis='y', alpha=.18)
axes[1].set_axisbelow(True)
fig.suptitle('HUG static reference comparison: measured surface correspondence', fontsize=14)
fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=3, bbox_to_anchor=(.5,-.045))
fig.text(.5,-.11,'Six existing development poses, one MANO shape; source poor proposals retained. Area density is a child-triangle area bound, not uniform spacing.\nDistances query native triangles and individual target colliders, including penetrated sites. These are geometric diagnostics, not physical contact or mapping pass rates.',ha='center',fontsize=9)
output = folder / 'surface_reference_tradeoff.png'
fig.savefig(output,dpi=180,bbox_inches='tight')
plt.close(fig)
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
(folder/'figure_metadata.json').write_text(json.dumps({'report_sha256':sha(report_path),'script_sha256':sha(Path(__file__)),'figure_sha256':sha(output),'native_nonempty_case_counts_by_part':dict(zip(parts,counts)),'scope':'Same six development poses, coarse query density, fixed2mm diagnostic band; both conditional retention and all-pose whole unsigned IoU shown; no success filtering'},indent=2)+'\n')
print(output)
