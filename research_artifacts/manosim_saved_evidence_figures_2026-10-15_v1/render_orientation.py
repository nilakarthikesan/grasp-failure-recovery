"""Render completed orientation evidence from saved masks only; no queries."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/manosim_saved_orientation_figure_2026-10-15_v1'
STUDY = ROOT / 'research_artifacts/manosim_orientation_qualification_2026-10-15_v2/run_v1'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    OUT.mkdir(parents=True, exist_ok=False)
    rows, inputs = [], []
    for case in (20, 28):
        path = STUDY / str(case) / 'result.json'
        report = json.loads(path.read_text())
        artifact = report['artifacts'][-1]
        assert sha(artifact['path']) == artifact['sha256']
        with np.load(artifact['path'], allow_pickle=False) as saved:
            reliable = int(np.any(saved['point_eligible'], axis=(1, 2)).sum())
            reversals = int(np.any(saved['point_sign_reversal'], axis=(1, 2)).sum())
            possible = int(saved['orientation_possible_required'].sum())
            stable = int(saved['cell_stable_matching'].sum())
            unresolved = int(saved['cell_unresolved'].sum())
        assert reversals == 0 and stable + unresolved == possible
        assert unresolved == report['measurement']['unresolved_required_cell_count']
        assert report['measurement']['strict_mapping_gate'] == 'INCOMPLETE'
        rows.append(dict(case=case, reliable_unique_sites=reliable,
                         reversed_unique_sites=reversals, possible_observations=possible,
                         stable_observations=stable, unresolved_observations=unresolved))
        inputs.extend([dict(path=str(path), sha256=sha(path)),
                       dict(path=artifact['path'], sha256=artifact['sha256'])])
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    labels = ['Case 20 (milk)', 'Case 28 (mug)']
    values = [r['reliable_unique_sites'] for r in rows]
    bars = ax[0].bar(labels, values, color=['#3979a0', '#cb7839'])
    for bar, value in zip(bars, values):
        ax[0].text(bar.get_x()+bar.get_width()/2, value+35,
                   f'{value:,} sites\n0 sign reversals', ha='center', va='bottom')
    ax[0].set_ylim(0, max(values)*1.25)
    ax[0].set_ylabel('Unique reliable original query sites')
    ax[0].set_title('Reliable point signs agree')
    ax[0].grid(axis='y', alpha=.2)
    for i, row in enumerate(rows):
        unresolved = row['unresolved_observations']
        stable = row['stable_observations']
        ax[1].barh(i, unresolved, color='#c5c9cc', label='Unresolved' if i == 0 else None)
        ax[1].barh(i, stable, left=unresolved, color='#3979a0', label='Certified matching' if i == 0 else None)
        ax[1].text(0, i+.27, f"{unresolved:,} unresolved; {stable} certified matching", fontsize=10)
    ax[1].set_yticks(range(2), labels)
    ax[1].set_ylim(-.55, 1.8)
    ax[1].set_xlim(0, max(r['possible_observations'] for r in rows)*1.08)
    ax[1].set_xlabel('Part × proximity-band cell observations (overlap)')
    ax[1].set_title('Continuous orientation remains INCOMPLETE')
    ax[1].legend(loc='upper right', fontsize=9)
    ax[1].grid(axis='x', alpha=.2)
    fig.suptitle('Fixed-pose contact-facing orientation: agreement with explicit uncertainty', fontsize=14)
    fig.text(.5, -.045, 'All original sites queried; ties, tangencies and unproved exterior coverage stay unresolved.\n'
             'Source orientation assumes closed, nonselfintersecting surfaces. No motion, physics, collection or training qualification.',
             ha='center', fontsize=9)
    target = OUT / 'orientation_evidence.png'
    fig.savefig(target, dpi=160, bbox_inches='tight')
    (OUT/'receipt.json').write_text(json.dumps(dict(
        schema='saved_orientation_illustration_v1', source=dict(path=str(Path(__file__)), sha256=sha(__file__)),
        inputs=inputs, measurements=rows, output=dict(path=str(target), sha256=sha(target)),
        new_queries_models_skin_or_physics=0), indent=2)+'\n')
    print(target)

if __name__ == '__main__':
    main()
