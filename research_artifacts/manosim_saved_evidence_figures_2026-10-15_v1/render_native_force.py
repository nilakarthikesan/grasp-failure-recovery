"""Render separated static-force and precision verdicts from saved operands."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'research_artifacts/manosim_native_point_contact_coupling_2026-10-15_v1'
OUT=ROOT/'runs/manosim_native_force_evidence_2026-10-15_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    gates=json.loads((STUDY/'protocol.json').read_text())['gates']
    rows,inputs=[],[]
    for case in (20,28):
        path=STUDY/f'run_v1/{case}/result.json';result=json.loads(path.read_text());m=result['measurement']
        item=next(a for a in result['artifacts'] if a['path'].endswith('surface_tangent_and_alignment.npz'))
        assert sha(item['path'])==item['sha256']
        with np.load(item['path'],allow_pickle=False) as z:
            J,FD=z['jacobian'],z['central_difference']
            derivative=float(np.max(np.abs(J-FD)/(gates['derivative_absolute']+gates['derivative_relative']*np.abs(FD))))
            vjp=float(np.max(np.abs(z['aggregate_vjp_tau']-z['aggregate_explicit_tau'])))/gates['virtual_work_absolute_W']
        mass=max(max(c['translation_max_residual_N']/gates['translation_mass_residual_N'],
                     c['angular_max_residual_Nm']/gates['angular_mass_residual_Nm']) for c in m['conditions'])
        normal=m['metrics']['source32_normal_max_component_error']/gates['source_alignment_normal_component']
        vertex=m['metrics']['source32_vertex_max_component_error_m']/gates['source_alignment_vertex_m']
        assert all(c['passed'] for c in m['conditions']) and m['static_point_coupling_status']=='PASS'
        assert m['source32_alignment_status']=='FAIL' and derivative<=1 and mass<=1 and normal>1
        rows.append(dict(case=case,ratios=[derivative,vjp,mass,vertex,normal],conditions_passed=12))
        inputs.extend([dict(path=str(path),sha256=sha(path)),dict(path=item['path'],sha256=item['sha256'])])
    fig,ax=plt.subplots(figsize=(11,5.5),layout='constrained')
    x=np.arange(5)
    for j,row in enumerate(rows):
        values=row['ratios'];bars=ax.bar(x+(j-.5)*.36,values,.36,label=f"Case {row['case']}",color=('#397b9b','#ca7c3f')[j])
        for bar,value in zip(bars,values):
            ax.text(bar.get_x()+bar.get_width()/2,value*1.35,f'{value:.2g}×',ha='center',va='bottom',fontsize=9)
    ax.axhline(1,color='#ac2828',linestyle='--',label='Declared limit (1×)')
    ax.set(yscale='log',ylim=(1e-7,100),xticks=x,xticklabels=['51 derivatives\nvs finite difference','Force transpose\nvs reverse derivative','Native mass\nresponse','Physical64 / source32\nvertices','Physical64 / source32\nface normals'],
           ylabel='Measured residual ÷ its original declared limit',title='Native static force coupling passes; strict source-normal alignment fails')
    ax.grid(axis='y',alpha=.2);ax.legend(loc='upper left',fontsize=9)
    fig.text(.5,-.08,'24/24 known-plane point loads pass across two native hands. Original skeleton and inertia preserved; contacts and actuation disabled.\n'
             'Separate smooth64 physical model and exact source32 transfer profile. Zero integration steps; full native contact, collection and training remain blocked.',
             ha='center',fontsize=9)
    target=OUT/'native_force_coupling_and_remaining_gap.png';fig.savefig(target,dpi=170,bbox_inches='tight')
    (OUT/'receipt.json').write_text(json.dumps(dict(schema='saved_native_static_force_illustration_v1',
        source=dict(path=str(Path(__file__)),sha256=sha(__file__)),inputs=inputs,measurements=rows,
        output=dict(path=str(target),sha256=sha(target)),new_engine_skin_geometry_or_physics_operations=0),indent=2)+'\n')
    print(target)

if __name__=='__main__':main()
