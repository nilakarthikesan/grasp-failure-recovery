"""Scientific figure from immutable static HUG seed-restoration records only."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).parent
STAGE=OUT.parent/'seed_stage'
CANDIDATE=STAGE/'candidate_targets.json'
SUMMARY=STAGE/'fit_summary.json'
PROTOCOL=STAGE/'protocol.json'
FINGERS=('Thumb','Index','Middle','Ring','Little')
LABELS={8:'Apple A',12:'Apple B',16:'Milk carton A',20:'Milk carton B',24:'Mug A',28:'Mug B'}
ORDER=(8,12,16,20,24,28)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
    before={str(p):sha(p) for p in (CANDIDATE,SUMMARY,PROTOCOL)}
    candidate,summary,protocol=map(read,(CANDIDATE,SUMMARY,PROTOCOL))
    assert sha(CANDIDATE)==summary['candidate_sha256']=='90fd9186fb72c174644fd806b1af26f49e372d066c36295430a37636b3c9d2c5'
    assert sha(PROTOCOL)==summary['protocol_sha256']
    assert summary['all_six_feasible_improving_poses_selected']
    records={c['source_inventory_index']:c for c in candidate['cases']}
    rows={c['index']:c for c in summary['six_case_rows']}
    assert len(records)==6 and tuple(sorted(records))==ORDER
    data=[]
    for idx in ORDER:
        c,row=records[idx],rows[idx]
        original=np.asarray(c['authoritative_original_tip_caps_sqrt_einsum_m'])
        selected=np.asarray(c['selected_tip_norms_sqrt_einsum_m'])
        assert original.shape==selected.shape==(5,)
        assert np.all(np.isfinite(original)) and np.all(np.isfinite(selected))
        assert np.all(selected<=original) and selected.mean()<=.010
        assert np.array_equal(original,np.asarray(c['solver_audit']['baseline']['tip_error_norm_m']))
        assert np.array_equal(selected,np.asarray(c['solver_audit']['final']['tip_error_norm_m']))
        reduction=100*(1-row['returned_objective']/row['original_objective'])
        assert abs(reduction-row['objective_reduction_percent'])<1e-12
        data.append({'source_inventory_index':idx,'label':LABELS[idx],
            'object_id':c['object_id'],'source_case':c['source_case'],'native_sha256':c['native_sha256'],
            'original_cap_errors_mm':(original*1000).tolist(),'selected_errors_mm':(selected*1000).tolist(),
            'selected_minus_original_nm':((selected-original)*1e9).tolist(),
            'original_mean_mm':float(original.mean()*1000),'selected_mean_mm':float(selected.mean()*1000),
            'original_objective':row['original_objective'],'selected_objective':row['returned_objective'],
            'frozen_objective_reduction_percent':reduction,'all_five_exact_caps_pass':True,
            'selected_nonoriginal_pose':c['solver_audit']['surface_candidate_returned'],
            'inner_restoration_optimizer':c['solver_audit']['proposals'][0].get('restoration_optimizer')})
    plotdata={'schema':'hug_feasible_seed_fingertip_plot_data_v1','source_sha256':before,
        'case_order':list(ORDER),'finger_order':list(FINGERS),
        'source_fields':{'original':'authoritative_original_tip_caps_sqrt_einsum_m',
            'selected':'selected_tip_norms_sqrt_einsum_m'},
        'transformation':'m to mm by multiplication by 1000; paired distance norms computed by sqrt(einsum). No averaging across cases.',
        'original_caps_are_existing_mapping_errors_not_zero_error_or_native_geometry':True,
        'mapping_fidelity_acceptance_claim':False,'development_only':True,'new_geometry_queries':0,
        'physics_steps':0,'new_eval_data':False,'cases':data}
    (OUT/'fingertip_plot_data.json').write_text(json.dumps(plotdata,indent=2,allow_nan=False)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.titlesize':13,
        'axes.labelsize':12,'xtick.labelsize':11,'ytick.labelsize':11,
        'pdf.fonttype':42,'ps.fonttype':42})
    fig,axs=plt.subplots(3,2,figsize=(12,10),sharey=True)
    fig.subplots_adjust(left=.095,right=.98,bottom=.175,top=.82,hspace=.52,wspace=.12)
    x=np.arange(5);width=.34;legend=[]
    for ax,c in zip(axs.flat,data,strict=True):
        orig=ax.bar(x-width/2,c['original_cap_errors_mm'],width=width,color='#c9ced4',edgecolor='#59636d',linewidth=.8,
            label="Original mapping (each tip's cap)")
        sel=ax.bar(x+width/2,c['selected_errors_mm'],width=width,color='#256786',edgecolor='#256786',linewidth=.8,
            label='FK-restored seed')
        if not legend:legend=[orig,sel]
        scene=c['source_case'].replace('_seed_0','').replace('scene_','scene ')
        mean=f"Five-tip mean: {c['original_mean_mm']:.3f} → {c['selected_mean_mm']:.3f} mm"
        if c['source_inventory_index']==28:mean='Five-tip mean: 6.609 → 6.609 mm (≈1 nm per tip)'
        ax.set_title(f"{c['label']} · case {c['source_inventory_index']} ({scene})\n{mean}",loc='left',pad=11,
            fontsize=12 if c['source_inventory_index']==28 else 13)
        ax.set_xticks(x,labels=FINGERS);ax.set_ylim(0,20);ax.set_yticks((0,5,10,15,20))
        ax.grid(axis='y',color='#e3e7eb',linewidth=.7);ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
        ax.spines[['left','bottom']].set_color('#9aa3ab');ax.tick_params(length=3,color='#9aa3ab')
    fig.suptitle('HUG fingertip error after constraint-feasible seed restoration',x=.095,y=.98,ha='left',fontsize=17,fontweight='bold')
    fig.text(.095,.942,'All 30 computed tip errors satisfy their original caps; six saved development poses.',ha='left',fontsize=12)
    fig.legend(legend,["Original mapping (each tip's cap)",'FK-restored seed'],loc='upper left',
        bbox_to_anchor=(.091,.928),ncol=2,frameon=False,fontsize=12,handlelength=1.5,columnspacing=2.5)
    fig.supylabel('Distance from native HUG fingertip position (mm)',x=.022,y=.535,fontsize=13)
    fig.text(.095,.12,'Each tip is capped by its own original mapping error. The separate 10 mm guard applies to the five-tip mean.',fontsize=11,ha='left')
    fig.text(.095,.091,'These caps preserve existing errors; they do not mean zero error or exact native geometry.',fontsize=11,ha='left')
    fig.text(.095,.062,'Constraint passage does not certify surface/contact fidelity. Static diagnostics only; no new evaluation episodes.',fontsize=11,ha='left')
    fig.savefig(OUT/'feasible_seed_fingertip_errors.png',dpi=200,facecolor='white')
    fig.savefig(OUT/'feasible_seed_fingertip_errors.pdf',facecolor='white')
    plt.close(fig)
    # A separate compact figure avoids crowding the anatomical error panels.
    fig,ax=plt.subplots(figsize=(10,5.3));fig.subplots_adjust(left=.11,right=.96,bottom=.30,top=.78)
    y=np.array([c['frozen_objective_reduction_percent'] for c in data]);bars=ax.bar(np.arange(6),y,color='#256786',width=.58)
    ax.set_xticks(np.arange(6),labels=[f"{c['label']}\ncase {c['source_inventory_index']}" for c in data]);ax.set_ylim(0,44)
    ax.set_ylabel('Reduction in frozen fitting objective (%)');ax.set_xlabel('Saved HUG development case',labelpad=10)
    ax.grid(axis='y',color='#e3e7eb',linewidth=.7);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    for bar,v in zip(bars,y,strict=True):ax.text(bar.get_x()+bar.get_width()/2,v+.8,f'{v:.2f}%',ha='center',va='bottom',fontsize=11)
    fig.suptitle('Native-relative fitting objective decreases after feasible seed restoration',x=.11,y=.95,ha='left',fontsize=14,fontweight='bold')
    fig.text(.11,.883,'Same frozen distance, closest-point, facing, anchor and joint objective; original reference per case.',fontsize=10)
    fig.text(.11,.093,'A lower fitting objective and passing tip constraints do not establish complete mapping fidelity.',fontsize=10)
    fig.text(.11,.057,'All six cases shown, including the small Mug B change. Separate scientific surface audit required.',fontsize=10)
    fig.savefig(OUT/'feasible_seed_objective_reductions.png',dpi=200,facecolor='white')
    fig.savefig(OUT/'feasible_seed_objective_reductions.pdf',facecolor='white')
    plt.close(fig)
    for p,h in before.items():assert sha(p)==h,p
    budget_exhausted=sum(not c['inner_restoration_optimizer']['success'] and (c['inner_restoration_optimizer']['status']==9 or c['inner_restoration_optimizer']['iterations']>=protocol['solver_configuration']['restoration_maxiter']) for c in data)
    assert budget_exhausted==4
    captions={'fingertip_figure':'Original and restored-seed native-relative fingertip errors on all six saved HUG development cases. Each original bar is the existing mapping error used as that fingertip\'s cap; it is not an exact native mesh representation. All 30 selected errors are no larger than their original caps by the declared computed norm comparison; the separate 10 mm guard applies to the mean over five tips. Mug B changes by approximately 1 nm per tip and therefore appears unchanged at the millimetre axis scale. Constraint passage is distinct from scientific surface/contact fidelity. Four inner restorations reached their fixed iteration budget; the figure reports final verified tip errors, not optimizer convergence.',
        'objective_figure':'Per-case percentage reduction relative to the original frozen objective, retaining all six cases and the 1.52% Mug B change. Weights, native-relative targets and invalid direction penalties remain fixed. A fitting score is neither a physical success probability nor a mapping acceptance criterion.'}
    outputs=('fingertip_plot_data.json','feasible_seed_fingertip_errors.png','feasible_seed_fingertip_errors.pdf','feasible_seed_objective_reductions.png','feasible_seed_objective_reductions.pdf')
    meta={'schema':'hug_feasible_seed_figure_metadata_v1','plot_script_sha256':sha(__file__),
        'source_sha256':before,'plot_data_sha256':sha(OUT/'fingertip_plot_data.json'),
        'outputs_sha256':{f:sha(OUT/f) for f in outputs},'captions':captions,
        'matplotlib_version':matplotlib.__version__,'numpy_version':np.__version__,
        'view_image_QA_pending':True,'inner_restoration_budget_exhaustion_count':budget_exhausted,'optimizer_convergence_claim':False,'standard_scientific_static_figures':True,'new_eval_data':False,'new_geometry_queries':0}
    (OUT/'figure_metadata.json').write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'outputs':list(outputs),'cases':6,'tip_comparisons':30,'all_exact_caps_pass':True,'source_preserved':True}))
if __name__=='__main__':main()
