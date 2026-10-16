"""Scientific figure from frozen saved alignment operands only; no queries/FK.

Uses NumPy, Pillow and Matplotlib; never imports engine, MANO or producer code.
The unchanged gate is distance/allowance <=1 for all32 selected points.
"""
from pathlib import Path
import argparse,hashlib,io,json,pickle,sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from PIL import Image

REPORT_SHA="55ced44bb921f1ae0a76dd46768de3920132ece4a28ae9a75b552d2bbe2fa9c6"
MECHANICAL_SHA="d093594729839b18d70a480b0e9125207e6b069d2a9c9f9f462295e1f5eac726"
COHORT_SHA="8caeb6dcc973b3488c6bb5097013ffa567c59232e398cdacd261e93dc4b1636b"
GENERATION_SHA="f0a1e26b2aa73c3c3bca2a8ba6e5a8091b7b8e28fd215b802fc44eb18b94fc7b"

def require(x,msg):
 if not bool(x):raise ValueError(msg)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def bind(path):
 p=Path(path).resolve();require(not any(x.is_symlink() for x in (p,*p.parents)),"Aliased input/output");return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def npz(path):
 with np.load(path,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}

class ImageObservationUnpickler(pickle.Unpickler):
 def find_class(self,module,name):
  allowed={('numpy','ndarray'),('numpy','dtype'),('_codecs','encode')}|{(m,n) for m in ('numpy.core.multiarray','numpy._core.multiarray') for n in ('_reconstruct','scalar')}|{(m,'_frombuffer') for m in ('numpy.core.numeric','numpy._core.numeric')}
  require((module,name) in allowed,"Unsupported pickle global");return super().find_class(module,name)

def render(root,output):
 study=root/'research_artifacts/manosim_native_object_pilot_2026-10-16_v1';pp=study/'protocol.json';rp=study/'run_v1/report.json';mp=study/'run_v1/independent_readback.json'
 cp=root/'research_artifacts/manosim_native_pilot_cohort_audit_2026-10-16_v1/independent_readback.json';gp=root/'runs/manosim_native_hug_pilot_generation_2026-10-16_v2/generation_report.json'
 require(sha(rp)==REPORT_SHA and sha(mp)==MECHANICAL_SHA and sha(cp)==COHORT_SHA and sha(gp)==GENERATION_SHA,"Exact independently verified lineage required")
 p=read(pp);r=read(rp);m=read(mp);c=read(cp);require(r['status']=='COMPLETED' and r['actual_counts']['steps']==0 and c['primary']['eligible']==0,"Refused zero-step pilot lineage differs")
 require(c['artifact_count']==649 and c['closure_before']['active_bindings']==1534 and m['status'].startswith('VERIFIED_'),"Independent readback unavailable")
 raw=Path(p['raw_root']);artifacts={x['path']:x for x in r['raw_artifacts']};paths={pp,rp,mp,cp,gp,Path(p['catalog_binding']['path'])}
 records=[]
 for oid,spec in sorted(p['observations'].items()):
  path=raw/'alignment'/(oid+'.npz');row=artifacts[str(path)];require(bind(path)=={k:row[k] for k in ('path','sha256','bytes')},"Saved alignment creation hash differs")
  obs=Path(spec['path']);require(sha(obs)==p['active_bindings'][str(obs)],"Original generation observation changed");paths|={path,obs}
  a=npz(path);d=np.asarray(a['distances_m']);allow=np.asarray(a['allowances_m']);require(d.shape==allow.shape==(32,) and np.isfinite(d).all() and np.isfinite(allow).all() and np.all(d>=0) and np.all(allow>0),"Exactly32 finite saved residuals required")
  require(a['pixel_uv'].shape==(32,2) and bool(a['passed_support']),"Original query support changed")
  ratios=d/allow;records.append(dict(observation_id=oid,object_id=spec['object_id'],input_path=str(obs),alignment_path=str(path),ratios=ratios,distances_m=d,allowances_m=allow,pixels=a['pixel_uv']))
 require(len(records)==60,"Fixed60 saved observations required")
 before=[bind(x) for x in sorted(paths)];allratios=np.concatenate([x['ratios'] for x in records]);passed=[bool(np.all(x['ratios']<=1)) for x in records]
 require(len(allratios)==1920 and sum(passed)==0,"Expected unchanged actual all32 gate differs")
 first=p['schedule'][0];representative=next(x for x in records if x['observation_id']==first['observation_id']);observation=ImageObservationUnpickler(io.BytesIO(Path(representative['input_path']).read_bytes())).load();require(observation['grasp'] is None and isinstance(observation['image'],bytes),"Unlabeled saved RGB bytes required")
 image=np.asarray(Image.open(io.BytesIO(observation['image'])).convert('RGB'));require(image.shape[:2]==(224,224),"Literal cropped RGB shape differs")
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':11,'figure.facecolor':'#f6f8fb','axes.facecolor':'white','axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'#f6f8fb'})
 fig=plt.figure(figsize=(15,10.2),layout=None);gs=fig.add_gridspec(2,2,left=.065,right=.95,bottom=.15,top=.84,wspace=.25,hspace=.47,height_ratios=[1,.92]);ax=fig.add_subplot(gs[0,0]);rgb=fig.add_subplot(gs[0,1]);bx=fig.add_subplot(gs[1,:])
 fig.text(.065,.955,'Saved RGBD–visual-mesh alignment',fontsize=22,weight='bold',color='#172438')
 fig.text(.065,.917,'0 / 60 observations pass the frozen all-32-point gate',fontsize=17,weight='bold',color='#a52b35')
 fig.text(.065,.879,'120 primary + 15 matched slots refused setup · 0 task steps · all outcomes null · training closed',fontsize=11,color='#4b596c')
 order=np.sort(allratios);ecdf=np.arange(1,len(order)+1)/len(order);ax.step(order,ecdf,where='post',color='#285b8f',lw=2)
 ax.axvline(1,color='#b32635',lw=1.8,ls='--',label='Unchanged threshold = 1');ax.set_xscale('log');ax.set_xlim(.001,150);ax.set_ylim(0,1.01);ax.set_xlabel('Saved distance / frozen allowance (log scale)');ax.set_ylabel('Cumulative fraction of 1,920 points');ax.set_title('A   All saved query residuals',loc='left',weight='bold',pad=12);ax.grid(axis='y',alpha=.18);ax.legend(loc='lower right',frameon=False,fontsize=9)
 within=int(np.sum(allratios<=1));ax.text(.03,.89,f'{within:,} / 1,920 points within allowance\n({within/len(allratios):.1%}); every view has a violation',transform=ax.transAxes,color='#37475d',fontsize=10,bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=5))
 rgb.imshow(image);sc=rgb.scatter(representative['pixels'][:,0],representative['pixels'][:,1],c=representative['ratios'],norm=LogNorm(vmin=.01,vmax=100),cmap='coolwarm',s=36,edgecolor='white',linewidth=.55)
 rgb.set_xlim(-.5,223.5);rgb.set_ylim(223.5,-.5);rgb.set_aspect('equal');rgb.set_axis_off();rgb.set_title('B   Representative saved RGB\nAll 32 query pixels',loc='left',weight='bold',pad=12)
 rgb.text(.5,-.045,'headphones · first scheduled capture 00000064\nPixel colors show distance / allowance; no new geometry query',transform=rgb.transAxes,ha='center',va='top',fontsize=9,color='#4b596c')
 cb=fig.colorbar(sc,ax=rgb,pad=.03,fraction=.045);cb.set_label('Distance / allowance',fontsize=9);cb.set_ticks([.01,.1,1,10,100]);cb.set_ticklabels(['0.01','0.1','1','10','100'])
 ranked=sorted(records,key=lambda x:(float(np.median(x['ratios'])),x['observation_id']));positions=np.arange(1,61)
 stats=[]
 for x in ranked:
  v=x['ratios'];q=np.percentile(v,[0,25,50,75,100]);stats.append(dict(med=q[2],q1=q[1],q3=q[3],whislo=q[0],whishi=q[4],fliers=[]))
 bx.bxp(stats,positions=positions,widths=.64,showfliers=False,patch_artist=True,boxprops=dict(facecolor='#d8e5f1',edgecolor='#41688e',linewidth=.65),medianprops=dict(color='#153b61',linewidth=1),whiskerprops=dict(color='#7993ac',linewidth=.7),capprops=dict(color='#7993ac',linewidth=.7))
 bx.axhline(1,color='#b32635',ls='--',lw=1.8);bx.set_yscale('log');bx.set_ylim(.001,150);bx.set_xlim(0,61);bx.set_xticks([1,10,20,30,40,50,60]);bx.set_xticklabels(['1','10','20','30','40','50','60']);bx.set_xlabel('All 60 observations, ordered by median ratio (order is descriptive)');bx.set_ylabel('Distance / allowance (log scale)');bx.set_title('C   Per-observation residual distributions: median, IQR and full range of 32 points',loc='left',weight='bold',pad=12);bx.grid(axis='y',alpha=.18)
 fig.text(.065,.085,'Readout: each observation needs all 32 ratios ≤ 1. Individual points below the line do not qualify a view.',fontsize=10,weight='bold',color='#34455b')
 fig.text(.065,.058,'Distance and depth/pixel allowances are preserved v1 operands. The separate literal-camera precision guard also refused every slot.',fontsize=9,color='#5b6779')
 fig.text(.065,.035,'This is setup-alignment evidence, not a physical grasp failure rate. No model/FK/nearest-surface queries, physics or fitting were run for this figure.',fontsize=9,color='#5b6779')
 output.mkdir(parents=True,exist_ok=True);png=output/'saved_alignment_diagnostic.png';require(not png.exists(),"Fresh figure path required");fig.savefig(png,dpi=180);plt.close(fig)
 after=[bind(x) for x in sorted(paths)];require(before==after,"Figure inputs changed during render")
 quantiles={str(q):float(np.percentile(allratios,q)) for q in (0,10,25,50,75,90,95,99,100)}
 summary=dict(schema_version='manosim_saved_alignment_figure_v1',status='RENDERED_SAVED_ONLY',inputs_before=before,inputs_after=after,figure=binding_png(png),source=bind(Path(__file__)),frozen_threshold=1.,new_acceptance_criterion=False,observations=60,points=1920,points_within_allowance=within,observations_all32_pass=sum(passed),ratio_quantiles=quantiles,primary_slots=120,matched_slots=15,all_outcomes_null=True,task_steps=0,scientific_training_allowed=False,representative=dict(observation_id=representative['observation_id'],object_id=representative['object_id'],input_path=representative['input_path'],not_selected_for_best_or_worst_quality=True,selection='first scheduled slot',query_count=32),per_observation=[dict(observation_id=x['observation_id'],object_id=x['object_id'],input_path=x['input_path'],alignment_path=x['alignment_path'],points_within_allowance=int(np.sum(x['ratios']<=1)),all32_pass=bool(np.all(x['ratios']<=1)),ratio_min=float(x['ratios'].min()),ratio_median=float(np.median(x['ratios'])),ratio_max=float(x['ratios'].max()),distance_max_m=float(x['distances_m'].max()),allowance_min_m=float(x['allowances_m'].min()),allowance_max_m=float(x['allowances_m'].max())) for x in records],operations=dict(new_geometry_queries=0,native_or_model_calls=0,physics_steps=0,fits=0),runtime=dict(python=sys.version,numpy=np.__version__,matplotlib=matplotlib.__version__,Pillow=Image.__version__))
 mp=output/'figure_manifest.json'
 with mp.open('x') as f:json.dump(summary,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
 print(json.dumps(dict(figure=bind(png),manifest=bind(mp),observations_all32_pass=sum(passed),points_within_allowance=within)))

def binding_png(path):
 with Image.open(path) as im:return {**bind(path),"width":im.width,"height":im.height}

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();render(a.root.resolve(),a.output.resolve())
if __name__=='__main__':main()
