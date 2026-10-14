from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

r=json.loads(Path('/private/tmp/hug-palm-anchor-2026-10-14/palm_anchor_report.json').read_text())
t=r['global_translation_diagnostics']; a=r['anatomical_proxy_summary']
fig,axes=plt.subplots(1,2,figsize=(11.5,5.0),layout='constrained')
values=[t['tips5']['original_point_error_m']['mean']*1000,t['tips5']['single_common_removed_point_error_m']['mean']*1000,r['cross_point_family_translation_diagnostic']['fingertip_errors_if_MCP_mean_algebraically_removed_m']['mean']*1000]
labels=['Original','Tip mean\nremoved','MCP-proxy mean\nremoved']
axes[0].bar(labels,values,color=['#637383','#28759c','#ab7135'],width=.62)
for x,y in enumerate(values):axes[0].text(x,y+.2,f'{y:.2f}',ha='center',fontsize=10)
axes[0].set_ylim(0,12);axes[0].set_ylabel('Mean fingertip separation (mm)');axes[0].set_title('Residual decomposition on the same 32 saved poses')
axes[0].grid(axis='y',alpha=.2);axes[0].set_axisbelow(True)
vals=[a['native_wrist_to_MCP_centroid_m']['mean']*1000,a['shadow_palm_to_MCP_centroid_m']['mean']*1000,a['shadow_actual_wrist_to_MCP_centroid_m']['mean']*1000]
labels=['Native wrist','Shadow palm\norigin','Shadow wrist\nbody']
axes[1].bar(labels,vals,color=['#637383','#28759c','#ab7135'],width=.62)
for x,y in enumerate(vals):axes[1].text(x,y+2,f'{y:.2f}',ha='center',fontsize=10)
axes[1].set_ylim(0,150);axes[1].set_ylabel('Distance to finger-base proxy centroid (mm)');axes[1].set_title('Body names do not establish anatomical correspondence')
axes[1].grid(axis='y',alpha=.2);axes[1].set_axisbelow(True)
fig.suptitle('HUG wrist-to-palm check: placement policy and remaining geometry differences',fontsize=14)
fig.text(.5,-.04,'Source: existing Oct 11 poses, Oct 14 static diagnostic. MCP denotes the four nonthumb finger-base proxies.\nLeft comparisons subtract measured mean residuals algebraically; no robot pose or mapping was changed. No physical contact or prediction result.',ha='center',fontsize=9)
out=Path('/private/tmp/hug_anchor_diagnostic_2026-10-14.png');fig.savefig(out,dpi=180,bbox_inches='tight');plt.close(fig)
print(str(out))
