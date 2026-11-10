"""Measure controlled readout effects versus true motion and temporal gap."""
import json
from pathlib import Path
import numpy as np

def ci(values):
    values=np.asarray(values);values=values[np.isfinite(values)]
    if not len(values):return [None,None,None]
    rng=np.random.default_rng(42)
    means=values[rng.integers(len(values),size=(10000,len(values)))].mean(-1)
    return [float(values.mean()),*np.quantile(means,[.025,.975]).tolist()]

def main():
    root=Path('/share/yty/heft');output={};bins=[0,2,4,8,16,32,64,float('inf')]
    for model in ['wan','cosmos']:
        records=[]
        for path in sorted((root/f'results/probe_{model}_real_t25').glob('*.npz')):
            with np.load(path) as f:record={k:f[k] for k in ['errors','valid','variant_names','layers']}
            metadata=json.loads(path.with_suffix('.json').read_text())['feature_metadata']
            with np.load(root/f'data/davis_records/{path.stem}.annotations.npz') as a:
                points=a['points'][~a['occluded'][:,0]][:,metadata['sparse_frames']].transpose(1,0,2)
            motion=np.linalg.norm((points[1:]-points[:1])*256,axis=-1)
            score=(record['errors'][...,None]<[1,2,4,8,16]).mean(-1)*100
            record.update(score=score,motion=motion,frames=metadata['sparse_frames'][1:]);records.append(record)
        names=records[0]['variant_names'].tolist();vi=names.index('post_full');rows={}
        for variant in ['post_full','post_low15','time_removed','space_removed','time0_space050','post_centered']:
            values=[];counts=[]
            for lower,upper in zip(bins[:-1],bins[1:]):
                pervideo=[];number=0
                for r in records:
                    mask=r['valid']&(r['motion']>=lower)&(r['motion']<upper);number+=int(mask.sum())
                    if mask.any():pervideo.append(float((r['score'][:,:,names.index(variant),mask]-r['score'][:,:,vi,mask]).mean()))
                values.append(ci(pervideo));counts.append(number)
            rows[variant]={'motion_bin_gain_ci':values,'point_frames':counts}
        output[model]={'motion_bins_256px':bins,'population_readout_gain':rows,
                       'units':'PCK percentage points; macro over heads and point-frames within each video, bootstrap videos'}
    (root/'results/motion_stratified_readout.json').write_text(json.dumps(output,indent=2))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),sharey=True)
    for ax,(model,result) in zip(axes,output.items(),strict=True):
        for variant,label in [('post_low15','Remove fastest 15%'),('time_removed','Remove temporal phase'),('space_removed','Remove spatial phase')]:
            data=np.array(result['population_readout_gain'][variant]['motion_bin_gain_ci'],dtype=float)
            ax.errorbar(np.arange(7),data[:,0],yerr=[data[:,0]-data[:,1],data[:,2]-data[:,0]],marker='o',label=label,capsize=2)
        ax.axhline(0,color='k',lw=.7);ax.set_xticks(np.arange(7),['0–2','2–4','4–8','8–16','16–32','32–64','64+'])
        ax.set(title=model.capitalize(),xlabel='True displacement (pixels at 256 resolution)')
    axes[0].set_ylabel('Direct matching gain over full keys (PCK points)');axes[1].legend(frameon=False,fontsize=8)
    fig.tight_layout()
    for extension in ['png','pdf']:fig.savefig(root/f'figures/motion_stratified_readout.{extension}',dpi=180,bbox_inches='tight')
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()
