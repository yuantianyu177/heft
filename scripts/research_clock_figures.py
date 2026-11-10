"""Render compact scientific figures from saved statistics, without inference."""
import json
from pathlib import Path
import numpy as np

def make_clock_figures(root,summary,cal):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':8.5,'axes.titlesize':9.5,'axes.labelsize':9,'pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(7.5,6.0));grid=fig.add_gridspec(2,2,height_ratios=[1.15,.95])
    clock=fig.add_subplot(grid[0,0]);lag=fig.add_subplot(grid[0,1]);bench=fig.add_subplot(grid[1,:])
    for mode,rows in cal['results'].items():
        a=np.array([r['delay_nominal_minus_effective'] for r in rows]);times=rows[0]['nominal_times']
        label='Native latent' if mode=='native' else 'Frame-aligned latent'
        clock.plot(times,a.mean(0),'o-',label=label,ms=3,lw=1.2)
        clock.fill_between(times,np.quantile(a,.1,axis=0),np.quantile(a,.9,axis=0),alpha=.13)
    if 'codec_clock' in summary:
        clock.plot(range(1,25),summary['codec_clock']['decoded_rgb_summary']['mean_delay_by_time'],color='#33936e',lw=1.5,label='Full-codec RGB')
    clock.axhline(1.25,color='grey',ls='--',lw=.7);clock.set(xlabel='Nominal video frame',ylabel='Content delay (frames)',title='(a) Encoder and full codec');clock.legend(fontsize=7,loc='center right')
    records=summary['datasets']['DAVIS diagnostic']['projected_prediction_lag']
    before=records['native_polyphase']['per_video_median'];after=records['polyphase_corrected']['per_video_median']
    for name in sorted(before):lag.plot([0,1],[before[name],after[name]],'o-',ms=2.5,lw=.7,alpha=.35,color='#397caf')
    lag.axhline(1.25,ls='--',lw=1,color='#a74c4c',label='VAE-only estimate');lag.axhline(0,lw=.6,color='grey')
    lag.set_xticks([0,1],['Native','Calibrated']);lag.set(ylabel='Median prediction lag (frames)',title='(b) Every DAVIS video');lag.legend(fontsize=7,loc='upper right')
    labels=[];means=[];errors=[]
    for key,label in [('DAVIS complete official','Wan / DAVIS'),('Cosmos DAVIS complete official','Cosmos / DAVIS'),
        ('RGB-Stacking complete','Wan / RGB-Stacking'),('PointOdyssey 200-frame prefixes','Wan / PointOdyssey (200f)')]:
        row=summary['datasets'][key]['paired_comparisons']['polyphase_corrected minus native_polyphase']['AJ_mean_ci95']
        m,lo,hi=row;labels.append(label);means.append(m);errors.append([m-lo,hi-m])
    bench.barh(labels[::-1],means[::-1],xerr=np.array(errors)[::-1].T,color='#397caf',capsize=2)
    for i,(m,e) in enumerate(zip(means[::-1],errors[::-1],strict=True)):bench.text(m+e[1]+.12,i,f'+{m:.2f}',va='center',fontsize=8)
    bench.axvline(0,lw=.6,color='grey');bench.set(xlabel='Paired AJ gain (percentage points), video bootstrap 95% CI',
        title='(c) Frozen encoder-only clock, head fixed within each backbone',xlim=(-.1,max(m+e[1] for m,e in zip(means,errors,strict=True))+1.6))
    fig.tight_layout(pad=1.2,h_pad=1.8,w_pad=1.5);main=root/'figures/temporal_content_calibration.png'
    fig.savefig(main,dpi=220,bbox_inches='tight');fig.savefig(main.with_suffix('.pdf'),bbox_inches='tight');plt.close(fig)
    d=json.loads((root/'results/clock_initialization_summary.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(7.5,2.9));titles=['Frame-zero query','First appearance','Continuous query at frame 8']
    for ax,(key,g),title in zip(axes,d['groups'].items(),titles,strict=True):
        matrix=np.array([[g['pck'][f'query{q}_target{t}'][0] for t in [0,1]] for q in [0,1]])
        im=ax.imshow(matrix,vmin=40,vmax=75,cmap='viridis')
        for q in [0,1]:
            for t in [0,1]:ax.text(t,q,f'{matrix[q,t]:.1f}',ha='center',va='center',color='white' if matrix[q,t]<64 else 'black',fontsize=10)
        ax.set_xticks([0,1],['Original','Aligned']);ax.set_yticks([0,1],['Original','Aligned'])
        ax.set(xlabel='Target clock',ylabel='Query clock',title=f'{title}\n{g["queries"]} queries, {g["videos"]} videos')
        ax.tick_params(labelsize=7.5);ax.title.set_fontsize(8)
    fig.subplots_adjust(left=.08,right=.9,bottom=.23,top=.78,wspace=.65)
    cax=fig.add_axes([.925,.25,.015,.49]);fig.colorbar(im,cax=cax,label='Direct PCK (%)')
    factorial=root/'figures/query_target_clock_factorial.png';fig.savefig(factorial,dpi=220,bbox_inches='tight');fig.savefig(factorial.with_suffix('.pdf'),bbox_inches='tight');plt.close(fig)
    return main,factorial

if __name__=='__main__':
    root=Path('/share/yty/heft');make_clock_figures(root,json.loads((root/'results/temporal_calibration_summary.json').read_text()),
        json.loads((root/'results/vae_temporal_alignment.json').read_text()))
