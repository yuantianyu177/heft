"""Motion-conditional and phase-conditional native extraction comparisons."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

def ci(x):
    x=np.asarray(x,float);x=x[np.isfinite(x)]
    if not len(x):return None
    r=np.random.default_rng(42);means=x[r.integers(len(x),size=(10000,len(x)))].mean(1)
    return [float(x.mean()),*np.quantile(means,[.025,.975]).tolist()]

def main():
    root=Path('/share/yty/heft');summary={};bins=[0,.5,1,2,4,8,float('inf')]
    for label,base,tag in [('rgb_stacking',root/'rgb_stacking/results','benchmark_native_'),('davis_initial_visible',root/'results','tracking_wan_native_')]:
        suffix='' if label=='rgb_stacking' else '_full'
        interpolate=base/(tag+'interpolate'+suffix);poly=base/(tag+'polyphase'+suffix)
        rows=[];conditional=[];phases=[]
        for path in sorted(interpolate.glob('*_l18h05_post_full.npz')):
            other=poly/path.name
            if not other.exists():continue
            with np.load(path) as f:
                gt=f['gt'];valid=(~f['gt_occluded'])&f['gt_valid'];query=f['query_points'][:,0].astype(int)
                valid&=np.arange(gt.shape[1])[None]>query[:,None]
                a=np.linalg.norm((f['tracks']-gt)*256,axis=-1)
            with np.load(other) as f:b=np.linalg.norm((f['tracks']-gt)*256,axis=-1)
            score_a=(a[...,None]<[1,2,4,8,16]).mean(-1)*100;score_b=(b[...,None]<[1,2,4,8,16]).mean(-1)*100
            speed=np.concatenate((np.zeros_like(a[:,:1]),np.linalg.norm(np.diff(gt,axis=1)*256,axis=-1)),axis=1)
            delta=score_b-score_a;record=[]
            for low,high in zip(bins[:-1],bins[1:],strict=True):
                mask=valid&(speed>=low)&(speed<high);record.append(float(delta[mask].mean()) if mask.any() else np.nan)
            conditional.append(record);phase=[]
            for i in range(4):
                # Features are chunked into25-frame contexts; use within-chunk phase.
                mask=valid&((np.arange(a.shape[1])[None]%25)%4==i);phase.append(float(delta[mask].mean()) if mask.any() else np.nan)
            phases.append(phase);rows.append({'video':path.stem.split('_l18h05')[0],'mean_speed':float(speed[valid].mean()),
                'polyphase_minus_interpolation_pck':float(delta[valid].mean())})
        if not rows:continue
        summary[label]={'videos':len(rows),'mean_video_delta_ci95':ci([r['polyphase_minus_interpolation_pck'] for r in rows]),
            'speed_bin_edges':bins[:-1]+['infinity'],'speed_bin_video_delta_ci95':[ci(x) for x in np.array(conditional).T],
            'within_chunk_frame_phase_delta_ci95':[ci(x) for x in np.array(phases).T],
            'speed_vs_delta_spearman':float(spearmanr([r['mean_speed'] for r in rows],[r['polyphase_minus_interpolation_pck'] for r in rows]).statistic),
            'records':rows,'limitation':'descriptive conditional analysis with ground-truth speed; not a motion-adaptive method or held-out selection'}
    (root/'results/native_motion_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
