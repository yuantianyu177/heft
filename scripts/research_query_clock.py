"""Test the pre-recorded query-relative clock prediction without fitting it."""
import hashlib,json
from pathlib import Path
import numpy as np
from research_track import official_metrics

def ci(x):
    x=np.asarray(x,float);x=x[np.isfinite(x)]
    if not len(x):return None
    r=np.random.default_rng(42);a=x[r.integers(len(x),size=(10000,len(x)))].mean(1)
    return [float(x.mean()),*np.quantile(a,[.025,.975]).tolist()]

def projected_lag(gt,pred,occ,valid,query):
    velocity=(gt[:,2:]-gt[:,:-2])*128;error=(pred[:,1:-1]-gt[:,1:-1])*256
    power=(velocity**2).sum(-1);visible=(~occ)&valid
    mask=visible[:,1:-1]&visible[:,:-2]&visible[:,2:]&(power>.25)
    mask&=np.arange(1,gt.shape[1]-1)[None]>query[:,0,None]
    lag=-(error*velocity).sum(-1)/np.maximum(power,1e-20)
    return float(np.median(lag[mask])) if mask.any() else None

def main():
    root=Path('/share/yty/heft');base=root/'results';evaluate,_=official_metrics(root)
    folders={'native':'benchmark_davis_polyphase','calibrated':'benchmark_davis_polyphase_corrected',
        'uniform_posthoc':'posthoc_davis_official_polyphase_chunk_1p25',
        'queryaware_posthoc':'posthoc_davis_official_polyphase_queryaware_1p25'}
    records={};q0check=[]
    for method,folder in folders.items():
        rows={'query_frame0':[],'query_later':[]}
        for p in sorted((base/folder).glob('*_l18h05_post_full.npz')):
            with np.load(p) as f:
                gt=f['gt'];pred=f['tracks'];vis=f['visibility'];occ=f['gt_occluded'];valid=f['gt_valid'];query=f['query_points']
            for label,selected in [('query_frame0',query[:,0]==0),('query_later',query[:,0]>0)]:
                if not selected.any():continue
                gt_s,pred_s,vis_s,occ_s,valid_s,q_s=[x[selected] for x in [gt,pred,vis,occ,valid,query]]
                metric=evaluate(q_s[None],occ_s[None],gt_s[None]*256,(~vis_s)[None],pred_s[None]*256,'first',valid_mask=valid_s[None])
                row={'video':p.stem.split('_l18h05')[0],'queries':int(selected.sum()),'AJ':float(np.mean(metric['average_jaccard']))*100,
                     'lag':projected_lag(gt_s,pred_s,occ_s,valid_s,q_s)}
                rows[label].append(row)
                if method in ['native','calibrated'] and label=='query_frame0':
                    old='tracking_wan_native_polyphase_full' if method=='native' else 'tracking_wan_native_polyphase_latency125_full'
                    with np.load(base/old/p.name) as reference:
                        assert reference['tracks'].shape==pred_s.shape
                        error=float(np.max(np.abs(reference['tracks']-pred_s)))
                    q0check.append({'method':method,'video':row['video'],'max_normalized_track_difference':error})
        records[method]=rows
    summary={'hypothesis':json.loads((base/'relative_clock_hypothesis.json').read_text()),'methods':{},'comparisons':{},
        'firstvisible_wrapper_consistency':{'checks':len(q0check),'maximum_normalized_track_difference':max(r['max_normalized_track_difference'] for r in q0check)}}
    for method,groups in records.items():
        summary['methods'][method]={}
        for group,rows in groups.items():
            summary['methods'][method][group]={'videos':len(rows),'queries':sum(r['queries'] for r in rows),
                'AJ':float(np.mean([r['AJ'] for r in rows])),'video_median_lag_mean_ci95':ci([r['lag'] for r in rows if r['lag'] is not None]),'records':rows}
    for group in ['query_frame0','query_later']:
        before={r['video']:r for r in records['native'][group]}
        for method in ['calibrated','uniform_posthoc','queryaware_posthoc']:
            after={r['video']:r for r in records[method][group]};assert after.keys()==before.keys(),(method,group)
            summary['comparisons'][group+'/'+method+' minus native']={'AJ_mean_ci95':ci([after[n]['AJ']-before[n]['AJ'] for n in sorted(before)])}
    summary['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (base/'query_clock_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='hypothesis'},indent=2))

if __name__=='__main__':main()
