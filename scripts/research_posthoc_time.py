"""Fixed clock correction after tracking, as a simpler mechanistic control."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from research_track import official_metrics

def main():
    root=Path('/share/yty/heft');began=time.monotonic();design={'delay':1.25,
        'calibration':'same frozen VAE-only estimate; no ground-truth fitting',
        'purpose':'distinguish descriptor clock correction from simply shifting completed trajectories',
        'boundaries':['within each25framechunk','globalvideo'],'smoothing_control':.25,
        'query_anchor':'provided query-frame prediction retained exactly','visibility':'linear interpolation of binary visibility, threshold0.5'}
    (root/'results/posthoc_time_design.json').write_text(json.dumps(design,indent=2))
    datasets=[('davis_initial_visible',root,'tracking_wan_native_polyphase_full'),
        ('davis_official',root,'benchmark_davis_polyphase'),
        ('rgb_stacking',root/'rgb_stacking','benchmark_native_polyphase'),
        ('pointodyssey',root/'pointodyssey','native_transfer_polyphase')]
    result={}
    for label,data,source_tag in datasets:
        evaluator,evaluator_hash=official_metrics(data);outcomes={}
        for variant,delay,boundary in [(v,d,b) for v in ['post_full','post_low15'] for d,b in [(1.25,'chunk'),(.25,'chunk'),(1.25,'global'),(1.25,'queryaware')]]:
            folder=data/'results'/f'posthoc_{label}_polyphase_{boundary}_{str(delay).replace(".","p")}'
            folder.mkdir(parents=True,exist_ok=True);rows=[]
            for p in sorted((data/'results'/source_tag).glob(f'*_l18h05_{variant}.npz')):
                with np.load(p) as f:
                    tracks=f['tracks'];visibility=f['visibility'];gt=f['gt'];occ=f['gt_occluded'];valid=f['gt_valid']
                    if 'query_points' in f:query=f['query_points'].copy()
                    else:query=np.column_stack([np.zeros(len(gt)),gt[:,0,1]*256,gt[:,0,0]*256])
                nt=tracks.shape[1];position=np.arange(nt,dtype=float)+delay
                if boundary in ['chunk','queryaware']:position=np.minimum(position,(np.arange(nt)//25+1)*25-1)
                position=np.clip(position,0,nt-1);left=np.floor(position).astype(int);right=np.ceil(position).astype(int);alpha=position-left
                prediction=tracks[:,left]*(1-alpha)[None,:,None]+tracks[:,right]*alpha[None,:,None]
                visible=(visibility[:,left]*(1-alpha)[None]+visibility[:,right]*alpha[None])>=.5
                qtime=query[:,0].astype(int);prediction[np.arange(len(gt)),qtime]=tracks[np.arange(len(gt)),qtime]
                visible[np.arange(len(gt)),qtime]=visibility[np.arange(len(gt)),qtime]
                if boundary=='queryaware':
                    # Constant-delay approximation: only the first frame has
                    # zero encoder delay. Later source/target clock errors cancel.
                    prediction[qtime>0]=tracks[qtime>0];visible[qtime>0]=visibility[qtime>0]
                metrics=evaluator(query[None],occ[None],gt[None]*256,(~visible)[None],prediction[None]*256,'first',valid_mask=valid[None])
                metrics={k:float(np.mean(v))*100 for k,v in metrics.items()};out=folder/p.stem
                np.savez_compressed(out.with_suffix('.npz'),tracks=prediction,visibility=visible,gt=gt,gt_occluded=occ,gt_valid=valid,query_points=query)
                payload={'video':p.stem.split('_l18h05')[0],'metrics':metrics,'delay':delay,'boundary':boundary,'source':str(p),
                    'evaluator_sha256':evaluator_hash,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'model_forward_calls':0}
                out.with_suffix('.json').write_text(json.dumps(payload,indent=2));rows.append(payload)
            key=('low15_' if variant=='post_low15' else '')+f'{boundary}_{delay}'
            outcomes[key]={'videos':len(rows),'AJ':float(np.mean([r['metrics']['average_jaccard'] for r in rows])),
                'delta_average':float(np.mean([r['metrics']['average_pts_within_thresh'] for r in rows]))}
        result[label]=outcomes
    (root/'results/posthoc_time_summary.json').write_text(json.dumps({'design':design,'results':result,'seconds':time.monotonic()-began},indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
