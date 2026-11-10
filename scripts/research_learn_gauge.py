"""Calibrate positive rotary-pair metrics from known image warps only.

This is self-supervised metric learning, not a training-free method. The 64
scales per head lie in a QK function-preserving gauge: they can be folded into
K RMSNorm weights with reciprocal Q changes. The generator need not change.
Neither human trajectories nor real-video scores are read by calibration.
"""
import argparse, hashlib, json, math, time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from torch.nn import functional as F

from research_probe import sample, grid_points, annotations, probe


def load_calibration(args):
    records=[]
    tags=args.calibration_features.split(',') if args.calibration_features else ['wan_translate_t25','wan_translate_appearance_t25']
    for tag in tags:
        source=args.root/'features'/tag
        for path in sorted(source.glob('*.safetensors')):
            metadata=json.loads(path.with_suffix('.json').read_text())
            assert metadata['split']=='calibration' and metadata['transform']!='real'
            with safe_open(path,framework='pt',device='cpu') as file:
                dense=all(f'l{layer:02d}_h{head:02d}_dense_key' in file.keys() for layer,head in args.head_ids)
                if args.dense_calibration:
                    assert dense
                    features=torch.stack([file.get_tensor(f'l{layer:02d}_h{head:02d}_dense_key') for layer,head in args.head_ids])
                else:
                    bylayer={layer:file.get_tensor(f'l{layer:02d}_key') for layer in args.layers}
                    features=torch.stack([bylayer[layer][head] for layer,head in args.head_ids])
            h,w=metadata['source_size'];frames=list(range(metadata['frames'])) if args.dense_calibration else metadata['sparse_frames']
            shifts=np.asarray(metadata['shifts'],np.float32)[frames]/np.array([w,h],np.float32)
            records.append({'name':path.stem,'transform':metadata['transform'],'features':features,
                            'shifts':torch.from_numpy(shifts),'source':str(path),
                            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    expected=7*len(tags)
    assert len(records)==expected,(len(records),expected)
    return records


def freeze_config(args,records):
    config={'layers':args.layers,'heads_per_layer':12,'parameters_per_head':64,'steps':args.steps,
            'head_ids':args.head_ids,'calibration_features':args.calibration_features,
            'dense_calibration':args.dense_calibration,
            'learning_rate':.05,'temperature':.07,'log_metric_regularization':.001,
            'maximum_amplitude':16,'grid_queries_per_image':64,'queries_per_update':32,
            'optimizer':'Adam','seed':2026,'checkpoint_selection':'last step fixed before real-video evaluation',
            'primary_readout':'power-of-two rounded pair amplitudes; continuous amplitudes secondary diagnostic',
            'uses_human_annotations':False,'uses_real_video_scores':False,
            'training_sources':[{'name':r['name'],'transform':r['transform'],'sha256':r['sha256']} for r in records],
            'status':'exploratory method designed after DAVIS diagnostics; PointOdyssey still untouched',
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    path=args.root/'results'/f'{args.tag}_protocol.json'
    if path.exists():assert json.loads(path.read_text())==config,'protocol already fixed differently'
    else:path.write_text(json.dumps(config,indent=2))
    return config


def calibrate(args):
    torch.manual_seed(2026);np.random.seed(2026)
    started=time.monotonic();records=load_calibration(args);config=freeze_config(args,records)
    device=f'cuda:{args.gpu}';torch.cuda.set_device(device)
    for record in records:
        record['features']=record['features'].to(device)
        record['shifts']=record['shifts'].to(device)
    nh=len(args.head_ids)
    theta=torch.nn.Parameter(torch.zeros(nh,64,device=device))
    optimizer=torch.optim.Adam([theta],lr=.05)
    points=grid_points(device)
    history=[]
    for step in range(args.steps):
        record=records[np.random.randint(len(records))]
        features=record['features'];frame=int(np.random.randint(1,features.shape[1]))
        chosen=torch.randperm(len(points),device=device)[:32];source_points=points[chosen]
        truth=source_points+record['shifts'][frame]
        source=sample(features[:,:1].float(),source_points[None])[:,0]
        target=features[:,frame].flatten(1,2).float()
        scale=theta.exp().sqrt().repeat_interleave(2,-1)
        src=F.normalize(source*scale[:,None],dim=-1)
        dst=F.normalize(target*scale[:,None],dim=-1)
        similarity=torch.bmm(src,dst.transpose(-1,-2))/.07
        logprob=similarity.log_softmax(-1)
        h,w=features.shape[2:4]
        grid=truth*torch.tensor([w,h],device=device)-.5
        lower=grid.floor().long();fraction=grid-lower
        per_head_loss=torch.zeros(nh,device=device)
        for dx,dy in [(0,0),(0,1),(1,0),(1,1)]:
            xy=lower+torch.tensor([dx,dy],device=device)
            index=(xy[:,1].clamp(0,h-1)*w+xy[:,0].clamp(0,w-1))[None,:,None].expand(nh,-1,-1)
            weight=(fraction[:,0] if dx else 1-fraction[:,0])*(fraction[:,1] if dy else 1-fraction[:,1])
            per_head_loss-= (logprob.gather(-1,index)[:,:,0]*weight[None]).mean(-1)
        loss=per_head_loss.mean()+.001*theta.square().mean()
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        with torch.no_grad():
            theta-=theta.mean(-1,keepdim=True)
            theta.clamp_(-2*math.log(16),2*math.log(16))
        if step%50==0 or step==args.steps-1:
            item={'step':step+1,'loss':float(loss.detach()),'per_head_cross_entropy':per_head_loss.detach().cpu().tolist()}
            history.append(item);print(json.dumps({'step':step+1,'loss':item['loss']}),flush=True)
    scales=theta.detach().exp().sqrt()
    payload={'config':config,'scales':scales.cpu().tolist(),
             'scales_pow2':torch.pow(2.,scales.log2().round()).cpu().tolist(),
             'head_ids':args.head_ids,
             'history':history,'seconds':time.monotonic()-started}
    destination=args.root/'results'/f'{args.tag}.json'
    assert not destination.exists(),'frozen metrics already exist'
    destination.write_text(json.dumps(payload,indent=2))
    print(json.dumps({'file':str(destination),'seconds':payload['seconds']}),flush=True)


@torch.inference_mode()
def evaluate(args):
    device=f'cuda:{args.gpu}';torch.cuda.set_device(device)
    path=args.root/'results'/f'{args.tag}.json';frozen=json.loads(path.read_text())
    ids=frozen['head_ids'];scales=torch.tensor(frozen['scales'],device=device).repeat_interleave(2,-1)
    rounded=torch.tensor(frozen['scales_pow2'],device=device).repeat_interleave(2,-1)
    destination=args.root/f'results/probe_{args.tag}_{args.features}';destination.mkdir(exist_ok=True,parents=True)
    for feature_path in sorted((args.root/f'features/{args.features}').glob('*.safetensors')):
        output=destination/feature_path.stem
        if output.with_suffix('.npz').exists():continue
        meta=json.loads(feature_path.with_suffix('.json').read_text())
        points,truth,valid=annotations(args.root,feature_path.stem,meta,device)
        errors=[];rounded_errors=[]
        with safe_open(feature_path,framework='pt',device=str(device)) as file:
            for layer in frozen['config']['layers']:
                selected=[i for i,(l,h) in enumerate(ids) if l==layer]
                key=file.get_tensor(f'l{layer:02d}_key')[[ids[i][1] for i in selected]].float()
                scale=scales[selected,None,None,None]
                result=probe(key*scale,points,truth,valid,grid_points(device))
                errors.append(result['errors'])
                result=probe(key*rounded[selected,None,None,None],points,truth,valid,grid_points(device))
                rounded_errors.append(result['errors'])
        np.savez_compressed(output.with_suffix('.npz'),errors=np.stack(errors),valid=valid.cpu().numpy(),
                            errors_pow2=np.stack(rounded_errors),head_ids=ids,
                            metric_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        print(json.dumps({'video':feature_path.stem,'mean_direct_pck':float(((np.stack(errors)[...,None]<[1,2,4,8,16]).mean(-1)*valid.cpu().numpy()).sum()/len(ids)/valid.sum().item()*100)}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['calibrate','evaluate'])
    parser.add_argument('--root',type=Path,default=Path('/share/yty/heft'));parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--layers',default='15,18,20');parser.add_argument('--steps',type=int,default=600)
    parser.add_argument('--tag',default='synthetic_gauge_v1');parser.add_argument('--features',default='wan_real_t25')
    parser.add_argument('--heads',help='Optional fixed subset, e.g.18:5')
    parser.add_argument('--calibration-features',help='Comma-separated feature folders; default two original synthetic pools')
    parser.add_argument('--dense-calibration',action='store_true')
    args=parser.parse_args();args.layers=list(map(int,args.layers.split(',')));torch.set_num_threads(3)
    args.head_ids=[list(map(int,x.split(':'))) for x in args.heads.split(',')] if args.heads else [[l,h] for l in args.layers for h in range(12)]
    args.layers=list(dict.fromkeys(l for l,h in args.head_ids))
    if args.mode=='calibrate':calibrate(args)
    else:evaluate(args)


if __name__=='__main__':main()
