"""Function-preserving QK gauges versus extracted key correspondence geometry.

Theory of QK gauge symmetries is established prior work. The experiments here
test their consequences for video feature extraction and two function-defined
readouts, rather than claiming discovery of the symmetry itself.
"""
import argparse, hashlib, json, time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from research_probe import annotations, grid_points, probe
from research_rope import frequency_pairs, pair_channels

VARIANTS=['post_full','post_low15','gauge_fast16','gauge_slow16','gauge_random',
          'balanced_pairs','balanced_pairs_low15','query_metric','query_metric_centered',
          'query_covariance_centered','key_mahalanobis','post_centered',
          'balanced_centered','query_row_profile','bidirectional_profile']


def second_moment(values,center=False):
    flat=values.flatten(1,3).double()
    if center:flat=flat-flat.mean(1,keepdim=True)
    return flat.transpose(-1,-2)@flat/flat.shape[1]


def spectral_factor(matrix,inverse=False):
    eigenvalues,vectors=torch.linalg.eigh(matrix)
    if inverse:
        # Explicitly report relative rank truncation: not exact GL invariance
        # on singular or poorly conditioned subspaces.
        threshold=eigenvalues.amax(-1,keepdim=True)*1e-8
        scale=torch.where(eigenvalues>threshold,eigenvalues.clamp_min(1e-30).rsqrt(),0)
    else:scale=eigenvalues.clamp_min(0).sqrt()
    return (vectors*scale[...,None,:]).float()


def readouts(query,key):
    qenergy=query.reshape(*query.shape[:-1],64,2).double().square().sum(-1).mean((1,2,3))
    kenergy=key.reshape(*key.shape[:-1],64,2).double().square().sum(-1).mean((1,2,3))
    balanced=(qenergy/kenergy.clamp_min(1e-30)).pow(.25).float().repeat_interleave(2,dim=-1)
    high=pair_channels(frequency_pairs(0,.15)).to(key.device)
    keep=pair_channels(frequency_pairs(.15,1)).to(key.device)
    fast=torch.ones(128,device=key.device);fast[high]=16
    slow=torch.ones(128,device=key.device);slow[keep]=16
    random=torch.pow(2.,torch.randint(-4,5,(64,),generator=torch.Generator().manual_seed(7382))).repeat_interleave(2).to(key.device)
    centered=key-key.mean((1,2,3),keepdim=True)
    query_factor=spectral_factor(second_moment(query))
    covariance_factor=spectral_factor(second_moment(query,True))
    whitening_factor=spectral_factor(second_moment(key,True),inverse=True)
    key_covariance_factor=spectral_factor(second_moment(key,True))
    centered_kenergy=centered.reshape(*key.shape[:-1],64,2).double().square().sum(-1).mean((1,2,3))
    centered_balance=(qenergy/centered_kenergy.clamp_min(1e-30)).pow(.25).float().repeat_interleave(2,dim=-1)
    # Standard matmul broadcasts the same per-head matrix across all tokens.
    def multiply(value,factor):
        return (value.flatten(1,3)@factor).reshape_as(value)
    column_profile=multiply(centered,query_factor)
    row_profile=multiply(query,key_covariance_factor)
    return {'post_full':key,'post_low15':key[...,keep],
            'gauge_fast16':key*fast,'gauge_slow16':key*slow,'gauge_random':key*random,
            'balanced_pairs':key*balanced[:,None,None,None],
            'balanced_pairs_low15':(key*balanced[:,None,None,None])[...,keep],
            'query_metric':multiply(key,query_factor),
            'query_metric_centered':column_profile,
            'query_covariance_centered':multiply(centered,covariance_factor),
            'key_mahalanobis':multiply(centered,whitening_factor),
            'post_centered':centered,
            'balanced_centered':centered*centered_balance[:,None,None,None],
            'query_row_profile':row_profile,
            'bidirectional_profile':torch.cat((torch.nn.functional.normalize(column_profile,dim=-1),
                                               torch.nn.functional.normalize(row_profile,dim=-1)),dim=-1)},balanced


def sanity():
    torch.manual_seed(7)
    q=torch.randn(2,3,2,3,128);k=torch.randn_like(q)
    # Pairwise scaling is in the commutant of all axial rotations.
    gauge=torch.pow(2.,torch.randint(-4,5,(64,))).repeat_interleave(2)
    before,_=readouts(q,k);after,_=readouts(q/gauge,k*gauge)
    results={}
    for variant in ['balanced_pairs','query_metric','query_metric_centered','query_covariance_centered',
                    'balanced_centered','query_row_profile','bidirectional_profile']:
        a=before[variant].flatten(1,3);b=after[variant].flatten(1,3)
        gram_a=a@a.transpose(-1,-2);gram_b=b@b.transpose(-1,-2)
        results[variant]=float((gram_a-gram_b).abs().max()/gram_a.abs().max())
        assert results[variant]<2e-5,results
    qflat=q.flatten(1,3);kflat=k.flatten(1,3)
    results['logit_relative_error']=float(((qflat/gauge)@(kflat*gauge).transpose(-1,-2)-qflat@kflat.transpose(-1,-2)).abs().max())
    assert results['logit_relative_error']<1e-5
    return results


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('/share/yty/heft'))
    parser.add_argument('--features',default='wan_real_t25')
    parser.add_argument('--gpu',type=int,default=7)
    parser.add_argument('--names')
    parser.add_argument('--layers',default='all')
    parser.add_argument('--sanity',action='store_true')
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args();torch.set_num_threads(3)
    if args.sanity:print(json.dumps(sanity(),indent=2));return
    device=f'cuda:{args.gpu}';torch.cuda.set_device(device)
    source=args.root/'features'/args.features
    paths=sorted(source.glob('*.safetensors'))
    if args.names:paths=[p for p in paths if p.stem in args.names.split(',')]
    output_root=args.root/f'results/gauge_geometry_{args.features}';output_root.mkdir(exist_ok=True,parents=True)
    for path in paths:
        output=output_root/path.stem
        if output.with_suffix('.npz').exists() and not args.overwrite:continue
        started=time.monotonic();meta=json.loads(path.with_suffix('.json').read_text())
        layers=list(range(meta['layers'])) if args.layers=='all' else list(map(int,args.layers.split(',')))
        points,truth,valid=annotations(args.root,path.stem,meta,device)
        rows=[];balance=[]
        with safe_open(path,framework='pt',device=str(device)) as file:
            for layer in layers:
                query=file.get_tensor(f'l{layer:02d}_query').float()
                key=file.get_tensor(f'l{layer:02d}_key').float()
                variants,scales=readouts(query,key)
                rows.append([probe(variants[v],points,truth,valid,grid_points(device)) for v in VARIANTS])
                balance.append(scales.cpu().numpy())
        errors=np.stack([[r['errors'] for r in row] for row in rows]).transpose(0,2,1,3,4)
        np.savez_compressed(output.with_suffix('.npz'),errors=errors,valid=valid.cpu().numpy(),variant_names=VARIANTS,
                            layers=layers,balance=balance)
        info={'video':path.stem,'feature_metadata':meta,'seconds':time.monotonic()-started,
              'selection_annotation_access':False,'evaluation_annotation_access':True,
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'protocol':'fixed label-free metrics; gauges preserve QK logits; correspondence is evaluated independently'}
        output.with_suffix('.json').write_text(json.dumps(info,indent=2))
        score=((errors[...,None]<np.array([1,2,4,8,16])).mean(-1)*valid.cpu().numpy()).sum((-1,-2))/valid.sum().item()*100
        print(json.dumps({'video':path.stem,'seconds':info['seconds'],'population':dict(zip(VARIANTS,score.mean((0,1)).tolist()))}),flush=True)


if __name__=='__main__':main()
