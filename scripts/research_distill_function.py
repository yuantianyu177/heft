"""Calibrate key metrics against unlabelled generator response distributions.

The original generator supplies targets through finite input perturbations.
No human coordinates/visibility/motion annotations are opened. A balanced-pair
canonical basis removes positive pairwise QK rescaling before optimization.
Leave-one-video-out target losses select among four predeclared saved heads.
"""
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np,torch
from safetensors.torch import load_file
from torch.nn import functional as F
from research_probe import sample


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,default=7);parser.add_argument('--steps',type=int,default=600)
    args=parser.parse_args();torch.set_num_threads(3);device=f'cuda:{args.gpu}';torch.cuda.set_device(device)
    root=Path('/share/yty/heft');output=root/'results/functional_gauge_v1.json'
    if output.exists():raise FileExistsError('function-derived metric already frozen')
    design=json.loads((root/'results/functional_teacher_design.json').read_text())
    balance=json.loads((root/'results/functional_student_balance.json').read_text());ids=balance['head_ids']
    base=torch.tensor(balance['scales'],device=device).repeat_interleave(2,-1)
    candidates=[[15,2],[18,1],[18,5],[20,8]]
    protocol={'teacher_design':design,'parameters_per_head':64,'heads':ids,'candidate_heads':candidates,
              'canonical_basis':'global balanced rotary pair energies Q/K, fixed from all four unlabelled student inputs',
              'loss':'response-energy-weighted spatial cross entropy; each response map normalized over target positions',
              'temperature':.07,'optimizer':'Adam','learning_rate':.05,'log_metric_regularization':.001,
              'steps':args.steps,'maximum_relative_amplitude':16,'seed':2026,
              'selection':'minimum leave-one-video-out response target cross entropy among four fixed candidates',
              'primary_metric':'continuous amplitudes','secondary_metric':'powers-of-two rounded amplitudes',
              'uses_human_annotations':False,'uses_RGB_Stacking':False,
              'heldout_note':'response targets are held out in each fold; the unlabelled canonical energy basis uses all four videos',
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    protocol_path=root/'results/functional_gauge_v1_protocol.json'
    if protocol_path.exists():assert json.loads(protocol_path.read_text())==protocol
    else:protocol_path.write_text(json.dumps(protocol,indent=2))
    records=[]
    for name in design['videos']:
        path=root/f'results/functional_transport_wan/{name}_sigma0500_p4_q16_fp32_grid4.npz'
        data=np.load(path);assert not bool(data['ground_truth_available'])
        meta=json.loads(path.with_suffix('.json').read_text());assert not meta['human_annotation_access']
        features=load_file(str(path.with_name(path.stem+'_student.safetensors')),device=device)
        key=torch.cat([features[f'l{l:02d}_key'] for l in design['student_layers']]).float()
        points=torch.tensor(data['query_points'],device=device)
        dimensions=torch.tensor([52,30],device=device)
        centers=((points*dimensions).floor()+.5)/dimensions
        source=sample(key[:,:1],centers[None])[:,0]*base[:,None]
        target=key[:,1:].flatten(2,3)*base[:,None,None]
        energy=torch.tensor(data['energy'][0],device=device).permute(1,0,2,3).flatten(2)
        mass=energy.sum(-1);probability=energy/mass[...,None].clamp_min(1e-30)
        weights=mass/mass.mean().clamp_min(1e-30)
        records.append({'name':name,'source':source,'target':target,'probability':probability,'weights':weights,
                        'response_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    def loss_for(theta,record,frame):
        scale=(theta*.5).exp().repeat_interleave(2,-1)
        source=F.normalize(record['source']*scale[:,None],dim=-1)
        target=F.normalize(record['target'][:,frame]*scale[:,None],dim=-1)
        logits=torch.bmm(source,target.transpose(-1,-2))/.07
        ce=-(logits.log_softmax(-1)*record['probability'][frame][None]).sum(-1)
        return (ce*record['weights'][frame][None]).mean(-1)
    def fit(indices):
        theta=torch.nn.Parameter(torch.zeros(len(ids),64,device=device))
        optimizer=torch.optim.Adam([theta],lr=.05);rng=np.random.default_rng(2026)
        for step in range(args.steps):
            record=records[indices[int(rng.integers(len(indices)))]];frame=int(rng.integers(6))
            losses=loss_for(theta,record,frame);loss=losses.mean()+.001*theta.square().mean()
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            with torch.no_grad():theta-=theta.mean(-1,keepdim=True);theta.clamp_(-2*math.log(16),2*math.log(16))
        return theta.detach()
    @torch.no_grad()
    def evaluate(theta,record):return torch.stack([loss_for(theta,record,t) for t in range(6)]).mean(0).cpu().numpy()
    started=time.monotonic();heldout=[];baseline=[]
    for fold in range(len(records)):
        theta=fit([i for i in range(len(records)) if i!=fold])
        heldout.append(evaluate(theta,records[fold]));baseline.append(evaluate(torch.zeros_like(theta),records[fold]))
        print(json.dumps({'heldout_video':records[fold]['name'],'mean_response_loss':float(np.mean(heldout[-1])),
                          'mean_canonical_baseline_loss':float(np.mean(baseline[-1]))}),flush=True)
    heldout=np.array(heldout);baseline=np.array(baseline)
    candidate_indices=[ids.index(head) for head in candidates]
    selected=min(candidate_indices,key=lambda i:heldout[:,i].mean())
    theta=fit(list(range(len(records))));scale=base[:,::2]*(theta*.5).exp()
    payload={'config':protocol,'head_ids':ids,'scales':scale.cpu().tolist(),
             'scales_pow2':torch.pow(2.,scale.log2().round()).cpu().tolist(),
             'selected_head':ids[selected],'selection_loss':float(heldout[:,selected].mean()),
             'crossvalidation_response_loss':heldout.tolist(),'crossvalidation_canonical_loss':baseline.tolist(),
             'training_response_loss':[evaluate(theta,r).tolist() for r in records],
             'response_sources':[{'video':r['name'],'sha256':r['response_sha256']} for r in records],
             'seconds':time.monotonic()-started,'uses_human_annotations':False}
    output.write_text(json.dumps(payload,indent=2))
    print(json.dumps({'selected':payload['selected_head'],'heldout_loss':payload['selection_loss'],
                      'file':str(output),'seconds':payload['seconds']}),flush=True)

if __name__=='__main__':main()
