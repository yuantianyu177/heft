"""Complete-benchmark summaries and query-amortized response diagnostics."""
import argparse,csv,json
from pathlib import Path
import numpy as np,torch
from safetensors.torch import load_file
from torch.nn import functional as F
from research_artifacts import response_stem
from research_decode_response import queries,metrics
from research_rope import frequency_pairs,pair_channels

def ci(x):
    x=np.asarray(x,float);r=np.random.default_rng(42)
    return [float(x.mean()),*np.quantile(x[r.integers(len(x),size=(10000,len(x)))].mean(1),[.025,.975]).tolist()]

def benchmark(root):
    folder=root/'rgb_stacking/results';rows={};out={'videos':50,'query_mode':'official first-visible, query frame excluded','configurations':{}}
    for tag in ['raw','normalized_mode_time025','native_polyphase','native_interpolate','native_phase_interpolate','native_polyphase_anchorfirst']:
        for variant in ['post_full','post_low15']:
            values={}
            for path in sorted((folder/f'benchmark_{tag}').glob(f'*_l18h05_{variant}.json')):
                d=json.loads(path.read_text());values[d['video']]=[d['metrics'][m] for m in ['average_jaccard','average_pts_within_thresh','occlusion_accuracy']]
            if not values:continue
            assert len(values)==50,(tag,variant,len(values))
            key=tag+'/'+variant;rows[key]=values;a=np.array(list(values.values()))
            out['configurations'][key]={'n':len(a),'AJ':float(a[:,0].mean()),'delta_average':float(a[:,1].mean()),'OA':float(a[:,2].mean())}
    comparisons=[('raw/post_full','raw/post_low15'),('normalized_mode_time025/post_full','normalized_mode_time025/post_low15'),
        ('raw/post_full','normalized_mode_time025/post_full'),('normalized_mode_time025/post_full','native_polyphase/post_full'),
        ('normalized_mode_time025/post_full','native_interpolate/post_full'),('native_polyphase/post_full','native_interpolate/post_full'),
        ('native_interpolate/post_full','native_phase_interpolate/post_full'),('native_polyphase_anchorfirst/post_full','native_polyphase/post_full')]
    out['paired_comparisons']={}
    for before,after in comparisons:
        x=np.array([np.array(rows[after][n])-rows[before][n] for n in sorted(rows[before])])
        out['paired_comparisons'][after+' minus '+before]={'AJ_mean_ci95':ci(x[:,0]),'delta_average_mean_ci95':ci(x[:,1])}
    out['interpretation_limit']='raw to normalized_mode_time025 bundles normalization, posterior mode, noise precision and time coordinates; it is not a pure normalization ablation'
    (root/'results/rgb_stacking_summary.json').write_text(json.dumps(out,indent=2))
    return out

@torch.inference_mode()
def head_controls(root,name,device):
    path=response_stem(root,'rademacher',name,32,.5).with_suffix('.safetensors')
    saved=load_file(str(path),device=device)
    if 'baseline_l18_key' not in saved:return None
    points,valid=queries(root,name,0);out={}
    for head,feature in zip([1,5],saved['baseline_l18_key'].float(),strict=True):
        for variant,channels in [('full',torch.arange(128,device=device)),('low15',pair_channels(frequency_pairs(.15,1)).to(device))]:
            x=feature[...,channels].permute(0,3,1,2)
            for upsample in [False,True]:
                target=F.interpolate(x,size=(60,104),mode='bilinear',align_corners=False) if upsample else x
                source=F.grid_sample(x[:1],torch.tensor(points[0],device=device)[None,None]*2-1,align_corners=False,padding_mode='border')[0,:,0].T
                field=torch.einsum('qc,tchw->qthw',F.normalize(source,dim=-1),F.normalize(target[1:],dim=1)).flatten(2)
                h,w=target.shape[-2:];label=f'l18h{head}_{variant}_'+('fine' if upsample else 'coarse')
                out[label]=metrics(field,points,valid,h,w)[0]
    return out

def sketches(root,device):
    names=['gold-fish','horsejump-high','bike-packing','blackswan','bmx-trees','breakdance'];records={}
    for name in names:
        stem=response_stem(root,'rademacher',name,32,.5);points,_=queries(root,name,0)
        path=stem.with_name(stem.name+f'_decoded_q{len(points[0])}').with_suffix('.json')
        if not path.exists():continue
        record=json.loads(path.read_text());record['head_controls_same_sigma']=head_controls(root,name,device)
        records[name]=record
    summary={'evaluation':'all initially visible points, targets4,8,12,16,20,24; direct PCK, not benchmark AJ',
             'records':records,'validation_videos':names[2:]}
    validation=[records[n] for n in names[2:] if n in records]
    if len(validation)==4:
        summary['validation_means']={k:float(np.mean([d['pck'][k] for d in validation])) for k in validation[0]['pck']}
        summary['validation_head_means']={k:float(np.mean([d['head_controls_same_sigma'][k] for d in validation])) for k in validation[0]['head_controls_same_sigma']}
    (root/'results/response_sketch_summary.json').write_text(json.dumps(summary,indent=2));return summary

def plot(root,rgb,sketch):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.2))
    keys=['raw/post_full','raw/post_low15','normalized_mode_time025/post_full','native_polyphase/post_full','native_interpolate/post_full']
    labels=['Original full','Original low15','Matched dense','Native polyphase','Native interpolate']
    values=[rgb['configurations'][k]['AJ'] for k in keys]
    axes[0].barh(labels[::-1],values[::-1],color=['#2563a7','#4d92cc','#a6c4db','#b181b8','#cf7479'][::-1])
    for j,v in enumerate(values[::-1]):axes[0].text(v+.3,j,f'{v:.2f}',va='center',fontsize=9)
    axes[0].set(xlabel='Average Jaccard (%)',title='RGB-Stacking: all 50 videos, all 250 frames',xlim=(0,max(values)+6))
    names=[n for n in sketch['validation_videos'] if n in sketch['records']]
    if len(names)==4:
        x=np.arange(4);width=.24
        fields=[('k16_nearest_explained_fine','64 model inputs'),('k32_nearest_explained_fine','128 model inputs')]
        for i,(field,label) in enumerate(fields):
            axes[1].bar(x+(i-1)*width,[sketch['records'][n]['pck'][field] for n in names],width,label=label)
        axes[1].bar(x+width,[sketch['records'][n]['head_controls_same_sigma']['l18h5_full_fine'] for n in names],width,label='Key head18/5, same noise')
        axes[1].set_xticks(x,names,rotation=20);axes[1].set(ylabel='Direct PCK (%)',title='Fixed decoder: four additional clips')
        axes[1].legend(fontsize=8)
    fig.tight_layout();path=root/'figures/benchmark_and_coded_validation.png'
    fig.savefig(path,dpi=180,bbox_inches='tight');fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight')

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/share/yty/heft'));p.add_argument('--gpu',type=int,default=2);a=p.parse_args()
    torch.set_num_threads(3);device=f'cuda:{a.gpu}';torch.cuda.set_device(device)
    rgb=benchmark(a.root);sketch=sketches(a.root,device);plot(a.root,rgb,sketch)
    print(json.dumps({'rgb':rgb,'sketch':sketch},indent=2))

if __name__=='__main__':main()
