"""Prepare all50 RGB-Stacking videos without subsampling points or frames."""
import json,pickle
from pathlib import Path
import numpy as np

def main():
    root=Path('/share/yty/heft');target=root/'rgb_stacking'
    source=root/'data/rgb_stacking/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl'
    with source.open('rb') as stream:records=pickle.load(stream)
    assert len(records)==50
    output=target/'data/davis_records';output.mkdir(exist_ok=True,parents=True)
    for name in ['models','scripts']:
        if name=='models':
            if not (target/name).exists():(target/name).symlink_to(root/name)
        else:
            (target/name).mkdir(exist_ok=True)
            evaluator=target/name/'tapvid_official_evaluation_datasets.py'
            if not evaluator.exists():evaluator.symlink_to(root/name/evaluator.name)
    manifest=[]
    for i,record in enumerate(records):
        name=f'rgb_{i:03d}';video=record['video'];points=record['points'];occluded=record['occluded']
        path=output/f'{name}.video.npy'
        if not path.exists():np.save(path,video)
        np.savez_compressed(output/f'{name}.annotations.npz',points=points,occluded=occluded)
        manifest.append({'name':name,'frames':len(video),'points':len(points),'split':'evaluation',
                         'source_size':list(video.shape[1:3]),'dataset':'TAP-Vid-RGB-Stacking'})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    protocol={'source':'https://storage.googleapis.com/dm-tapnet/tapvid_rgb_stacking.zip','videos':50,
              'frames':'all250 per video','points':'all provided tracks','query_mode':'official first-visible',
              'primary_head':[18,5],'primary_extraction':'native_polyphase','primary_readout':'post_full',
              'primary_comparator':'original frame-aligned VAE, raw latent convention, samehead low15',
              'matched_control':'frame-aligned VAE with pretrained latent normalization, mode posterior and source-frame/4 RoPE',
              'secondary_controls':['native interpolation','phase-aware native interpolation','phase-zero anchor override','low15'],
              'tracking_constants':{'argmax_radius':16,'search_radius':48,'visibility_threshold':24,'ema':.05},
              'configuration_selection_on_rgbstack':False,
              'functional_teacher_calibration':'may be evaluated secondarily after its weights are frozen; no RGB data used for its calibration'}
    (target/'protocol.json').write_text(json.dumps(protocol,indent=2))
    print(json.dumps({'prepared':len(manifest),'point_counts':sorted({r['points'] for r in manifest}),'protocol':protocol},indent=2))

if __name__=='__main__':main()
