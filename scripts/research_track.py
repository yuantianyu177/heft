"""Run unchanged HeFT tracking with experimental descriptor readouts.

Loads dense saved tensors via a scoped reader adapter. Uses the official
TAP-Vid metric function, excluding the query frame. The 25-frame pilot is
reported explicitly as a diagnostic, not the full DAVIS benchmark.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, replace
import hashlib
import json
import time
from pathlib import Path
from typing import Mapping
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from safetensors import safe_open

from heft.attn_hook import FeatureKind
from heft.tracking import FeatureSelection, FeatureTracker, FeatureVideo, FrameRange, RopePairing, RopeSpec, TrackingConfig

from research_probe import VARIANTS, norm_matched_indices, readout
from research_rope import frequency_pairs, pair_channels, phase_readout, wan_angles
from research_transport import make_transport_ops


def official_metrics(root):
    source = root / "scripts/tapvid_official_evaluation_datasets.py"
    module = ast.parse(source.read_text())
    function = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "compute_tapvid_metrics")
    # PointOdyssey provides validity separately from occlusion. Keep the
    # official equations, intersecting their evaluation mask only when supplied.
    position=next(i for i,node in enumerate(function.body) if isinstance(node,ast.Assign)
                  and any(isinstance(t,ast.Name) and t.id=='evaluation_points' for t in node.targets))
    function.body[position+1:position+1]=ast.parse('if _heft_valid_mask is not None:\n    evaluation_points = evaluation_points & _heft_valid_mask\n').body
    namespace = {"np": np, "Mapping": Mapping,"_heft_valid_mask":None}
    tree=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))
    exec(compile(tree, str(source), "exec"), namespace)
    def evaluate(*args,valid_mask=None,**kwargs):
        namespace['_heft_valid_mask']=valid_mask
        return namespace[function.name](*args,**kwargs)
    return evaluate, hashlib.sha256(source.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/share/yty/heft"))
    parser.add_argument("--features", default="wan_real_t25")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--names")
    parser.add_argument("--heads", default="18:1,15:2")
    parser.add_argument("--variants", default="post_full,post_low15,content_full,time_removed,space_removed,high15_phase_removed,rope_centered,rope_high15_centered")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ema", type=float, default=.05)
    parser.add_argument("--output-tag")
    parser.add_argument("--chunked", action="store_true")
    parser.add_argument("--boundary-mode", choices=["prepend", "interpolate"], default="prepend")
    parser.add_argument("--gauge-file",type=Path,help="Frozen synthetic per-pair metric amplitudes")
    parser.add_argument('--anchor-features',help='Replace each chunk frame-zero descriptor from this feature folder')
    parser.add_argument('--query-mode',choices=['frame0','first-visible'],default='frame0')
    parser.add_argument('--time-advance',type=float,default=0.,help='Fixed readout timestamp correction measured without motion annotations')
    parser.add_argument('--retime-phase',action='store_true',help='Remove temporal RoPE before resampling, then restore target timestamps')
    args = parser.parse_args()
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)
    evaluate, evaluator_hash = official_metrics(args.root)
    feature_root = args.root / "features" / args.features
    output_root = args.root / "results" / (args.output_tag or f"tracking_{args.features}")
    output_root.mkdir(exist_ok=True, parents=True)
    paths = sorted(feature_root.glob("*.safetensors"))
    if args.chunked:
        paths = [p for p in paths if p.stem.endswith("_c000")]
    if args.names:
        paths = [p for p in paths if (p.stem.rsplit("_c", 1)[0] if args.chunked else p.stem) in args.names.split(",")]
    heads = [tuple(map(int, x.split(":"))) for x in args.heads.split(",")]
    variants = [v for v in VARIANTS if v[0] in args.variants.split(",")]
    for candidate in ("transport_space", "transport_space_low15"):
        if candidate in args.variants.split(","):
            variants.append((candidate, 0, 1, .15 if candidate.endswith("low15") else 0, None))
    gauge_lookup={}
    if args.gauge_file:
        gauge_payload=json.loads(args.gauge_file.read_text())
        for label,field in [('gauge_calibrated','scales'),('gauge_calibrated_pow2','scales_pow2')]:
            if label in args.variants.split(','):
                variants.append((label,1,1,0,None))
                gauge_lookup[label]={tuple(head):torch.tensor(scale,device=device).repeat_interleave(2)
                                     for head,scale in zip(gauge_payload['head_ids'],gauge_payload[field],strict=True)}
    for path in paths[args.shard::args.shards]:
        metadata = json.loads(path.with_suffix(".json").read_text())
        name = metadata["video"]
        chunk_paths = sorted(feature_root.glob(f"{name}_c*.safetensors")) if args.chunked else [path]
        chunk_metadata = [json.loads(p.with_suffix(".json").read_text()) for p in chunk_paths]
        ranges = []
        for i, m in enumerate(chunk_metadata):
            start = m.get("start_frame", 0)
            count = m.get("valid_frames", m["frames"])
            ranges.append(FrameRange(index=i, start=start, stop=start+count, feature_frames=count))
        nt = ranges[-1].stop
        height, width = metadata["input_size"]
        h, w = metadata["feature_size"]
        with np.load(args.root / f"data/davis_records/{name}.annotations.npz") as annotation:
            total_annotated_frames = annotation["points"].shape[1]
            available=~annotation['occluded'][:,:nt]
            if 'valid' in annotation:available&=annotation['valid'][:,:nt]
            selected=available.any(-1) if args.query_mode=='first-visible' else available[:,0]
            gt = annotation["points"][selected, :nt].copy()
            occluded = annotation["occluded"][selected, :nt].copy()
            gt_valid=annotation['valid'][selected,:nt].copy() if 'valid' in annotation else np.ones_like(occluded)
            query_frames=available[selected].argmax(-1) if args.query_mode=='first-visible' else np.zeros(len(gt),dtype=np.int64)
        if args.chunked and nt != total_annotated_frames:
            print(f"SKIP_INCOMPLETE {name}: {nt}/{total_annotated_frames} frames", flush=True)
            continue
        initial_points=gt[np.arange(len(gt)),query_frames]
        query_normalized = torch.tensor(initial_points, dtype=torch.float32)
        queries = torch.cat((query_normalized * torch.tensor([width-1, height-1]), torch.tensor(query_frames[:,None],dtype=torch.float32)), -1)
        video = FeatureVideo(root=path.parent, name=name, model="wan-research",
                             num_frames=nt, frame_size=(height, width), feature_size=(h, w),
                             chunks=tuple(ranges),
                             step=49, layers=tuple(sorted({x[0] for x in heads})),
                             heads=tuple(sorted({x[1] for x in heads})), feature_kinds=("key",),
                             rope=RopeSpec(pairing=RopePairing.ADJACENT, temporal_pairs=22, height_pairs=21, width_pairs=21))
        config = TrackingConfig(query_feature=FeatureKind.KEY, target_feature=FeatureKind.KEY,
                                 update_feature=FeatureKind.KEY, argmax_radius=17, search_radius=42,
                                 visibility_threshold=18, feature_ema_alpha=args.ema)
        for layer, head in heads:
            feature_chunks = []
            for cp, frame_range in zip(chunk_paths, ranges, strict=True):
                with safe_open(cp, framework="pt", device="cpu") as file:
                    feature_chunks.append(file.get_tensor(f"l{layer:02d}_h{head:02d}_dense_key")[:frame_range.feature_frames].to(device)[None])
                if args.anchor_features:
                    with safe_open(args.root/'features'/args.anchor_features/cp.name,framework='pt',device='cpu') as anchor:
                        feature_chunks[-1][:,0]=anchor.get_tensor(f'l{layer:02d}_h{head:02d}_dense_key')[0].to(device)
            all_features = torch.cat(feature_chunks, dim=1)
            norm_indices, _ = norm_matched_indices(all_features)
            del all_features
            for variant in variants:
                output = output_root / f"{name}_l{layer:02d}h{head:02d}_{variant[0]}"
                if output.with_suffix(".json").exists() and not args.overwrite:
                    continue
                started = time.monotonic()
                is_transport = variant[0].startswith("transport_")
                active_config = config
                volumes, canonical_volumes = [], []
                for feature in feature_chunks:
                    angles = wan_angles(feature.shape[1], h, w, device=device)
                    angles[...,:22]*=metadata.get('rope_temporal_scale',1.)
                    if args.time_advance:
                        source=phase_readout(feature,angles,temporal=0,spatial=1) if args.retime_phase else feature.float()
                        positions=(torch.arange(feature.shape[1],device=device,dtype=torch.float32)+args.time_advance).clamp(0,feature.shape[1]-1)
                        left=positions.floor().long();right=positions.ceil().long();weight=(positions-left)[None,:,None,None,None]
                        retimed=source[:,left]*(1-weight)+source[:,right]*weight
                        retimed[:,0]=source[:,0]
                        feature=phase_readout(retimed,angles,temporal=2,spatial=1) if args.retime_phase else retimed
                    if is_transport:
                        canonical = phase_readout(feature, angles, temporal=0, spatial=0)[0]
                        target = phase_readout(feature, angles, temporal=0, spatial=1)[0]
                        if variant[3]:
                            channels = pair_channels(frequency_pairs(variant[3], 1)).to(device)
                            canonical, target = canonical[..., channels], target[..., channels]
                        canonical_volumes.append(canonical.permute(0, 3, 1, 2).contiguous())
                        volumes.append(target.permute(0, 3, 1, 2).contiguous())
                    else:
                        transformed=(feature.float()*gauge_lookup[variant[0]][(layer,head)]) if variant[0] in gauge_lookup else readout(feature, angles, variant, norm_indices)
                        volumes.append(transformed[0].permute(0, 3, 1, 2).contiguous())
                if args.boundary_mode == "prepend":
                    # Supply the actual previous frame as the chunk's handoff anchor.
                    # The original reader instead interpolates T maps to T+1 time steps.
                    original_volumes = list(volumes)
                    original_canonical = list(canonical_volumes)
                    for ci in range(1, len(volumes)):
                        volumes[ci] = torch.cat((original_volumes[ci-1][-1:], original_volumes[ci]), dim=0)
                        if is_transport:
                            canonical_volumes[ci] = torch.cat((original_canonical[ci-1][-1:], original_canonical[ci]), dim=0)
                elif is_transport and len(volumes) > 1:
                    raise ValueError("transport control requires an actual handoff frame, use --boundary-mode prepend")
                transport_state = {}
                if is_transport:
                    transport_ops = make_transport_ops(None, None, feature_size=(h, w), cutoff=variant[3], state=transport_state)
                    active_config = replace(config, query_feature=FeatureKind.QUERY, update_feature=FeatureKind.QUERY)

                class ReaderAdapter:
                    def __init__(self, video, *, device):
                        pass

                    def prefetch_chunks(self, chunks, **kwargs):
                        for chunk in chunks:
                            volume = volumes[chunk.index]
                            maps = {FeatureKind.KEY: volume}
                            if is_transport:
                                canonical_volume = canonical_volumes[chunk.index]
                                maps[FeatureKind.QUERY] = canonical_volume
                                transport_state.update(canonical_volume=canonical_volume, target_volume=volume)
                            yield chunk, maps

                def track_active(active_video,active_queries):
                    tracker=FeatureTracker(config=active_config,device=device)
                    with patch('heft.tracking.tracker.FeatureReader',ReaderAdapter):
                        if is_transport:
                            with patch('heft.tracking.tracker.TrackingOps',transport_ops):
                                return tracker.track(active_video,active_queries,FeatureSelection(layer=layer,head=head))
                        return tracker.track(active_video,active_queries,FeatureSelection(layer=layer,head=head))
                if np.all(query_frames==0):
                    result=track_active(video,queries)
                else:
                    # Group by the provided query timestamp and expose a sliced
                    # feature video beginning there. Preserve subsequent chunk
                    # boundaries and use the original, already-extracted tensors.
                    saved_volumes=list(volumes);saved_canonical=list(canonical_volumes)
                    raw_volumes=[v[1:] if i>0 and args.boundary_mode=='prepend' else v for i,v in enumerate(saved_volumes)]
                    raw_canonical=[v[1:] if i>0 and args.boundary_mode=='prepend' else v for i,v in enumerate(saved_canonical)]
                    tracks=torch.zeros((len(gt),nt,2));visibility=torch.zeros((len(gt),nt),dtype=torch.bool)
                    for query_start in np.unique(query_frames):
                        indices=np.flatnonzero(query_frames==query_start)
                        local_volumes=[];local_canonical=[];local_ranges=[]
                        for ci,frame_range in enumerate(ranges):
                            if frame_range.stop<=query_start:continue
                            begin=max(frame_range.start,int(query_start));offset=begin-frame_range.start
                            count=frame_range.stop-begin
                            local_volumes.append(raw_volumes[ci][offset:offset+count])
                            if is_transport:local_canonical.append(raw_canonical[ci][offset:offset+count])
                            local_ranges.append(FrameRange(index=len(local_ranges),start=begin-int(query_start),
                                                           stop=frame_range.stop-int(query_start),feature_frames=count))
                        for ci in range(len(local_volumes)-1,0,-1):
                            local_volumes[ci]=torch.cat((local_volumes[ci-1][-1:],local_volumes[ci]))
                            if is_transport:local_canonical[ci]=torch.cat((local_canonical[ci-1][-1:],local_canonical[ci]))
                        volumes=local_volumes;canonical_volumes=local_canonical
                        active_video=replace(video,num_frames=nt-int(query_start),chunks=tuple(local_ranges))
                        active_queries=queries[indices].clone();active_queries[:,2]=0
                        partial=track_active(active_video,active_queries)
                        tracks[indices,int(query_start):]=partial.tracks
                        visibility[indices,int(query_start):]=partial.visibility
                    volumes=saved_volumes;canonical_volumes=saved_canonical
                    result=SimpleNamespace(tracks=tracks,visibility=visibility)
                prediction_normalized = result.tracks.numpy() / np.array([width-1, height-1], dtype=np.float32)
                official_queries = np.concatenate((query_frames[:,None].astype(np.float32), initial_points[:,::-1]*256), axis=-1)
                metrics = evaluate(official_queries[None], occluded[None], gt[None]*256,
                                   (~result.visibility.numpy())[None], prediction_normalized[None]*256, "first",
                                   valid_mask=gt_valid[None])
                metrics = {key: float(value[0])*100 for key, value in metrics.items()}
                np.savez_compressed(output.with_suffix(".npz"), tracks=prediction_normalized,
                                    visibility=result.visibility.numpy(), gt=gt, gt_occluded=occluded,gt_valid=gt_valid,
                                    query_points=official_queries)
                payload = {"video": name, "layer": layer, "head": head, "variant": variant,
                           "metrics": metrics, "tracker": asdict(active_config), "frames": nt,
                           "query_policy": f"{args.query_mode}; query frame excluded from metrics",
                           "official_evaluator_sha256": evaluator_hash,
                           "seconds": time.monotonic()-started, "feature_sources": [str(cp) for cp in chunk_paths],
                           "boundary_mode": args.boundary_mode, "chunked": args.chunked}
                payload['invalid_point_frames_excluded']=int((~gt_valid[:,1:]).sum())
                payload['anchor_features_override']=args.anchor_features
                payload['time_advance']=args.time_advance
                payload['retime_phase']=args.retime_phase
                payload["tracking_source_sha256"] = source_hash
                if variant[0] in gauge_lookup:
                    payload['gauge_file']=str(args.gauge_file)
                    payload['gauge_file_sha256']=hashlib.sha256(args.gauge_file.read_bytes()).hexdigest()
                output.with_suffix(".json").write_text(json.dumps(payload, indent=2, default=str))
                print(json.dumps({key: payload[key] for key in ("video", "layer", "head", "variant", "metrics", "seconds")}), flush=True)
                del volumes, canonical_volumes
            del feature_chunks


if __name__ == "__main__":
    main()
