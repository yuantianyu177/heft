"""Isolated CUDA timing for the matched encoder variants on one full chunk."""
import argparse
import gc
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from research_cvpr_readout import Extractor, PRIMARY_HEADS


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=PRIMARY_HEADS, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--root', type=Path, default=Path('/share/yty/heft'))
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(3)
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    input_clip = args.root/'data/davis_records/drift-straight.video.npy'
    frames = np.load(input_clip, mmap_mode='r')[:25]
    data = {'model': args.model, 'gpu': torch.cuda.get_device_name(device), 'clip': str(input_clip),
            'frames': len(frames), 'repeats': args.repeats, 'rows': {}}
    for mode in ('native1', 'native2', 'native4', 'framewise', 'stride1'):
        extractor = Extractor(args.root, args.model, device, PRIMARY_HEADS[args.model], [(mode, 'head', 0.)], 42)
        extractor.get(frames)
        torch.cuda.synchronize(device)
        times, memory = [], []
        for _ in range(args.repeats):
            torch.cuda.reset_peak_memory_stats(device)
            started = time.monotonic()
            features, profile = extractor.get(frames)
            torch.cuda.synchronize(device)
            times.append(time.monotonic()-started)
            memory.append(max(torch.cuda.max_memory_allocated(device),
                              *(val for key, val in profile.items() if key.endswith('_peak_bytes'))))
            del features
        data['rows'][mode] = {'median_seconds': float(np.median(times)), 'median_peak_GiB': float(np.median(memory))/2**30,
                              'seconds': times, 'peak_bytes': memory}
        print(mode, data['rows'][mode], flush=True)
        del extractor
        gc.collect()
        torch.cuda.empty_cache()
    output = args.root/'cvpr27/summary'/f'isolated_cost_{args.model}.json'
    output.write_text(json.dumps(data, indent=2))


if __name__ == '__main__':
    main()
