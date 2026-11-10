"""Prepare the released ITTO MOSE component with its three query groups."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    catalog = json.loads((args.source/'parquet.json').read_text())['parquet_files']
    annotated = {p.name for p in (args.source/'mose/annotations').iterdir() if p.is_dir()}
    frames = defaultdict(dict)
    for entry in catalog:
        path = args.source/entry['filename']
        assert path.stat().st_size == entry['size']
        parquet = pq.ParquetFile(path)
        labels = json.loads(parquet.schema_arrow.metadata[b'huggingface'])['info']['features']['label']['names']
        for batch in parquet.iter_batches(columns=['image', 'label'], batch_size=128):
            data = batch.to_pydict()
            for image, label in zip(data['image'], data['label'], strict=True):
                name = labels[label]
                if name in annotated:
                    frame = int(Path(image['path']).stem)
                    assert frame not in frames[name]
                    frames[name][frame] = image['bytes']
        print(path.name, len(frames), sum(map(len, frames.values())), flush=True)
    assert frames.keys() == annotated, (frames.keys() - annotated, annotated - frames.keys())
    manifest = []
    for source_id in sorted(frames):
        ordered = sorted(frames[source_id])
        groups = []
        for group in ('gradient', 'random', 'background'):
            path = args.source/'mose/annotations'/source_id/f'{source_id}_{group}.npy'
            if path.exists():
                a = np.load(path, allow_pickle=False)
                assert a.ndim == 3 and a.shape[1] == len(ordered) and a.shape[2] >= 3
                assert set(np.unique(a[..., 2])) <= {0, 1}
                groups.append(a)
        assert len(groups) >= 2
        tracks = np.concatenate(groups)
        visible = tracks[..., 2] == 1
        eligible = visible.any(1) & (visible.argmax(1) < len(ordered) - 1)
        assert eligible.any(), f'No evaluable visible query in {source_id}'
        first = cv2.imdecode(np.frombuffer(frames[source_id][ordered[0]], np.uint8), cv2.IMREAD_COLOR)
        height, width = first.shape[:2]
        coordinates = tracks[..., :2]
        finite = np.isfinite(coordinates).all(-1)
        assert finite[visible].all()
        if ((coordinates[visible] < -1).any() or (coordinates[visible] > np.array([width+1, height+1])).any()):
            raise ValueError(f'Visible coordinates outside frame {source_id} of size {(height, width)}')
        points = np.nan_to_num(coordinates / np.array([width, height]), nan=-1., posinf=-1., neginf=-1.)
        name = f'itto_{source_id}'
        output = np.lib.format.open_memmap(args.output/f'{name}.video.npy', mode='w+', dtype=np.uint8,
                                          shape=(len(ordered), height, width, 3))
        for index, frame in enumerate(ordered):
            bgr = cv2.imdecode(np.frombuffer(frames[source_id].pop(frame), np.uint8), cv2.IMREAD_COLOR)
            output[index] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        output.flush()
        np.savez_compressed(args.output/f'{name}.annotations.npz', points=points.astype(np.float32), occluded=~visible)
        row = dict(name=name, official_name=source_id, frames=len(ordered), points=len(tracks), eligible=int(eligible.sum()),
                   source_size=[height, width], ordered_frame_numbers=ordered)
        manifest.append(row)
        print(name, len(ordered), len(tracks), flush=True)
    output = args.output/'manifest.json'
    output.write_text(json.dumps(manifest, indent=2))
    print('Manifest SHA256', hashlib.sha256(output.read_bytes()).hexdigest(), flush=True)


if __name__ == '__main__':
    main()
