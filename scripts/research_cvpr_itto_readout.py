"""Project subpixel ITTO queries at the border onto the image for conditioning."""
import json

import numpy as np

import research_cvpr_readout as readout


def main():
    run_tracker = readout.run_tracker

    def track(volumes, positions, query_frames, extractor, ema, upsample):
        clipped = np.clip(positions, 0, 1)
        outside = np.any(positions != clipped, axis=1)
        if outside.any():
            if np.max(np.abs(positions[outside] - clipped[outside])) > 0.001:
                raise ValueError('ITTO query more than a subpixel distance from the image')
            print(json.dumps({'border_query_frames': query_frames[outside].tolist(),
                              'annotated_positions': positions[outside].tolist(),
                              'conditioning_positions': clipped[outside].tolist()}), flush=True)
        return run_tracker(volumes, clipped, query_frames, extractor, ema, upsample)

    readout.run_tracker = track
    readout.main()


if __name__ == '__main__':
    main()
