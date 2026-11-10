"""Reproduce signed-lag transfer and conditional motion contrasts on frozen tracks."""
from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np

from research_cvpr_motion import THRESHOLDS
from research_cvpr_summarize import paired_interval


ROOT = Path('/share/yty/heft')
REVISION = ROOT / 'cvpr27'
CALIBRATION_VIDEOS = {
    'drift-straight', 'gold-fish', 'horsejump-high', 'india', 'kite-surf', 'lab-coat',
    'paragliding-launch',
}


def read_tracks(paths):
    with ExitStack() as stack:
        rows = [stack.enter_context(np.load(path)) for path in paths]
        for row in rows[1:]:
            for key in ('gt', 'gt_occluded', 'gt_valid', 'query_points'):
                assert np.array_equal(rows[0][key], row[key]), (paths, key)
        gt = rows[0]['gt'] * 256
        visible = ~rows[0]['gt_occluded'] & rows[0]['gt_valid']
        query = rows[0]['query_points'][:, 0].astype(int)
        errors = [row['tracks'][:, 1:-1] * 256 - gt[:, 1:-1] for row in rows]
    return gt, visible, query, errors


def visible_centers(visible, query):
    times = np.arange(1, visible.shape[1] - 1)
    return visible[:, :-2] & visible[:, 1:-1] & visible[:, 2:] & (times[None] > query[:, None])


def lag_statistics(velocity, errors, mask):
    squared_speed = (velocity ** 2).sum(axis=-1)
    moving = mask & (squared_speed > .25)
    assert moving.any()
    v = velocity[moving]
    s2 = squared_speed[moving]
    medians = [float(np.median(-(error[moving] * v).sum(axis=-1) / s2)) for error in errors]
    return dict(medians=medians, qualifying_tracks=int(moving.any(axis=-1).sum()),
                point_frames=int(moving.sum()))


def lag_from_tracks(before, after):
    gt, visible, query, errors = read_tracks([before, after])
    velocity = (gt[:, 2:] - gt[:, :-2]) / 2
    mask = visible_centers(visible, query) & (query[:, None] == 0)
    return dict(**lag_statistics(velocity, errors, mask), frames=gt.shape[1],
                frame_zero_queries=int((query == 0).sum()))


def summarize_lag(rows):
    lags = np.asarray([row['medians'] for row in rows.values()])
    counts = {key: sum(row[key] for row in rows.values())
              for key in ('frames', 'frame_zero_queries', 'qualifying_tracks', 'point_frames')}
    return dict(videos=len(rows), **counts, nominal=paired_interval(lags[:, 0]),
                retimed=paired_interval(lags[:, 1]), reduction=paired_interval(lags[:, 0] - lags[:, 1]),
                rows=rows)


def feature_pairs(before, after, layer_head, count):
    suffix = f'_{layer_head}_post_full.npz'
    matches = sorted(before.glob(f'*{suffix}'))
    assert len(matches) == count, (before, len(matches))
    pairs = {path.name.removesuffix(suffix): (path, after / path.name) for path in matches}
    assert len(pairs) == count
    return pairs


def native_pairs(folder, left, right, count):
    videos = sorted(path.stem for path in folder.glob('*.json'))
    assert len(videos) == count, (folder, len(videos))
    return {name: tuple(folder / f'{name}_native4_head_{offset}.npz' for offset in (left, right))
            for name in videos}


def direct_lag():
    rows = {head: {} for head in ('l15h2', 'l18h1', 'l18h5', 'l20h8')}
    for path in sorted((ROOT / 'results/head_time_probe').glob('*.npz')):
        with np.load(path) as data:
            gt = data['points'].transpose(1, 0, 2) * 256
            visible = np.concatenate([np.ones((len(gt), 1), dtype=bool), data['valid'].T], axis=1)
            query = np.zeros(len(gt), dtype=int)
            velocity = (gt[:, 2:] - gt[:, :-2]) / 2
            mask = visible_centers(visible, query)
            for head, group in rows.items():
                errors = [data[f'{head}_full_{mode}'][:-1].transpose(1, 0, 2) * 256 - gt[:, 1:-1]
                          for mode in ('original', 'calibrated')]
                group[path.stem] = dict(**lag_statistics(velocity, errors, mask),
                                       frames=gt.shape[1], frame_zero_queries=len(gt))
    assert all(len(group) == 30 for group in rows.values())
    return {head: summarize_lag(group) for head, group in rows.items()}


def two_axis_rows(pairs):
    rows = {}
    for name, (before, retimed) in pairs.items():
        trimmed = retimed.with_name(retimed.name.replace('_post_full', '_post_low15'))
        gt, visible, query, errors = read_tracks([before, retimed, trimmed])
        velocity = (gt[:, 2:] - gt[:, :-2]) / 2
        fast = np.linalg.norm(velocity, axis=-1) >= 2
        far = np.linalg.norm(gt[:, 1:-1] - gt[np.arange(len(query)), query][:, None], axis=-1) >= 16
        allowed = visible_centers(visible, query)
        pck = [(np.linalg.norm(error, axis=-1)[..., None] < THRESHOLDS).mean(axis=-1) * 100 for error in errors]
        changes = {'time': pck[1] - pck[0], 'band': pck[2] - pck[1]}
        rows[name] = {}
        for speed, fast_only in [('slow', ~fast), ('fast', fast)]:
            for distance, far_only in [('near', ~far), ('far', far)]:
                use = allowed & fast_only & far_only
                count = int(use.sum())
                rows[name][f'{speed}_{distance}'] = dict(
                    count=count, **({key: float(value[use].mean()) for key, value in changes.items()} if count else {}))
    return rows


def summarize_two_axis(rows):
    contrasts = {}
    for intervention, low, high in [
            ('time', 'slow_near', 'fast_near'), ('time', 'slow_far', 'fast_far'),
            ('band', 'slow_near', 'slow_far'), ('band', 'fast_near', 'fast_far')]:
        paired = [row[high][intervention] - row[low][intervention] for row in rows.values()
                  if min(row[low]['count'], row[high]['count']) >= 20]
        contrasts[f'{intervention}:{high}-{low}'] = dict(videos=len(paired),
                                                        pck=paired_interval(paired) if paired else None)
    return dict(contrasts=contrasts, rows=rows)


def cosmos_head_control():
    labeled = REVISION / 'davis_matched_cosmos'
    synthetic = REVISION / 'davis_cosmos_synthetic_head'
    names = sorted(path.name for path in labeled.glob('*.json'))
    assert len(names) == 30 and set(names) == {path.name for path in synthetic.glob('*.json')}
    variant = 'native4:head:1.25'
    varying = {'head', 'variants', 'source_sha256'}
    rows = {}
    for name in names:
        development, independent = [json.loads((folder / name).read_text()) for folder in (labeled, synthetic)]
        assert all(development[key] == independent[key] for key in ('video', 'frames', 'queries'))
        protocols = [row['protocol'] for row in (development, independent)]
        assert [protocol['head'] for protocol in protocols] == [6, 7]
        assert all(protocol['upsample_features'] is False for protocol in protocols)
        settings = [{key: value for key, value in protocol.items() if key not in varying} for protocol in protocols]
        assert settings[0] == settings[1], name
        rows[development['video']] = [row['results'][variant]['metrics']['average_jaccard']
                                      for row in (development, independent)]
    labeled_aj, synthetic_aj = np.asarray(list(rows.values())).T
    return dict(videos=len(rows), labeled_head=6, synthetic_head=7, advance=1.25,
                labeled_aj=float(labeled_aj.mean()), synthetic_aj=float(synthetic_aj.mean()),
                labeled_minus_synthetic=paired_interval(labeled_aj - synthetic_aj), rows=rows)


def main():
    result = ROOT / 'results'
    wan = feature_pairs(result/'benchmark_davis_polyphase', result/'benchmark_davis_polyphase_corrected', 'l18h05', 30)
    cosmos = feature_pairs(result/'benchmark_cosmos_polyphase', result/'benchmark_cosmos_polyphase_corrected', 'l18h06', 30)
    pairs = {
        'davis_wan_upsampled': wan,
        'davis_cosmos_upsampled': cosmos,
        'davis_wan_grid': native_pairs(REVISION/'davis_matched_wan', 0.0, 1.25, 30),
        'davis_cosmos_grid': native_pairs(REVISION/'davis_matched_cosmos', 0.0, 1.25, 30),
        'davis_cog_grid': native_pairs(REVISION/'davis_cog', 0.0, 0.75, 30),
        'robotap_wan_grid': native_pairs(REVISION/'robotap_split0/wan', 0.0, 1.25, 53),
        'itto_mose_wan_grid': native_pairs(REVISION/'itto_mose/wan', 0.0, 1.25, 44),
    }
    lag = {label: summarize_lag({name: lag_from_tracks(*pair) for name, pair in group.items()})
           for label, group in pairs.items()}
    unseen = {name: pair for name, pair in wan.items() if name not in CALIBRATION_VIDEOS}
    assert len(unseen) == 23
    lag['davis_wan_excluding_probe_sources'] = summarize_lag(
        {name: row for name, row in lag['davis_wan_upsampled']['rows'].items() if name in unseen})
    aj = np.asarray([[json.loads(path.with_suffix('.json').read_text())['metrics']['average_jaccard'] for path in pair]
                     for pair in unseen.values()])
    holdout = dict(videos=len(unseen), excluded=sorted(CALIBRATION_VIDEOS),
                   nominal_aj=float(aj[:, 0].mean()), retimed_aj=float(aj[:, 1].mean()),
                   gain=paired_interval(aj[:, 1] - aj[:, 0]))

    domains = {
        'davis': wan,
        'rgb': feature_pairs(ROOT/'rgb_stacking/results/benchmark_native_polyphase',
                             ROOT/'rgb_stacking/results/latency_transfer_polyphase', 'l18h05', 50),
        'po': feature_pairs(ROOT/'pointodyssey/results/native_transfer_polyphase',
                            ROOT/'pointodyssey/results/latency_transfer_polyphase', 'l18h05', 13),
    }
    data = dict(
        estimator=dict(metric_canvas=256, min_speed=.5, query='frame zero', visibility='valid visible triple',
                       bootstrap=dict(seed=42, draws=10000, unit='video')),
        conditioning=dict(speed_cut=2, distance_cut=16, minimum_point_frames_per_cell=20),
        lag=lag, holdout=holdout, direct_lag=direct_lag(), cosmos_head_control=cosmos_head_control(),
        conditional={label: summarize_two_axis(two_axis_rows(group)) for label, group in domains.items()})
    output = REVISION / 'polish/audit/robustness.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'holdout': holdout, 'lag': {key: row['reduction'] for key, row in lag.items()},
                      'conditional': {key: row['contrasts'] for key, row in data['conditional'].items()}}, indent=2))


if __name__ == '__main__':
    main()
