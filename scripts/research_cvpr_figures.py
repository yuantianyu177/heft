"""Build vector paper figures from the frozen HeFT measurement and video files."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import matplotlib.patheffects as effects
import numpy as np
from safetensors import safe_open
import torch
from torch.nn import functional as F


WIDTH = 6.875
INK = '#172e43'
MUTED = '#526474'
GRID = '#d8e2e8'
PALE = '#f3f7fa'
TEAL = '#007a6f'
ORANGE = '#b44823'
VIOLET = '#6c4d8e'
BLUE = '#275c92'
TEAL_PHOTO = '#34e3c4'
ORANGE_PHOTO = '#f88754'
VIOLET_PHOTO = '#c1a1fa'
TEASER_EXAMPLE = ('parkour', 16, 17)
MATCH_EXAMPLE = ('horsejump-high', 1, 21)


def style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 8.0,
        'axes.labelsize': 8.0,
        'axes.labelcolor': INK,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': MUTED,
        'axes.linewidth': .6,
        'xtick.labelsize': 7.4,
        'ytick.labelsize': 7.4,
        'xtick.color': INK,
        'ytick.color': INK,
        'xtick.major.size': 2.4,
        'ytick.major.size': 2.4,
        'xtick.major.pad': 2.2,
        'ytick.major.pad': 2.2,
        'text.color': INK,
        'mathtext.fontset': 'dejavusans',
        'pdf.fonttype': 42,
        'figure.facecolor': 'white',
        'savefig.dpi': 300,
    })
    torch.set_num_threads(3)


def read_json(path):
    return json.loads(path.read_text())


def axis(fig, x, y, width, height):
    w, h = fig.get_size_inches()
    return fig.add_axes((x / w, y / h, width / w, height / h))


def label(fig, x, y, text, **kwargs):
    w, h = fig.get_size_inches()
    return fig.text(x / w, y / h, text, **kwargs)


def panel(fig, x, y, letter, title):
    label(fig, x, y, letter, va='center', fontsize=8, fontweight='bold', color='white',
          bbox={'boxstyle': 'round,pad=.22,rounding_size=.13', 'facecolor': INK, 'edgecolor': 'none'})
    label(fig, x + .19, y, title, va='center', fontsize=9.4, fontweight='bold')


def divider(fig, x, top, bottom):
    w, h = fig.get_size_inches()
    fig.add_artist(Line2D([x / w] * 2, [bottom / h, top / h], color=GRID,
                          lw=.65, transform=fig.transFigure))


def save(fig, out, name):
    fig.savefig(out / f'{name}.pdf', metadata={'Creator': 'HeFT figure renderer', 'CreationDate': None})
    plt.close(fig)


def photo(ax, frame, center=None, fraction=1.0):
    height, width = frame.shape[:2]
    _, _, w, h = ax.get_position().bounds
    fig_w, fig_h = ax.figure.get_size_inches()
    ratio = w * fig_w / (h * fig_h)
    span_x = min(width * fraction, height * ratio)
    span_y = span_x / ratio
    origin = np.asarray(center if center is not None else [.5, .5]) * [width, height]
    left, top = np.clip(origin - [.5 * span_x, .5 * span_y], [0, 0], [width - span_x, height - span_y])
    ax.imshow(frame, extent=(0, width, height, 0), interpolation='lanczos')
    ax.set(xlim=(left, left + span_x), ylim=(top + span_y, top), aspect='equal')
    ax.set_axis_off()


def mark(ax, point, size, color, symbol, diameter=32):
    x, y = np.asarray(point) * size
    if symbol == 'o':
        ax.scatter(x, y, s=diameter * 1.38, facecolor='none', edgecolor=INK, lw=1.5, zorder=8)
        ax.scatter(x, y, s=diameter * 1.38, facecolor='none', edgecolor=color, lw=1.02, zorder=9)
    else:
        ax.scatter(x, y, s=diameter, marker=symbol, facecolor=color, edgecolor=INK, lw=.75, zorder=10)


def photo_tag(ax, text):
    ax.text(.055, .88, text, transform=ax.transAxes, color='white', fontsize=7.4, fontweight='bold',
            va='center', bbox={'facecolor': INK, 'edgecolor': 'none', 'alpha': .91, 'pad': 2.1})


def legend(fig, x, y, handles, cols):
    w, h = fig.get_size_inches()
    return fig.legend(handles=handles, loc='center', bbox_to_anchor=(x / w, y / h),
                      ncols=cols, frameon=False, fontsize=7.3, handlelength=1.4,
                      handletextpad=.37, columnspacing=.92, borderaxespad=0)


def marker_handle(color, symbol, text, filled=True):
    return Line2D([], [], ls='', marker=symbol, ms=5, mec=color, mfc=color if filled else 'none',
                  label=text, mew=1)


def estimate(ax, x, y, ci, color, marker='o', horizontal=False):
    mean, lo, hi = ci
    errors = [[mean - lo], [hi - mean]]
    ax.errorbar(x, y, xerr=errors if horizontal else None, yerr=None if horizontal else errors,
                fmt=marker, color=color, mfc=color, mec='white', mew=.45, ms=5.0,
                elinewidth=1.05, capsize=2.4, zorder=4)


def teaser(root, out):
    summary = read_json(root / 'results/temporal_calibration_summary.json')
    fig = plt.figure(figsize=(WIDTH, 2.03))
    panel(fig, .055, 1.91, 'a', 'One point, two readouts')
    panel(fig, 3.08, 1.91, 'b', 'Translation probe')
    panel(fig, 5.07, 1.91, 'c', 'Lag on real tracks')
    divider(fig, 2.94, 1.76, .36)
    divider(fig, 4.94, 1.76, .36)

    name, point, t = TEASER_EXAMPLE
    video = np.load(root / 'data/davis_records' / f'{name}.video.npy', mmap_mode='r')
    size = np.array([video.shape[2], video.shape[1]])
    suffix = f'{name}_l18h05_post_full.npz'
    with np.load(root / 'results/benchmark_davis_polyphase' / suffix) as raw, np.load(
            root / 'results/benchmark_davis_polyphase_corrected' / suffix) as retimed:
        ground = raw['gt'][point]
        q = int(raw['query_points'][point, 0])
        query = axis(fig, .055, .48, .87, 1.14)
        target = axis(fig, .98, .48, 1.80, 1.14)
        photo(query, video[q], ground[q], .19)
        photo(target, video[t], ground[t], .31)
        photo_tag(query, f'Query · {q}')
        photo_tag(target, f'Target · {t}')
        mark(query, ground[q], size, 'white', 'o')
        path = ground[t - 2:t + 2] * size
        target.plot(*path.T, lw=1.15, ls=(0, (2, 2)), color='white', zorder=6,
                    path_effects=[effects.Stroke(linewidth=2.5, foreground=INK), effects.Normal()])
        mark(target, raw['tracks'][point, t], size, ORANGE_PHOTO, 's', 24)
        mark(target, ground[t], size, 'white', 'o', 43)
        mark(target, retimed['tracks'][point, t], size, TEAL_PHOTO, 'D', 20)

    legend(fig, 1.42, .25, [marker_handle(INK, 'o', 'Ground truth', False),
                           marker_handle(ORANGE, 's', 'Nominal'), marker_handle(TEAL, 'D', 'Retimed')], 3)
    label(fig, .055, .075, 'TAP-Vid DAVIS · parkour', color=MUTED, fontsize=7.4, va='center')

    clock = summary['codec_clock']
    b = axis(fig, 3.46, .51, 1.29, 1.09)
    b.plot(np.arange(0, 25, 4), [0, *clock['latent_summary']['mean_delay_by_time']],
           's-', color=ORANGE, ms=3.5, lw=1.5, label='Native latent')
    b.plot(np.arange(1, 25), clock['decoded_rgb_summary']['mean_delay_by_time'],
           color=TEAL, lw=1.55, label='Decoded RGB')
    b.set(xlim=(-.7, 24.9), ylim=(-.12, 1.8), xticks=[0, 8, 16, 24], yticks=[0, .5, 1, 1.5],
          xlabel='Input frame', ylabel='Lag (frames)')
    b.yaxis.labelpad = 1
    b.grid(axis='y', color=GRID, lw=.55)
    legend(fig, 4.01, .13, b.get_lines(), 2)

    methods = summary['query_time_subsets']['methods']
    c = axis(fig, 5.16, .57, 1.58, 1.03)
    c.axvline(1.25, color=MUTED, lw=.7, ls=(0, (2, 2)), zorder=0)
    c.axvline(0, color=GRID, lw=.8, zorder=0)
    for level, key, color, symbol, name in [(1.02, 'native', ORANGE, 's', 'Nominal'),
                                           (.31, 'calibrated', TEAL, 'D', 'Retimed')]:
        ci = methods[key]['query_frame0']['video_median_lag_mean_ci95']
        estimate(c, ci[0], level, ci, color, symbol, horizontal=True)
        c.text(ci[0], level + .20, f'{name} {ci[0]:.2f}', ha='left' if key == 'calibrated' else 'right',
               color=color, fontweight='bold', fontsize=7.6)
    c.spines['left'].set_visible(False)
    c.set(xlim=(-.045, 1.53), ylim=(-.10, 1.55), yticks=[], xticks=[0, .5, 1, 1.5], xlabel='Signed lag (frames)')
    c.text(1.25, .035, 'VAE probe', ha='right', va='bottom', color=MUTED, fontsize=6.8)
    label(fig, 5.15, .13, 'DAVIS AJ   32.84 → 39.96', fontweight='bold', color=TEAL, fontsize=8.0)
    save(fig, out, 'heft_content_time_teaser')


def draw_arrow(ax, start, stop, color=MUTED, connectionstyle='arc3'):
    ax.add_patch(FancyArrowPatch(start, stop, arrowstyle='-|>', color=color, lw=.85, mutation_scale=8.5,
                                 shrinkA=0, shrinkB=0, connectionstyle=connectionstyle))


def rounded_box(ax, xy, width, height, color, radius=.045, edge='none'):
    patch = FancyBboxPatch(xy, width, height, boxstyle=f'round,pad=0,rounding_size={radius}',
                           facecolor=color, edgecolor=edge, linewidth=.7)
    ax.add_patch(patch)
    return patch


def pipeline(root, out):
    fig = plt.figure(figsize=(WIDTH, 2.29))
    panel(fig, .055, 2.16, 'a', 'Sample native phases')
    panel(fig, 3.45, 2.16, 'b', 'Select channels')
    panel(fig, 5.17, 2.16, 'c', 'Correct time')
    divider(fig, 3.32, 2.02, .08)
    divider(fig, 5.02, 2.02, .08)
    canvas = axis(fig, 0, 0, WIDTH, 2.29)
    canvas.set(xlim=(0, WIDTH), ylim=(0, 2.29))
    canvas.axis('off')

    # The first nine frames show the four phase residues twice.
    name, point, _ = TEASER_EXAMPLE
    video = np.load(root / 'data/davis_records' / f'{name}.video.npy', mmap_mode='r')
    with np.load(root / 'results/benchmark_davis_polyphase' / f'{name}_l18h05_post_full.npz') as track:
        center = track['gt'][point, :9].mean(0)
    times = .57 + np.arange(9) * .297
    for t, x in enumerate(times):
        photo(axis(fig, x - .13, 1.64, .26, .26), video[t], center, .29)
        canvas.text(x, 1.53, str(t), ha='center', va='center', fontsize=7.3, color=MUTED)
    canvas.text(.055, 1.53, 'Frame', va='center', fontsize=7.6, fontweight='bold')
    colors = [BLUE, TEAL, VIOLET, ORANGE]
    for phase, color in enumerate(colors):
        y = 1.31 - .235 * phase
        canvas.text(.065, y, rf'$r={phase}$', va='center', fontsize=8.1)
        canvas.plot([times[0], times[-1]], [y, y], lw=.65, color=GRID)
        for t, x in enumerate(times):
            if t == 0 or t % 4 == phase:
                rounded_box(canvas, (x - .087, y - .074), .174, .148, INK if t == 0 else color)
            else:
                canvas.scatter(x, y, s=6.5, color=GRID)
    draw_arrow(canvas, (1.74, .50), (1.74, .34))
    for t, x in enumerate(times):
        rounded_box(canvas, (x - .117, .115), .234, .153, INK if t == 0 else colors[t % 4])
        canvas.text(x, .191, rf'$K_{t}$', color='white', ha='center', va='center', fontsize=6.9)
    canvas.text(.055, .187, 'Keys', va='center', fontsize=7.7, fontweight='bold')

    for i in range(4):
        for pair in range(8):
            x = 3.52 + .166 * pair
            y = 1.59 - .247 * i
            color = GRID if i != 1 else ('#f0bbab' if pair < 2 else TEAL)
            rounded_box(canvas, (x, y), .132, .152, color, radius=.016)
    canvas.text(3.53, 1.93, 'Heads in one layer', color=MUTED, fontsize=7.8)
    canvas.add_patch(Rectangle((3.485, 1.309), 1.369, .215, facecolor='none', edgecolor=INK, lw=.85))
    canvas.text(3.53, .72, 'High ω', color=ORANGE, fontsize=7.7, fontweight='bold')
    canvas.text(4.81, .72, 'Low ω', ha='right', color=TEAL, fontsize=7.7, fontweight='bold')
    draw_arrow(canvas, (3.58, .60), (4.76, .60))
    canvas.text(3.52, .32, 'One head', fontsize=8.0, fontweight='bold')
    canvas.text(3.52, .14, 'Full or trimmed band', fontsize=7.7, color=MUTED)

    for x, title, weight in [(5.26, r'$K_{t+1}$', '0.75'), (6.13, r'$K_{t+2}$', '0.25')]:
        rounded_box(canvas, (x, 1.42), .64, .39, PALE, edge=GRID)
        canvas.text(x + .32, 1.62, title, ha='center', va='center', fontsize=10.1, fontweight='bold')
        draw_arrow(canvas, (x + .32, 1.39), (6.02, .94), color=TEAL)
        canvas.text(x + .32, 1.23, weight, color=TEAL, ha='center', fontsize=8.0,
                    bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': .7})
    rounded_box(canvas, (5.54, .51), .96, .39, TEAL)
    canvas.text(6.02, .71, r'$D_t$', color='white', ha='center', va='center', fontsize=10.6)
    canvas.text(6.02, .31, 'Query · target · update', ha='center', fontsize=7.8, fontweight='bold')
    canvas.text(6.02, .12, 'Wan advance: 1.25 frames', ha='center', fontsize=7.4, color=MUTED)
    save(fig, out, 'heft_calibrated_pipeline')


def decoder_intervention(root, out):
    base = root / 'followup_temporal_response'
    data = read_json(base / 'codec_subspace/artifacts/codec_subspace_wan_bike-packing.json')
    row = next(r for r in data['records'] if r['curve'] == 'null_sine5_x' and r['amplitude'] == 6)
    summary = read_json(base / 'followup_summary.json')['codec']['null/6']
    entries = [('full', BLUE, '--', 'Full change'), ('position_component', ORANGE, '-.', 'Position projection'),
               ('residual_component', TEAL, ':', 'Residual')]
    fig = plt.figure(figsize=(WIDTH, 2.03))
    panel(fig, .055, 1.91, 'a', 'Latent position stays near zero')
    panel(fig, 4.01, 1.91, 'b', 'Motion recovered by the decoder')
    divider(fig, 3.86, 1.76, .38)
    a = axis(fig, .49, .55, 3.21, 1.05)
    b = axis(fig, 5.19, .55, 1.52, 1.05)
    a.plot(np.asarray(row['path'])[:, 0], color=INK, lw=1.65, label='Input motion')
    for key, color, ls, text in entries:
        a.plot(np.asarray(row[key]['registered_displacement'])[:, 0], color=color, ls=ls, lw=1.3, label=text)
    a.scatter(data['nominal_times'], np.asarray(row['latent_projected_displacement'])[:, 0],
              s=16, color=VIOLET, zorder=5, label='Latent position')
    a.grid(axis='y', color=GRID, lw=.55)
    a.set(xlim=(0, 24), xticks=np.arange(0, 25, 4), ylim=(-6.8, 6.8), yticks=[-6, 0, 6],
          xlabel='Input frame', ylabel='Displacement (px)')
    handles, _ = a.get_legend_handles_labels()
    legend(fig, WIDTH / 2, .105, handles, 5)

    for level, (key, color, _, text) in zip((2, 1, 0), entries, strict=True):
        ci = summary[f'registered/{key}/mae_pixels']
        b.hlines(level, 0, ci[0], color=color, alpha=.38, lw=1.7)
        estimate(b, ci[0], level, ci, color, horizontal=True)
        b.text(ci[0] + .14, level + .19, f'{ci[0]:.2f}', va='center', color=color, fontsize=8.0, fontweight='bold')
        b.text(-.16, level, text, ha='right', va='center', fontsize=7.8)
    b.grid(axis='x', color=GRID, lw=.55)
    b.spines['left'].set_visible(False)
    b.set(xlim=(-.04, 3.45), xticks=[0, 1, 2, 3], ylim=(-.43, 2.44), yticks=[],
          xlabel='Motion error (px)')
    save(fig, out, 'heft_decoder_intervention')


def similarity_map(file, key, q, t, advance):
    volume = file.get_slice(key)
    query_map = volume[0:1].float().permute(0, 3, 1, 2)
    q = torch.as_tensor(q, dtype=torch.float32).reshape(1, 1, 1, 2) * 2 - 1
    source = F.grid_sample(query_map, q, align_corners=False, padding_mode='border')[0, :, 0, 0]
    a, b = int(np.floor(t + advance)), int(np.ceil(t + advance))
    weight = t + advance - a
    key_map = (volume[a:a+1].float() * (1 - weight) + volume[b:b+1].float() * weight).permute(0, 3, 1, 2)
    target = F.interpolate(key_map, size=(60, 104), mode='bilinear', align_corners=False)[0]
    return torch.einsum('c,chw->hw', F.normalize(source, dim=0), F.normalize(target, dim=0)).numpy()


def match_strip(fig, root):
    name, index, t = MATCH_EXAMPLE
    video = np.load(root / 'data/davis_records' / f'{name}.video.npy', mmap_mode='r')
    size = np.array([video.shape[2], video.shape[1]])
    with np.load(root / 'results/head_time_probe' / f'{name}.npz') as data:
        q, ground = data['points'][[0, t], index]
        predictions = [data[key][t - 1, index] for key in (
            'l18h5_full_original', 'l18h5_full_calibrated', 'l20h8_full_calibrated')]
    path = root / 'features/wan_native_polyphase_chunk25' / f'{name}_c000.safetensors'
    with safe_open(path, framework='pt', device='cpu') as file:
        maps = [similarity_map(file, f'l{layer:02d}_h{head:02d}_dense_key', q, t, offset)
                for layer, head, offset in [(18, 5, 0), (18, 5, 1.25), (20, 8, 1.25)]]
    for score, pred in zip(maps, predictions, strict=True):
        y, x = np.unravel_index(np.argmax(score), score.shape)
        assert np.max(np.abs(np.array([(x + .5) / 104, (y + .5) / 60]) - pred)) < .011
    vmax = max(x.max() for x in maps)
    vmin = min(np.quantile(x, .45) for x in maps)
    names = [f'Query · frame 0', f'Target · frame {t}', '18/5 · nominal', '18/5 · retimed', '20/8 · retimed']
    for i, title in enumerate(names):
        x = .055 + 1.366 * i
        ax = axis(fig, x, 1.71, 1.303, .682)
        if i == 0:
            photo(ax, video[0], center=(q + ground) / 2)
            mark(ax, q, size, 'white', 'o', 25)
        else:
            photo(ax, video[t], center=(q + ground) / 2)
            if i > 1:
                sim = maps[i-2]
                limits = ax.get_xlim(), ax.get_ylim()
                alpha = .20 + .56 * np.clip((sim - vmin) / (vmax - vmin), 0, 1)
                rgba = plt.get_cmap('magma')(np.clip((sim - vmin) / (vmax - vmin), 0, 1))
                rgba[..., 3] = alpha
                ax.imshow(rgba, extent=(0, size[0], size[1], 0), interpolation='bilinear')
                ax.set(xlim=limits[0], ylim=limits[1])
                color, marker = [(ORANGE_PHOTO, 's'), (TEAL_PHOTO, 'D'), (VIOLET_PHOTO, '^')][i - 2]
                mark(ax, predictions[i - 2], size, color, marker, 20)
            mark(ax, ground, size, 'white', 'o', 30)
        label(fig, x + .6515, 1.585, title, ha='center', va='center', fontsize=7.7,
              fontweight='bold', color=[INK, INK, ORANGE, TEAL, VIOLET][i])


def diagnostics(root, out):
    heads = read_json(root / 'results/head_time_probe_summary.json')['heads']
    scores = read_json(root / 'results/query_clock_summary.json')['methods']
    fig = plt.figure(figsize=(WIDTH, 2.93))
    panel(fig, .055, 2.80, 'a', 'Which head matches the point?')
    match_strip(fig, root)
    legend(fig, 5.48, 2.80, [marker_handle(INK, 'o', 'True point', False),
                            marker_handle(MUTED, 'D', 'Match peak')], 2)
    label(fig, .055, 2.54, 'Cosine maps on the same target frame', va='center', color=MUTED, fontsize=7.6)
    scale = axis(fig, 5.37, 2.50, .94, .084)
    scale.imshow(np.linspace(0, 1, 128)[None, :], cmap='magma', aspect='auto')
    scale.set_axis_off()
    label(fig, 5.27, 2.542, 'Low', ha='right', va='center', color=MUTED, fontsize=7.0)
    label(fig, 6.40, 2.542, 'High', va='center', color=MUTED, fontsize=7.0)

    panel(fig, .055, 1.36, 'b', 'Gain from retiming each head')
    panel(fig, 3.72, 1.36, 'c', 'Where the correction is applied')
    divider(fig, 3.49, 1.24, .35)
    a = axis(fig, .49, .37, 2.71, .77)
    b = axis(fig, 4.09, .36, 2.60, .80)
    for level, key in zip(range(3, -1, -1), ('l15h2_full', 'l18h1_full', 'l18h5_full', 'l20h8_full'), strict=True):
        ci = heads[key]['gain_video_ci95']
        estimate(a, ci[0], level, ci, TEAL if level else VIOLET, 'D' if level else '^', horizontal=True)
        a.text(ci[2] + .31, level, f'{ci[0]:+.2f}', va='center', color=TEAL if level else VIOLET,
               fontweight='bold', fontsize=7.4)
    a.axvline(0, color=MUTED, lw=.7)
    a.set(yticks=range(4), yticklabels=['20/8', '18/5', '18/1', '15/2'], ylim=(-.45, 3.48), xlim=(-.7, 15),
          xticks=[0, 4, 8, 12], xlabel='Change in direct-match PCK (points)')
    a.tick_params(axis='y', length=0, pad=3.1)
    a.spines['left'].set_visible(False)
    a.grid(axis='x', color=GRID, lw=.55)

    bins = ('query_frame0', 'query_later')
    variants = [('native', MUTED, 'o', 'Nominal'), ('uniform_posthoc', ORANGE, 's', 'Shift tracks'),
                ('calibrated', TEAL, 'D', 'Retime keys')]
    for key, color, marker, text in variants:
        y = [scores[key][bucket]['AJ'] for bucket in bins]
        b.plot([0, 1], y, color=color, marker=marker, ms=4.4, lw=1.45, mec='white', mew=.4, label=text)
        if key == 'calibrated':
            b.text(1.08, y[-1], f'{y[-1]:.1f}', color=color, va='center', fontsize=7.4)
    b.set(xlim=(-.14, 1.31), ylim=(22.5, 42.5), xticks=[0, 1], xticklabels=['Frame-zero query', 'Later query'],
          yticks=[25, 30, 35, 40], ylabel='AJ')
    b.grid(axis='y', color=GRID, lw=.55)
    legend(fig, 5.39, .075, b.get_lines(), 3)
    save(fig, out, 'heft_time_diagnostics')


def motion_strata(root, out):
    data = read_json(root / 'cvpr27/summary/motion_strata.json')
    fig = plt.figure(figsize=(WIDTH, 2.18))
    panel(fig, .055, 2.04, 'a', 'When · time calibration')
    panel(fig, 3.66, 2.04, 'b', 'What · high-frequency trimming')
    divider(fig, 3.48, 1.87, .38)
    entries = [('davis', 'DAVIS', BLUE, 'o'), ('rgb', 'RGB-Stacking', TEAL, 's'),
               ('po', 'PointOdyssey', VIOLET, '^')]
    for x, key, dim, bins, xlabel in [
            (.51, 'time', 'speed', ['0–0.5', '0.5–2', '2–4', '≥4'], 'Instantaneous speed (metric px/frame)'),
            (4.14, 'band', 'displacement', ['0–4', '4–16', '16–32', '≥32'], 'Distance from query (metric px)')]:
        ax = axis(fig, x, .55, 2.58, 1.19)
        ax.axhline(0, color=MUTED, lw=.76)
        for i, (dataset, _, color, marker) in enumerate(entries):
            stats = np.asarray([row['pck'] for row in data[f'{dataset}_{key}'][dim]['stats']])
            positions = np.arange(4) + (i - 1) * .14
            ax.plot(positions, stats[:, 0], lw=.8, color=color, alpha=.66)
            for xx, ci in zip(positions, stats, strict=True):
                estimate(ax, xx, ci[0], ci, color, marker)
        ax.set(xlim=(-.38, 3.39), xticks=range(4), xticklabels=bins, xlabel=xlabel, ylabel='Change in PCK (points)')
        ax.grid(axis='y', color=GRID, lw=.55)
    legend(fig, WIDTH / 2, .092, [marker_handle(color, marker, name) for _, name, color, marker in entries], 3)
    save(fig, out, 'heft_motion_strata')


def lag_transfer(root, out):
    results = read_json(root / 'cvpr27/polish/audit/robustness.json')['lag']
    rows = [('davis_wan_upsampled', 'DAVIS · Wan'), ('davis_cosmos_upsampled', 'DAVIS · Cosmos'),
            ('davis_cog_grid', 'DAVIS · CogVideoX'), ('robotap_wan_grid', 'RoboTAP · Wan'),
            ('itto_mose_wan_grid', 'ITTO-MOSE · Wan')]
    fig = plt.figure(figsize=(3.3125, 1.70))
    ax = axis(fig, 1.12, .36, 2.00, 1.08)
    ax.axvline(0, color=MUTED, ls=(0, (2, 2)), lw=.7, zorder=0)
    for y, (key, _) in enumerate(reversed(rows)):
        nominal, retimed = (results[key][mode] for mode in ('nominal', 'retimed'))
        ax.hlines(y, retimed[0], nominal[0], color=GRID, lw=2.1, zorder=1)
        estimate(ax, nominal[0], y, nominal, ORANGE, 's', horizontal=True)
        estimate(ax, retimed[0], y, retimed, TEAL, 'D', horizontal=True)
    ax.set(yticks=range(len(rows)), yticklabels=[name for _, name in reversed(rows)], ylim=(-.55, 4.55),
           xlim=(-.43, 1.92), xticks=[0, .5, 1, 1.5], xlabel='Signed lag (frames)')
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0, pad=3.1)
    legend(fig, 2.08, 1.59, [marker_handle(ORANGE, 's', 'Nominal'),
                             marker_handle(TEAL, 'D', 'Retimed')], 2)
    save(fig, out, 'heft_lag_transfer')


def failure_cases(root, out):
    fig = plt.figure(figsize=(WIDTH, 2.55))
    panel(fig, .055, 2.40, 'a', 'An occlusion leaves a large residual error')
    legend(fig, 5.34, 2.40, [marker_handle(INK, 'o', 'Ground truth', False),
                            marker_handle(ORANGE, 's', 'Nominal'), marker_handle(TEAL, 'D', 'Retimed')], 3)
    name, index, t = 'itto_93e22d48', 0, 82
    base = root / 'cvpr27/itto_mose'
    video = np.load(base / 'data' / f'{name}.video.npy', mmap_mode='r')
    size = np.array([video.shape[2], video.shape[1]])
    with np.load(base / 'wan' / f'{name}_native4_head_0.0.npz') as nominal, np.load(
            base / 'wan' / f'{name}_native4_head_1.25.npz') as retimed:
        gt = nominal['gt'][index]
        q = int(nominal['query_points'][index, 0])
        center = gt[[q, t, t + 4]].mean(0)
        for i, (frame, label_text) in enumerate(((q, 'Query'), (t - 2, 'Occluded'), (t, 'Reappears'), (t + 4, 'Later'))):
            ax = axis(fig, .055 + i * 1.713, 1.58, 1.624, .621)
            photo(ax, video[frame], center, .48)
            photo_tag(ax, f'{label_text} · {frame}')
            if frame > q:
                if nominal['visibility'][index, frame]:
                    mark(ax, nominal['tracks'][index, frame], size, ORANGE_PHOTO, 's', 29)
                if retimed['visibility'][index, frame]:
                    mark(ax, retimed['tracks'][index, frame], size, TEAL_PHOTO, 'D', 12)
            if not nominal['gt_occluded'][index, frame]:
                mark(ax, gt[frame], size, 'white', 'o', 24)
        errors = [np.linalg.norm((run['tracks'][index, t] - gt[t]) * 256) for run in (nominal, retimed)]
    label(fig, .055, 1.43, f'ITTO-MOSE · error at return {errors[0]:.1f} px nominal / {errors[1]:.1f} px retimed',
          va='center', fontsize=7.7, color=MUTED)

    panel(fig, .055, 1.16, 'b', 'A short displacement can benefit from the full band')
    name, index, t = 'rgb_026', 13, 22
    base = root / 'rgb_stacking'
    video = np.load(base / 'data/davis_records' / f'{name}.video.npy', mmap_mode='r')
    size = np.array([video.shape[2], video.shape[1]])
    folder = base / 'results/latency_transfer_polyphase'
    with np.load(folder / f'{name}_l18h05_post_full.npz') as full, np.load(
            folder / f'{name}_l18h05_post_low15.npz') as trim:
        gt = full['gt'][index]
        q = int(full['query_points'][index, 0])
        center = (gt[q] + gt[t]) / 2
        for i, (frame, run, tag, color, marker) in enumerate([
                (q, None, 'Query', INK, 'o'), (t, full, 'Full band', TEAL_PHOTO, 'D'),
                (t, trim, 'Trimmed band', VIOLET_PHOTO, '^')]):
            ax = axis(fig, .055 + i * 1.713, .238, 1.624, .66)
            photo(ax, video[frame], center, .36)
            photo_tag(ax, f'{tag} · {frame}')
            if run is not None:
                mark(ax, run['tracks'][index, frame], size, color, marker, 22)
            mark(ax, gt[frame], size, 'white', 'o', 34)
        distance = np.linalg.norm((gt[t] - gt[q]) * 256)
        errors = [np.linalg.norm((run['tracks'][index, t] - gt[t]) * 256) for run in (full, trim)]
    label(fig, 5.22, .783, 'RGB-Stacking', va='center', fontsize=8.2, fontweight='bold')
    label(fig, 5.22, .612, f'{distance:.2f} px displacement', va='center', fontsize=7.7, color=MUTED)
    label(fig, 5.22, .459, f'{errors[0]:.1f} px full-band error', va='center', fontsize=7.7, color=TEAL)
    label(fig, 5.22, .306, f'{errors[1]:.1f} px trimmed-band error', va='center', fontsize=7.7, color=VIOLET)
    label(fig, .055, .090, 'Both RGB-Stacking readouts use the same retimed keys.', va='center', fontsize=7.6, color=MUTED)
    save(fig, out, 'heft_failure_cases')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/share/yty/heft'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    style()
    teaser(args.root, args.out)
    decoder_intervention(args.root, args.out)
    pipeline(args.root, args.out)
    diagnostics(args.root, args.out)
    motion_strata(args.root, args.out)
    lag_transfer(args.root, args.out)
    failure_cases(args.root, args.out)


if __name__ == '__main__':
    main()
