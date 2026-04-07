import json
from collections import Counter, defaultdict
from pathlib import Path


SVG_WIDTH = 1500
SVG_HEIGHT = 1120
SLICE_SVG_WIDTH = 1500
SLICE_SVG_HEIGHT = 1120
HEATMAP_SVG_WIDTH = 1500
HEATMAP_SVG_HEIGHT = 940
ONE_PARAM_SVG_WIDTH = 1500
ONE_PARAM_SVG_HEIGHT = 1180

PARAM_KEYS = [
    'model_cfg.recon.mask_ratio',
    'model_cfg.recon.lambda',
    'model_cfg.recon.lambda_edge',
    'model_cfg.recon.edge_sobel_thr_ratio',
    'model_cfg.recon.edge_sobel_thr_abs',
]

SHORT_NAMES = {
    'model_cfg.recon.mask_ratio': 'mask_ratio',
    'model_cfg.recon.lambda': 'lambda',
    'model_cfg.recon.lambda_edge': 'lambda_edge',
    'model_cfg.recon.edge_sobel_thr_ratio': 'edge_sobel_thr_ratio',
    'model_cfg.recon.edge_sobel_thr_abs': 'edge_sobel_thr_abs',
}


def read_results(path: Path):
    data = json.loads(path.read_text(encoding='utf-8'))
    history = [item for item in data.get('history', []) if item.get('status') == 'ok' and item.get('metric') is not None]
    history.sort(key=lambda item: item['metric'], reverse=True)
    if not history:
        raise ValueError('No valid trials found in results.json')
    return data, history


def linear_map(value, src_min, src_max, dst_min, dst_max):
    if src_max == src_min:
        return (dst_min + dst_max) / 2.0
    return dst_min + (value - src_min) / (src_max - src_min) * (dst_max - dst_min)


def esc(text):
    return str(text).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def svg_text(x, y, text, size=14, weight='normal', anchor='start', fill='#222'):
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}" font-family="Segoe UI, Arial, sans-serif">{esc(text)}</text>'


def score_to_color(value, vmin, vmax):
    t = 0.5 if vmax == vmin else max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
    r = int(linear_map(t, 0, 1, 46, 215))
    g = int(linear_map(1 - abs(t - 0.5) * 2, 0, 1, 96, 180))
    b = int(linear_map(1 - t, 0, 1, 228, 64))
    return f'#{r:02x}{g:02x}{b:02x}'


def build_topk_summary(history, top_k=10):
    top = history[:top_k]
    lines = []
    lines.append('# Tuning Top-K Summary')
    lines.append('')
    lines.append(f'- top_k: {top_k}')
    lines.append(f'- best_trial: {top[0]["trial_id"]}')
    lines.append(f'- best_metric: {top[0]["metric"]:.3f}')
    lines.append('')
    lines.append('## Top-K Trials')
    lines.append('')
    lines.append('| rank | trial_id | metric | mask_ratio | lambda | lambda_edge | edge_thr_ratio | edge_thr_abs |')
    lines.append('| --- | --- | --- | --- | --- | --- | --- | --- |')
    for idx, item in enumerate(top, start=1):
        p = item['params']
        lines.append(
            f"| {idx} | {item['trial_id']} | {item['metric']:.3f} | {p['model_cfg.recon.mask_ratio']} | {p['model_cfg.recon.lambda']} | {p['model_cfg.recon.lambda_edge']} | {p['model_cfg.recon.edge_sobel_thr_ratio']} | {p['model_cfg.recon.edge_sobel_thr_abs']} |"
        )
    lines.append('')
    lines.append('## Parameter Frequency In Top-K')
    lines.append('')
    for key in top[0]['params']:
        counts = Counter(item['params'][key] for item in top)
        lines.append(f'- {key}: {counts.most_common()}')
    lines.append('')
    lines.append('## Repeated Edge Threshold Pairs')
    lines.append('')
    for (thr_ratio, thr_abs), count in Counter(
        (item['params']['model_cfg.recon.edge_sobel_thr_ratio'], item['params']['model_cfg.recon.edge_sobel_thr_abs'])
        for item in history
    ).most_common():
        if count < 2:
            continue
        subset = [item for item in history if item['params']['model_cfg.recon.edge_sobel_thr_ratio'] == thr_ratio and item['params']['model_cfg.recon.edge_sobel_thr_abs'] == thr_abs]
        metrics = [item['metric'] for item in subset]
        lines.append(
            f'- edge_sobel_thr_ratio={thr_ratio}, edge_sobel_thr_abs={thr_abs}: n={count}, mean_metric={sum(metrics)/len(metrics):.3f}, best_metric={max(metrics):.3f}'
        )
    lines.append('')
    lines.append('## One-Parameter Trend Slices')
    lines.append('')
    for info in choose_one_param_slices(history):
        fixed_desc = ', '.join(
            f'{SHORT_NAMES[key]}={value}' for key, value in zip(info['fixed_keys'], info['fixed_values'])
        )
        trend_desc = ', '.join(
            f'{value}: {mean_metric:.3f}' for value, mean_metric in info['means']
        )
        lines.append(
            f'- vary {SHORT_NAMES[info["vary_key"]]} with fixed [{fixed_desc}] -> n={len(info["subset"])}, values={trend_desc}'
        )
    lines.append('')
    return '\n'.join(lines)


def choose_fixed_slice(history):
    fixed_key1 = 'model_cfg.recon.edge_sobel_thr_ratio'
    fixed_key2 = 'model_cfg.recon.edge_sobel_thr_abs'
    pairs = Counter((item['params'][fixed_key1], item['params'][fixed_key2]) for item in history)
    pair, count = pairs.most_common(1)[0]
    subset = [item for item in history if item['params'][fixed_key1] == pair[0] and item['params'][fixed_key2] == pair[1]]
    subset.sort(key=lambda item: item['params']['model_cfg.recon.mask_ratio'])
    return {
        'fixed_keys': (fixed_key1, fixed_key2),
        'fixed_values': pair,
        'count': count,
        'subset': subset,
        'vary_key': 'model_cfg.recon.mask_ratio',
    }


def choose_multiple_slices(history, max_slices=4):
    fixed_key1 = 'model_cfg.recon.edge_sobel_thr_ratio'
    fixed_key2 = 'model_cfg.recon.edge_sobel_thr_abs'
    pairs = Counter((item['params'][fixed_key1], item['params'][fixed_key2]) for item in history)
    slices = []
    for pair, count in pairs.most_common():
        subset = [item for item in history if item['params'][fixed_key1] == pair[0] and item['params'][fixed_key2] == pair[1]]
        vary_counts = Counter(item['params']['model_cfg.recon.mask_ratio'] for item in subset)
        if count < 3 or len(vary_counts) < 2:
            continue
        subset.sort(key=lambda item: (item['params']['model_cfg.recon.mask_ratio'], -item['metric']))
        slices.append({
            'fixed_keys': (fixed_key1, fixed_key2),
            'fixed_values': pair,
            'count': count,
            'subset': subset,
            'vary_key': 'model_cfg.recon.mask_ratio',
        })
        if len(slices) >= max_slices:
            break
    return slices


def choose_one_param_slice(history, vary_key):
    other_keys = [key for key in PARAM_KEYS if key != vary_key]
    for fixed_count in (len(other_keys), len(other_keys) - 1):
        candidate_infos = []
        if fixed_count <= 0:
            continue
        from itertools import combinations
        for fixed_keys in combinations(other_keys, fixed_count):
            free_keys = [key for key in other_keys if key not in fixed_keys]
            groups = defaultdict(list)
            for item in history:
                fixed_values = tuple(item['params'][key] for key in fixed_keys)
                groups[fixed_values].append(item)

            for fixed_values, subset in groups.items():
                grouped = defaultdict(list)
                for item in subset:
                    grouped[item['params'][vary_key]].append(item)
                if len(grouped) < 2:
                    continue
                count = len(subset)
                best_metric = max(item['metric'] for item in subset)
                mean_metric = sum(item['metric'] for item in subset) / count
                score = (len(grouped), count, best_metric, mean_metric)
                means = []
                for value in sorted(grouped):
                    items = grouped[value]
                    means.append((value, sum(entry['metric'] for entry in items) / len(items)))
                candidate_infos.append({
                    'score': score,
                    'vary_key': vary_key,
                    'fixed_keys': fixed_keys,
                    'fixed_values': fixed_values,
                    'free_keys': free_keys,
                    'subset': sorted(subset, key=lambda item: (item['params'][vary_key], -item['metric'])),
                    'means': means,
                    'mode': 'strict' if not free_keys else 'aggregated',
                })

        if candidate_infos:
            candidate_infos.sort(key=lambda info: info['score'], reverse=True)
            best_info = candidate_infos[0]
            best_info.pop('score', None)
            return best_info
    return None


def choose_one_param_slices(history):
    slices = []
    for vary_key in PARAM_KEYS:
        info = choose_one_param_slice(history, vary_key)
        if info is not None:
            slices.append(info)
    return slices


def aggregate_pair_means(history, x_key, y_key):
    buckets = defaultdict(list)
    for item in history:
        buckets[(item['params'][x_key], item['params'][y_key])].append(item['metric'])
    return {
        pair: {
            'mean': sum(metrics) / len(metrics),
            'count': len(metrics),
            'best': max(metrics),
        }
        for pair, metrics in buckets.items()
    }


def panel_frame(x, y, w, h, title, subtitle=None):
    parts = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="#ffffff" stroke="#d8d8d8"/>']
    parts.append(svg_text(x + 14, y + 24, title, size=18, weight='bold'))
    if subtitle:
        parts.append(svg_text(x + 14, y + 46, subtitle, size=12, fill='#666'))
    return parts


def draw_color_legend(x, y, w, h, vmin, vmax):
    parts = []
    steps = 24
    cell_w = w / steps
    for i in range(steps):
        value = linear_map(i, 0, steps - 1, vmin, vmax)
        parts.append(
            f'<rect x="{x + i * cell_w:.1f}" y="{y:.1f}" width="{cell_w + 1:.1f}" height="{h:.1f}" fill="{score_to_color(value, vmin, vmax)}" stroke="none"/>'
        )
    parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="none" stroke="#cccccc"/>')
    parts.append(svg_text(x, y + h + 16, f'{vmin:.2f}', size=11, fill='#666'))
    parts.append(svg_text(x + w, y + h + 16, f'{vmax:.2f}', size=11, anchor='end', fill='#666'))
    parts.append(svg_text(x + w / 2, y - 6, 'metric color scale', size=11, anchor='middle', fill='#666'))
    return parts


def make_topk_bar_panel(history, x, y, w, h):
    top = history[:10]
    metrics = [item['metric'] for item in top]
    mmin, mmax = min(metrics), max(metrics)
    parts = panel_frame(x, y, w, h, 'Top-10 Metrics')
    base_y = y + h - 36
    left = x + 50
    right = x + w - 20
    top_y = y + 70
    usable_h = base_y - top_y
    bar_w = (right - left) / len(top) * 0.65
    gap = (right - left) / len(top)
    for i in range(5):
        gy = top_y + linear_map(i, 0, 4, usable_h, 0)
        gv = linear_map(i, 0, 4, mmin - 0.05, mmax + 0.05)
        parts.append(f'<line x1="{left}" y1="{gy:.1f}" x2="{right}" y2="{gy:.1f}" stroke="#efefef"/>')
        parts.append(svg_text(left - 8, gy + 4, f'{gv:.2f}', size=11, anchor='end', fill='#666'))
    for idx, item in enumerate(top):
        bx = left + idx * gap + (gap - bar_w) / 2
        by = top_y + linear_map(item['metric'], mmin - 0.05, mmax + 0.05, usable_h, 0)
        color = '#d7263d' if idx == 0 else '#2878ff'
        parts.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{base_y - by:.1f}" fill="{color}" opacity="0.9" rx="4"/>')
        parts.append(svg_text(bx + bar_w / 2, base_y + 18, item['trial_id'], size=11, anchor='middle', fill='#666'))
    return '\n'.join(parts)


def make_slice_panel(slice_info, x, y, w, h):
    subset = slice_info['subset']
    fixed_key1, fixed_key2 = slice_info['fixed_keys']
    fixed_val1, fixed_val2 = slice_info['fixed_values']
    subtitle = f'fixed {fixed_key1.split(".")[-1]}={fixed_val1}, {fixed_key2.split(".")[-1]}={fixed_val2} (n={len(subset)})'
    parts = panel_frame(x, y, w, h, 'Single-Parameter Slice', subtitle)
    if len(subset) < 2:
        parts.append(svg_text(x + 20, y + 90, 'Not enough points for a useful slice.', size=14, fill='#a33'))
        return '\n'.join(parts)

    grouped = defaultdict(list)
    for item in subset:
        grouped[item['params'][slice_info['vary_key']]].append(item)

    vals = sorted(grouped)
    metrics = [item['metric'] for item in subset]
    xmin, xmax = min(vals), max(vals)
    ymin, ymax = min(metrics) - 0.05, max(metrics) + 0.05
    left, right = x + 55, x + w - 24
    top_y, base_y = y + 70, y + h - 38
    for i in range(4):
        gy = top_y + linear_map(i, 0, 3, base_y - top_y, 0)
        gv = linear_map(i, 0, 3, ymin, ymax)
        parts.append(f'<line x1="{left}" y1="{gy:.1f}" x2="{right}" y2="{gy:.1f}" stroke="#efefef"/>')
        parts.append(svg_text(left - 8, gy + 4, f'{gv:.2f}', size=11, anchor='end', fill='#666'))

    mean_poly = []
    for value in vals:
        items = grouped[value]
        mean_metric = sum(item['metric'] for item in items) / len(items)
        px = linear_map(value, xmin, xmax, left, right)
        py = linear_map(mean_metric, ymin, ymax, base_y, top_y)
        mean_poly.append((px, py, value, items, mean_metric))

    parts.append(f'<polyline points="{" ".join(f"{px:.1f},{py:.1f}" for px, py, _, _, _ in mean_poly)}" fill="none" stroke="#d7263d" stroke-width="3"/>')
    for px, py, value, items, mean_metric in mean_poly:
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6" fill="#d7263d"/>')
        parts.append(svg_text(px, py - 12, f'mean={mean_metric:.3f} (n={len(items)})', size=10, anchor='middle', fill='#444'))
        if len(items) > 1:
            for idx, item in enumerate(sorted(items, key=lambda entry: entry['metric'], reverse=True)[:3]):
                dot_y = linear_map(item['metric'], ymin, ymax, base_y, top_y)
                dot_x = px - 10 + idx * 10
                parts.append(f'<circle cx="{dot_x:.1f}" cy="{dot_y:.1f}" r="3.5" fill="#2878ff" opacity="0.85"/>')
        else:
            item = items[0]
            dot_y = linear_map(item['metric'], ymin, ymax, base_y, top_y)
            parts.append(f'<circle cx="{px:.1f}" cy="{dot_y:.1f}" r="3.5" fill="#2878ff" opacity="0.85"/>')
    parts.append(svg_text((left + right) / 2, y + h - 10, slice_info['vary_key'].split('.')[-1], size=12, anchor='middle', fill='#666'))
    return '\n'.join(parts)


def make_one_param_panel(slice_info, x, y, w, h):
    fixed_desc = ', '.join(
        f'{SHORT_NAMES[key]}={value}' for key, value in zip(slice_info['fixed_keys'], slice_info['fixed_values'])
    )
    if slice_info.get('free_keys'):
        subtitle = f'fixed {fixed_desc}; averaged over {", ".join(SHORT_NAMES[key] for key in slice_info["free_keys"])}'
    else:
        subtitle = f'fixed {fixed_desc}'
    parts = panel_frame(x, y, w, h, f'Trend: {SHORT_NAMES[slice_info["vary_key"]]}', subtitle)
    subset = slice_info['subset']
    grouped = defaultdict(list)
    for item in subset:
        grouped[item['params'][slice_info['vary_key']]].append(item)

    xs = sorted(grouped)
    if len(xs) < 2:
        parts.append(svg_text(x + 20, y + 90, 'Not enough points for a useful trend.', size=14, fill='#a33'))
        return '\n'.join(parts)

    all_metrics = [item['metric'] for item in subset]
    ymin, ymax = min(all_metrics) - 0.05, max(all_metrics) + 0.05
    left, right = x + 60, x + w - 24
    top_y, base_y = y + 70, y + h - 42

    for i in range(4):
        gy = top_y + linear_map(i, 0, 3, base_y - top_y, 0)
        gv = linear_map(i, 0, 3, ymin, ymax)
        parts.append(f'<line x1="{left}" y1="{gy:.1f}" x2="{right}" y2="{gy:.1f}" stroke="#efefef"/>')
        parts.append(svg_text(left - 8, gy + 4, f'{gv:.2f}', size=11, anchor='end', fill='#666'))

    mean_points = []
    for value in xs:
        items = grouped[value]
        mean_metric = sum(item['metric'] for item in items) / len(items)
        px = linear_map(value, min(xs), max(xs), left, right)
        py = linear_map(mean_metric, ymin, ymax, base_y, top_y)
        mean_points.append((px, py, value, mean_metric, items))

    parts.append(f'<polyline points="{" ".join(f"{px:.1f},{py:.1f}" for px, py, _, _, _ in mean_points)}" fill="none" stroke="#d7263d" stroke-width="3"/>')
    for px, py, value, mean_metric, items in mean_points:
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6" fill="#d7263d"/>')
        parts.append(svg_text(px, py - 12, f'{value:g}: {mean_metric:.3f}', size=10, anchor='middle', fill='#444'))
        spread = sorted(item['metric'] for item in items)
        for idx, metric in enumerate(spread[:4]):
            dot_x = px - 12 + idx * 8
            dot_y = linear_map(metric, ymin, ymax, base_y, top_y)
            parts.append(f'<circle cx="{dot_x:.1f}" cy="{dot_y:.1f}" r="3.2" fill="#2878ff" opacity="0.85"/>')
        parts.append(svg_text(px, base_y + 18, f'{value:g}', size=11, anchor='middle', fill='#666'))

    parts.append(svg_text((left + right) / 2, y + h - 10, SHORT_NAMES[slice_info['vary_key']], size=12, anchor='middle', fill='#666'))
    return '\n'.join(parts)


def make_interaction_panel(history, x_key, y_key, x, y, w, h, title):
    parts = panel_frame(x, y, w, h, title, f'color = metric')
    xs = [item['params'][x_key] for item in history]
    ys = [item['params'][y_key] for item in history]
    ms = [item['metric'] for item in history]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    mmin, mmax = min(ms), max(ms)
    left, right = x + 60, x + w - 24
    top_y, base_y = y + 70, y + h - 46

    for i in range(5):
        gx = linear_map(i, 0, 4, left, right)
        gy = linear_map(i, 0, 4, top_y, base_y)
        parts.append(f'<line x1="{gx:.1f}" y1="{top_y}" x2="{gx:.1f}" y2="{base_y}" stroke="#f3f3f3"/>')
        parts.append(f'<line x1="{left}" y1="{gy:.1f}" x2="{right}" y2="{gy:.1f}" stroke="#f3f3f3"/>')
        xv = linear_map(i, 0, 4, xmin, xmax)
        yv = linear_map(i, 0, 4, ymax, ymin)
        parts.append(svg_text(gx, base_y + 18, f'{xv:g}', size=11, anchor='middle', fill='#666'))
        parts.append(svg_text(left - 8, gy + 4, f'{yv:g}', size=11, anchor='end', fill='#666'))

    for item in history:
        px = linear_map(item['params'][x_key], xmin, xmax, left, right)
        py = linear_map(item['params'][y_key], ymin, ymax, base_y, top_y)
        color = score_to_color(item['metric'], mmin, mmax)
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5.5" fill="{color}" opacity="0.88"/>')

    best = max(history, key=lambda item: item['metric'])
    bx = linear_map(best['params'][x_key], xmin, xmax, left, right)
    by = linear_map(best['params'][y_key], ymin, ymax, base_y, top_y)
    parts.append(f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="8" fill="none" stroke="#d7263d" stroke-width="2"/>')
    parts.append(svg_text((left + right) / 2, y + h - 12, x_key.split('.')[-1], size=12, anchor='middle', fill='#666'))
    parts.append(svg_text(x + 18, (top_y + base_y) / 2, y_key.split('.')[-1], size=12, fill='#666'))
    return '\n'.join(parts)


def make_heatmap_panel(history, x_key, y_key, x, y, w, h, title):
    agg = aggregate_pair_means(history, x_key, y_key)
    xs = sorted({pair[0] for pair in agg})
    ys = sorted({pair[1] for pair in agg})
    means = [entry['mean'] for entry in agg.values()]
    mmin, mmax = min(means), max(means)
    parts = panel_frame(x, y, w, h, title, 'cell color = mean metric, label = mean / count')
    left, right = x + 70, x + w - 20
    top_y, base_y = y + 80, y + h - 56
    cell_w = (right - left) / max(len(xs), 1)
    cell_h = (base_y - top_y) / max(len(ys), 1)

    for row, yv in enumerate(reversed(ys)):
        cy = top_y + row * cell_h
        parts.append(svg_text(left - 10, cy + cell_h / 2 + 4, f'{yv:g}', size=11, anchor='end', fill='#666'))
        for col, xv in enumerate(xs):
            cx = left + col * cell_w
            entry = agg.get((xv, yv))
            if entry is None:
                parts.append(f'<rect x="{cx:.1f}" y="{cy:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" fill="#f4f4f4" stroke="#ffffff"/>')
                parts.append(svg_text(cx + cell_w / 2, cy + cell_h / 2 + 4, '-', size=11, anchor='middle', fill='#999'))
                continue
            color = score_to_color(entry['mean'], mmin, mmax)
            parts.append(f'<rect x="{cx:.1f}" y="{cy:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" fill="{color}" stroke="#ffffff"/>')
            parts.append(svg_text(cx + cell_w / 2, cy + cell_h / 2 - 4, f"{entry['mean']:.2f}", size=11, weight='bold', anchor='middle', fill='#222'))
            parts.append(svg_text(cx + cell_w / 2, cy + cell_h / 2 + 12, f"n={entry['count']}", size=10, anchor='middle', fill='#333'))

    for col, xv in enumerate(xs):
        cx = left + col * cell_w + cell_w / 2
        parts.append(svg_text(cx, base_y + 18, f'{xv:g}', size=11, anchor='middle', fill='#666'))

    parts.extend(draw_color_legend(right - 220, y + 24, 190, 12, mmin, mmax))
    parts.append(svg_text((left + right) / 2, y + h - 16, x_key.split('.')[-1], size=12, anchor='middle', fill='#666'))
    parts.append(svg_text(x + 18, (top_y + base_y) / 2, y_key.split('.')[-1], size=12, fill='#666'))
    return '\n'.join(parts)


def build_svg(history):
    slice_info = choose_fixed_slice(history)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">']
    parts.append('<rect width="100%" height="100%" fill="#fafafa"/>')
    parts.append(svg_text(34, 38, 'OpenGait Tuning Analysis', size=28, weight='bold'))
    parts.append(svg_text(34, 64, 'Top-k summary + fixed-parameter slice + pairwise interactions', size=14, fill='#555'))

    parts.append(make_topk_bar_panel(history, 60, 100, 1380, 280))
    parts.append(make_slice_panel(slice_info, 60, 420, 680, 300))
    parts.append(make_interaction_panel(history, 'model_cfg.recon.mask_ratio', 'model_cfg.recon.lambda', 760, 420, 680, 300, 'Interaction: mask_ratio × lambda'))
    parts.append(make_interaction_panel(history, 'model_cfg.recon.lambda', 'model_cfg.recon.lambda_edge', 410, 760, 680, 300, 'Interaction: lambda × lambda_edge'))
    parts.append('</svg>')
    return '\n'.join(parts)


def build_multi_slice_svg(history):
    slices = choose_multiple_slices(history, max_slices=4)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{SLICE_SVG_WIDTH}" height="{SLICE_SVG_HEIGHT}" viewBox="0 0 {SLICE_SVG_WIDTH} {SLICE_SVG_HEIGHT}">']
    parts.append('<rect width="100%" height="100%" fill="#fafafa"/>')
    parts.append(svg_text(34, 38, 'OpenGait Fixed Slice Analysis', size=28, weight='bold'))
    parts.append(svg_text(34, 64, 'Different edge threshold pairs, each panel shows mean metric trend over mask_ratio', size=14, fill='#555'))

    if not slices:
        parts.append(svg_text(60, 120, 'No repeated edge-threshold slice with enough points was found.', size=16, fill='#a33'))
    else:
        positions = [
            (60, 110, 660, 420),
            (780, 110, 660, 420),
            (60, 580, 660, 420),
            (780, 580, 660, 420),
        ]
        for slice_info, (x, y, w, h) in zip(slices, positions):
            parts.append(make_slice_panel(slice_info, x, y, w, h))

    parts.append('</svg>')
    return '\n'.join(parts)


def build_heatmap_svg(history):
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{HEATMAP_SVG_WIDTH}" height="{HEATMAP_SVG_HEIGHT}" viewBox="0 0 {HEATMAP_SVG_WIDTH} {HEATMAP_SVG_HEIGHT}">']
    parts.append('<rect width="100%" height="100%" fill="#fafafa"/>')
    parts.append(svg_text(34, 38, 'OpenGait Mean-Metric Heatmaps', size=28, weight='bold'))
    parts.append(svg_text(34, 64, 'Each cell aggregates repeated parameter pairs using average metric', size=14, fill='#555'))
    parts.append(make_heatmap_panel(history, 'model_cfg.recon.mask_ratio', 'model_cfg.recon.lambda', 60, 110, 660, 360, 'Heatmap: mask_ratio × lambda'))
    parts.append(make_heatmap_panel(history, 'model_cfg.recon.lambda', 'model_cfg.recon.lambda_edge', 780, 110, 660, 360, 'Heatmap: lambda × lambda_edge'))
    parts.append(make_heatmap_panel(history, 'model_cfg.recon.edge_sobel_thr_ratio', 'model_cfg.recon.edge_sobel_thr_abs', 410, 520, 680, 340, 'Heatmap: edge_sobel_thr_ratio × edge_sobel_thr_abs'))
    parts.append('</svg>')
    return '\n'.join(parts)


def build_one_param_svg(history):
    slices = choose_one_param_slices(history)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{ONE_PARAM_SVG_WIDTH}" height="{ONE_PARAM_SVG_HEIGHT}" viewBox="0 0 {ONE_PARAM_SVG_WIDTH} {ONE_PARAM_SVG_HEIGHT}">']
    parts.append('<rect width="100%" height="100%" fill="#fafafa"/>')
    parts.append(svg_text(34, 38, 'OpenGait One-Parameter Trends', size=28, weight='bold'))
    parts.append(svg_text(34, 64, 'Prefer 4-fixed slices; if unavailable, fallback to 3-fixed slices with averaging over the remaining free parameter', size=14, fill='#555'))

    positions = [
        (60, 110, 660, 300),
        (780, 110, 660, 300),
        (60, 450, 660, 300),
        (780, 450, 660, 300),
        (410, 790, 680, 300),
    ]

    if not slices:
        parts.append(svg_text(60, 120, 'No valid one-parameter slices were found even after fallback.', size=16, fill='#a33'))
    else:
        for slice_info, (x, y, w, h) in zip(slices, positions):
            parts.append(make_one_param_panel(slice_info, x, y, w, h))

    parts.append('</svg>')
    return '\n'.join(parts)


def main():
    root = Path(__file__).resolve().parents[1]
    results_path = root / 'results.json'
    summary_path = root / 'results_topk_summary.md'
    svg_path = root / 'results_interactions.svg'
    multi_slice_svg_path = root / 'results_fixed_slices.svg'
    heatmap_svg_path = root / 'results_heatmaps.svg'
    one_param_svg_path = root / 'results_one_param_trends.svg'
    _, history = read_results(results_path)
    summary_path.write_text(build_topk_summary(history, top_k=10), encoding='utf-8')
    svg_path.write_text(build_svg(history), encoding='utf-8')
    multi_slice_svg_path.write_text(build_multi_slice_svg(history), encoding='utf-8')
    heatmap_svg_path.write_text(build_heatmap_svg(history), encoding='utf-8')
    one_param_svg_path.write_text(build_one_param_svg(history), encoding='utf-8')
    print(summary_path)
    print(svg_path)
    print(multi_slice_svg_path)
    print(heatmap_svg_path)
    print(one_param_svg_path)


if __name__ == '__main__':
    main()