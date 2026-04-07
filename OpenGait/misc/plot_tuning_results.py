import json
from pathlib import Path
from statistics import mean


SVG_WIDTH = 1400
SVG_HEIGHT = 1040
MARGIN_LEFT = 90
MARGIN_RIGHT = 40
MARGIN_TOP = 60
MARGIN_BOTTOM = 60


def read_results(path: Path):
    data = json.loads(path.read_text(encoding='utf-8'))
    history = [item for item in data.get('history', []) if item.get('status') == 'ok' and item.get('metric') is not None]
    if not history:
        raise ValueError('No valid trials found in results.json')
    return data, history


def linear_map(value, src_min, src_max, dst_min, dst_max):
    if src_max == src_min:
        return (dst_min + dst_max) / 2.0
    ratio = (value - src_min) / (src_max - src_min)
    return dst_min + ratio * (dst_max - dst_min)


def svg_text(x, y, text, size=14, weight='normal', anchor='start', fill='#222'):
    safe = (str(text).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}" font-family="Segoe UI, Arial, sans-serif">{safe}</text>'


def make_line_chart(history, x, y, width, height):
    metrics = [item['metric'] for item in history]
    trials = [item['trial_id'] for item in history]
    metric_min = min(metrics)
    metric_max = max(metrics)
    y_min = metric_min - 0.2
    y_max = metric_max + 0.2

    parts = []
    parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="#ffffff" stroke="#d9d9d9" rx="8"/>')

    # grid
    for i in range(5):
        gy = y + linear_map(i, 0, 4, height, 0)
        value = linear_map(i, 0, 4, y_min, y_max)
        parts.append(f'<line x1="{x}" y1="{gy:.1f}" x2="{x + width}" y2="{gy:.1f}" stroke="#ececec"/>')
        parts.append(svg_text(x - 12, gy + 5, f'{value:.2f}', size=12, anchor='end', fill='#666'))

    n = len(history)
    for i in range(n):
        gx = x + linear_map(i, 0, max(n - 1, 1), 0, width)
        if i < n - 1:
            parts.append(f'<line x1="{gx:.1f}" y1="{y}" x2="{gx:.1f}" y2="{y + height}" stroke="#f7f7f7"/>')
        if i % 4 == 0 or i == n - 1:
            parts.append(svg_text(gx, y + height + 22, trials[i], size=12, anchor='middle', fill='#666'))

    pts = []
    for idx, item in enumerate(history):
        px = x + linear_map(idx, 0, max(n - 1, 1), 0, width)
        py = y + linear_map(item['metric'], y_min, y_max, height, 0)
        pts.append((px, py, item))

    polyline = ' '.join(f'{px:.1f},{py:.1f}' for px, py, _ in pts)
    parts.append(f'<polyline points="{polyline}" fill="none" stroke="#2878ff" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>')

    best_idx = max(range(n), key=lambda idx: history[idx]['metric'])
    avg_metric = mean(metrics)
    avg_y = y + linear_map(avg_metric, y_min, y_max, height, 0)
    parts.append(f'<line x1="{x}" y1="{avg_y:.1f}" x2="{x + width}" y2="{avg_y:.1f}" stroke="#ff9f1c" stroke-width="2" stroke-dasharray="6 6"/>')
    parts.append(svg_text(x + width - 4, avg_y - 8, f'avg {avg_metric:.3f}', size=12, anchor='end', fill='#c46b00'))

    for idx, (px, py, item) in enumerate(pts):
        radius = 6 if idx == best_idx else 4
        color = '#d7263d' if idx == best_idx else '#2878ff'
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{radius}" fill="{color}" opacity="0.95"/>')

    best = history[best_idx]
    best_x = pts[best_idx][0]
    best_y = pts[best_idx][1]
    parts.append(f'<line x1="{best_x:.1f}" y1="{best_y:.1f}" x2="{best_x + 70:.1f}" y2="{best_y - 36:.1f}" stroke="#d7263d" stroke-width="1.5"/>')
    parts.append(svg_text(best_x + 76, best_y - 40, f'best trial {best["trial_id"]}: {best["metric"]:.3f}', size=13, weight='bold', fill='#d7263d'))

    parts.append(svg_text(x, y - 18, 'Metric By Trial', size=18, weight='bold'))
    return '\n'.join(parts)


def make_scatter_panel(history, key, title, x, y, width, height, color):
    values = [item['params'][key] for item in history]
    metrics = [item['metric'] for item in history]
    x_min = min(values)
    x_max = max(values)
    y_min = min(metrics) - 0.2
    y_max = max(metrics) + 0.2

    parts = []
    parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="#ffffff" stroke="#d9d9d9" rx="8"/>')
    parts.append(svg_text(x + 12, y + 22, title, size=14, weight='bold'))

    for i in range(4):
        gy = y + 34 + linear_map(i, 0, 3, height - 54, 0)
        parts.append(f'<line x1="{x + 40}" y1="{gy:.1f}" x2="{x + width - 14}" y2="{gy:.1f}" stroke="#f1f1f1"/>')

    for item in history:
        vx = item['params'][key]
        vm = item['metric']
        px = linear_map(vx, x_min, x_max, x + 46, x + width - 18)
        py = linear_map(vm, y_min, y_max, y + height - 20, y + 36)
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.5" fill="{color}" opacity="0.82"/>')

    parts.append(svg_text(x + 46, y + height + 18, f'min {x_min:g}', size=11, fill='#666'))
    parts.append(svg_text(x + width - 18, y + height + 18, f'max {x_max:g}', size=11, anchor='end', fill='#666'))
    return '\n'.join(parts)


def build_svg(data, history):
    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">')
    parts.append('<rect width="100%" height="100%" fill="#fafafa"/>')
    parts.append(svg_text(36, 38, 'OpenGait Hyper-Parameter Search', size=28, weight='bold'))

    best = data['best']
    summary = (
        f"best trial={best['trial_id']}  score={best['metric']:.3f}  "
        f"mask={best['params']['model_cfg.recon.mask_ratio']}  "
        f"lambda={best['params']['model_cfg.recon.lambda']}  "
        f"lambda_edge={best['params']['model_cfg.recon.lambda_edge']}"
    )
    parts.append(svg_text(36, 66, summary, size=14, fill='#444'))

    chart_x = MARGIN_LEFT
    chart_y = 96
    chart_w = SVG_WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    chart_h = 360
    parts.append(make_line_chart(history, chart_x, chart_y, chart_w, chart_h))

    keys = [
        ('model_cfg.recon.mask_ratio', 'mask_ratio', '#2878ff'),
        ('model_cfg.recon.lambda', 'lambda', '#00a896'),
        ('model_cfg.recon.lambda_edge', 'lambda_edge', '#ff595e'),
        ('model_cfg.recon.edge_sobel_thr_ratio', 'edge_sobel_thr_ratio', '#8ac926'),
        ('model_cfg.recon.edge_sobel_thr_abs', 'edge_sobel_thr_abs', '#6a4c93'),
    ]

    panel_w = 420
    panel_h = 210
    start_x = 70
    gap_x = 28
    start_y = 520
    gap_y = 44
    for idx, (key, title, color) in enumerate(keys):
        row = idx // 2
        col = idx % 2
        px = start_x + col * (panel_w + gap_x)
        py = start_y + row * (panel_h + gap_y)
        if idx == 4:
            px = (SVG_WIDTH - panel_w) / 2
        parts.append(make_scatter_panel(history, key, title, px, py, panel_w, panel_h, color))

    parts.append('</svg>')
    return '\n'.join(parts)


def main():
    root = Path(__file__).resolve().parents[1]
    results_path = root / 'results.json'
    out_path = root / 'results_plot.svg'
    data, history = read_results(results_path)
    out_path.write_text(build_svg(data, history), encoding='utf-8')
    print(out_path)


if __name__ == '__main__':
    main()