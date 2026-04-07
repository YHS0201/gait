import json
from pathlib import Path


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


def load_history(results_path: Path):
    data = json.loads(results_path.read_text(encoding='utf-8'))
    history = [
        {
            'trial_id': item['trial_id'],
            'metric': item['metric'],
            'params': {SHORT_NAMES[key]: item['params'][key] for key in PARAM_KEYS},
        }
        for item in data.get('history', [])
        if item.get('status') == 'ok' and item.get('metric') is not None
    ]
    if not history:
        raise ValueError('No valid trials found in results.json')
    return history


def build_html(history):
    data_json = json.dumps(history, ensure_ascii=False)
    dimensions_json = json.dumps(list(SHORT_NAMES.values()), ensure_ascii=False)
    return f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>OpenGait 5D Heatmap Explorer</title>
  <style>
    :root {{
      --bg: #f6f6f2;
      --panel: #ffffff;
      --line: #d7d5cf;
      --text: #1c1c1c;
      --muted: #666157;
      --accent: #b33a3a;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: radial-gradient(circle at top left, #fff9ef 0%, var(--bg) 55%);
      color: var(--text);
      font-family: "Segoe UI", "PingFang SC", sans-serif;
    }}
    .wrap {{ max-width: 1700px; margin: 0 auto; padding: 28px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    .sub {{ color: var(--muted); margin-bottom: 22px; }}
    .controls {{
      display: grid;
      grid-template-columns: repeat(6, minmax(150px, 1fr));
      gap: 12px;
      background: rgba(255,255,255,0.88);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px;
      backdrop-filter: blur(10px);
    }}
    label {{ display: flex; flex-direction: column; gap: 6px; font-size: 13px; color: var(--muted); }}
    select {{
      height: 38px;
      border-radius: 10px;
      border: 1px solid #ccc7bc;
      background: #fff;
      padding: 0 10px;
      color: var(--text);
    }}
    .summary {{ margin: 16px 0 18px; color: var(--muted); font-size: 14px; }}
    .legend {{ display: flex; align-items: center; gap: 12px; margin-bottom: 18px; }}
    .legend-bar {{
      width: 260px;
      height: 14px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: linear-gradient(90deg, #2e60e4 0%, #8bbf83 50%, #d64a4a 100%);
    }}
    .grid {{ display: grid; gap: 14px; }}
    .facet {{
      background: rgba(255,255,255,0.9);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 12px;
      min-height: 220px;
    }}
    .facet-title {{ font-size: 13px; color: var(--muted); margin-bottom: 10px; }}
    .heatmap {{
      display: grid;
      gap: 2px;
      align-items: stretch;
      justify-items: stretch;
    }}
    .y-label, .x-label {{
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--muted);
      font-size: 12px;
      min-height: 28px;
      min-width: 56px;
    }}
    .cell {{
      min-width: 92px;
      min-height: 54px;
      border-radius: 10px;
      padding: 6px;
      border: 1px solid rgba(255,255,255,0.8);
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      text-align: center;
      font-size: 11px;
      line-height: 1.2;
    }}
    .cell.empty {{ background: #ece9e1; color: #9b9588; }}
    .cell .m {{ font-weight: 700; font-size: 12px; }}
    .cell .n {{ opacity: 0.86; }}
    .empty-note {{ color: #9b9588; font-size: 13px; padding: 20px 8px; }}
    @media (max-width: 1200px) {{
      .controls {{ grid-template-columns: repeat(2, minmax(150px, 1fr)); }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>OpenGait 5D Heatmap Explorer</h1>
    <div class="sub">4 个维度分面显示，剩下 1 个维度用筛选控制；颜色表示平均 metric。</div>

    <div class="controls">
      <label>横轴
        <select id="xDim"></select>
      </label>
      <label>纵轴
        <select id="yDim"></select>
      </label>
      <label>分面行
        <select id="rowDim"></select>
      </label>
      <label>分面列
        <select id="colDim"></select>
      </label>
      <label>筛选维度
        <select id="filterDim"></select>
      </label>
      <label>筛选值
        <select id="filterVal"></select>
      </label>
    </div>

    <div class="summary" id="summary"></div>
    <div class="legend">
      <span>低</span>
      <div class="legend-bar"></div>
      <span>高</span>
      <span id="legendRange"></span>
    </div>
    <div id="grid" class="grid"></div>
  </div>

  <script>
    const data = {data_json};
    const dims = {dimensions_json};

    const defaults = {{
      xDim: 'mask_ratio',
      yDim: 'lambda',
      rowDim: 'edge_sobel_thr_ratio',
      colDim: 'edge_sobel_thr_abs',
      filterDim: 'lambda_edge'
    }};

    const state = {{ ...defaults, filterVal: null }};

    function uniqueSorted(values) {{
      return [...new Set(values)].sort((a, b) => Number(a) - Number(b));
    }}

    function clamp01(value) {{
      return Math.max(0, Math.min(1, value));
    }}

    function scoreToColor(value, min, max) {{
      const t = max === min ? 0.5 : clamp01((value - min) / (max - min));
      const r = Math.round(46 + (214 - 46) * t);
      const g = Math.round(96 + (191 - Math.abs(t - 0.5) * 2 * 95));
      const b = Math.round(228 + (74 - 228) * t);
      return `rgb(${{r}}, ${{g}}, ${{b}})`;
    }}

    function ensureUniqueDims(changedKey) {{
      const used = new Set();
      const order = ['xDim', 'yDim', 'rowDim', 'colDim', 'filterDim'];
      for (const key of order) {{
        if (key === changedKey) {{
          used.add(state[key]);
          continue;
        }}
        if (!used.has(state[key])) {{
          used.add(state[key]);
          continue;
        }}
        state[key] = dims.find(dim => !used.has(dim));
        used.add(state[key]);
      }}
    }}

    function populateSelect(selectId, selected, excluded = []) {{
      const select = document.getElementById(selectId);
      select.innerHTML = '';
      dims.filter(dim => !excluded.includes(dim)).forEach(dim => {{
        const option = document.createElement('option');
        option.value = dim;
        option.textContent = dim;
        option.selected = dim === selected;
        select.appendChild(option);
      }});
    }}

    function populateFilterValues() {{
      const values = uniqueSorted(data.map(item => item.params[state.filterDim]));
      const select = document.getElementById('filterVal');
      select.innerHTML = '';
      values.forEach(value => {{
        const option = document.createElement('option');
        option.value = String(value);
        option.textContent = String(value);
        option.selected = state.filterVal === null ? value === values[0] : String(value) === String(state.filterVal);
        select.appendChild(option);
      }});
      state.filterVal = select.value;
    }}

    function rebuildControls(changedKey = null) {{
      if (changedKey) ensureUniqueDims(changedKey);
      populateSelect('xDim', state.xDim);
      populateSelect('yDim', state.yDim);
      populateSelect('rowDim', state.rowDim);
      populateSelect('colDim', state.colDim);
      populateSelect('filterDim', state.filterDim);
      populateFilterValues();
    }}

    function aggregate(filtered) {{
      const buckets = new Map();
      for (const item of filtered) {{
        const rowValue = item.params[state.rowDim];
        const colValue = item.params[state.colDim];
        const xValue = item.params[state.xDim];
        const yValue = item.params[state.yDim];
        const key = [rowValue, colValue, xValue, yValue].join('|');
        if (!buckets.has(key)) {{
          buckets.set(key, {{ rowValue, colValue, xValue, yValue, metrics: [] }});
        }}
        buckets.get(key).metrics.push(item.metric);
      }}
      return [...buckets.values()].map(entry => {{
        const mean = entry.metrics.reduce((a, b) => a + b, 0) / entry.metrics.length;
        return {{ ...entry, mean, count: entry.metrics.length }};
      }});
    }}

    function render() {{
      const filtered = data.filter(item => String(item.params[state.filterDim]) === String(state.filterVal));
      const aggregated = aggregate(filtered);
      const rows = uniqueSorted(filtered.map(item => item.params[state.rowDim]));
      const cols = uniqueSorted(filtered.map(item => item.params[state.colDim]));
      const xs = uniqueSorted(filtered.map(item => item.params[state.xDim]));
      const ys = uniqueSorted(filtered.map(item => item.params[state.yDim]));
      const metrics = aggregated.map(item => item.mean);
      const minMetric = metrics.length ? Math.min(...metrics) : 0;
      const maxMetric = metrics.length ? Math.max(...metrics) : 1;

      document.getElementById('summary').textContent = `当前显示: filter ${{state.filterDim}}=${{state.filterVal}}，${{filtered.length}} 个 trial，${{aggregated.length}} 个聚合单元。`;
      document.getElementById('legendRange').textContent = `mean metric: ${{minMetric.toFixed(3)}} - ${{maxMetric.toFixed(3)}}`;

      const grid = document.getElementById('grid');
      grid.style.gridTemplateColumns = `repeat(${{Math.max(cols.length, 1)}}, minmax(320px, 1fr))`;
      grid.innerHTML = '';

      if (!filtered.length) {{
        const note = document.createElement('div');
        note.className = 'empty-note';
        note.textContent = '当前筛选下没有数据。';
        grid.appendChild(note);
        return;
      }}

      for (const rowValue of rows) {{
        for (const colValue of cols) {{
          const facet = document.createElement('div');
          facet.className = 'facet';
          const title = document.createElement('div');
          title.className = 'facet-title';
          title.textContent = `${{state.rowDim}}=${{rowValue}}, ${{state.colDim}}=${{colValue}}`;
          facet.appendChild(title);

          const heatmap = document.createElement('div');
          heatmap.className = 'heatmap';
          heatmap.style.gridTemplateColumns = `70px repeat(${{xs.length}}, minmax(92px, 1fr))`;
          heatmap.style.gridTemplateRows = `30px repeat(${{ys.length}}, minmax(54px, auto))`;

          const corner = document.createElement('div');
          corner.className = 'x-label';
          corner.textContent = `${{state.yDim}} \\ ${{state.xDim}}`;
          heatmap.appendChild(corner);

          for (const xValue of xs) {{
            const xLabel = document.createElement('div');
            xLabel.className = 'x-label';
            xLabel.textContent = xValue;
            heatmap.appendChild(xLabel);
          }}

          const facetData = aggregated.filter(item => item.rowValue === rowValue && item.colValue === colValue);
          for (const yValue of [...ys].reverse()) {{
            const yLabel = document.createElement('div');
            yLabel.className = 'y-label';
            yLabel.textContent = yValue;
            heatmap.appendChild(yLabel);

            for (const xValue of xs) {{
              const cell = document.createElement('div');
              const entry = facetData.find(item => item.xValue === xValue && item.yValue === yValue);
              if (!entry) {{
                cell.className = 'cell empty';
                cell.textContent = '-';
              }} else {{
                cell.className = 'cell';
                cell.style.background = scoreToColor(entry.mean, minMetric, maxMetric);
                cell.innerHTML = `<div class="m">${{entry.mean.toFixed(2)}}</div><div class="n">n=${{entry.count}}</div>`;
                cell.title = `mean=${{entry.mean.toFixed(3)}}, count=${{entry.count}}`;
              }}
              heatmap.appendChild(cell);
            }}
          }}

          facet.appendChild(heatmap);
          grid.appendChild(facet);
        }}
      }}
    }}

    function bindControl(selectId, stateKey) {{
      document.getElementById(selectId).addEventListener('change', event => {{
        state[stateKey] = event.target.value;
        rebuildControls(stateKey);
        render();
      }});
    }}

    rebuildControls();
    bindControl('xDim', 'xDim');
    bindControl('yDim', 'yDim');
    bindControl('rowDim', 'rowDim');
    bindControl('colDim', 'colDim');
    bindControl('filterDim', 'filterDim');
    document.getElementById('filterVal').addEventListener('change', event => {{
      state.filterVal = event.target.value;
      render();
    }});
    render();
  </script>
</body>
</html>
'''


def main():
    root = Path(__file__).resolve().parents[1]
    results_path = root / 'results.json'
    output_path = root / 'results_5d_heatmap.html'
    history = load_history(results_path)
    output_path.write_text(build_html(history), encoding='utf-8')
    print(output_path)


if __name__ == '__main__':
    main()