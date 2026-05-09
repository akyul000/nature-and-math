/**
 * SimViewer — generic loader for pre-computed simulation data.
 *
 * Usage:
 *   SimViewer.load({
 *     dataDir:     './data/',
 *     plotDiv:     'plot',
 *     statusDiv:   'sim-status',   // optional
 *     params: [
 *       { name: 'alpha', label: 'Thermal diffusivity α', min: 0.01, max: 1.0, step: 0.01, default: 0.1 }
 *     ],
 *     filePattern: (p) => `heat_alpha_${p.alpha.toFixed(2)}.json`,
 *     plotType:    'heatmap',      // 'heatmap' | 'line' | 'surface'
 *     // optional: called with (data, frameIdx) to build custom Plotly traces
 *     buildTraces: null,
 *   });
 */
const SimViewer = (function () {
  'use strict';

  let _cache = {};

  async function fetchData(url) {
    if (_cache[url]) return _cache[url];
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`Failed to load ${url}: ${resp.status}`);
    const data = await resp.json();
    _cache[url] = data;
    return data;
  }

  function setStatus(div, msg) {
    if (div) div.textContent = msg;
  }

  function buildDefaultTraces(data, frameIdx, plotType) {
    const frame = data.frames[frameIdx];
    const u = frame.u;
    const x = data.x;

    if (plotType === 'heatmap') {
      return [{
        type: 'heatmap',
        z: [u],
        x: x,
        colorscale: 'Inferno',
        zmin: 0, zmax: 1,
        showscale: true,
      }];
    }

    if (plotType === 'line') {
      return [{
        type: 'scatter',
        mode: 'lines',
        x: x,
        y: u,
        line: { color: '#d4a017', width: 2 },
      }];
    }

    // surface: 2D grid expected
    if (plotType === 'surface') {
      return [{
        type: 'surface',
        z: u,
        colorscale: 'Inferno',
        showscale: false,
      }];
    }

    return [];
  }

  function buildLayout(data, plotType) {
    const base = {
      paper_bgcolor: '#161b22',
      plot_bgcolor:  '#0d1117',
      font: { color: '#e6edf3', family: 'monospace' },
      margin: { t: 30, b: 40, l: 50, r: 20 },
      xaxis: { title: 'x', gridcolor: '#30363d', zerolinecolor: '#30363d' },
      yaxis: { title: plotType === 'heatmap' ? '' : 'u(x, t)', gridcolor: '#30363d', zerolinecolor: '#30363d' },
    };
    if (plotType === 'heatmap') {
      base.yaxis.showticklabels = false;
      base.height = 160;
    } else {
      base.height = 360;
      if (data.frames.length > 0) {
        const allVals = data.frames.flatMap(f => f.u);
        base.yaxis.range = [Math.min(...allVals) * 1.05, Math.max(...allVals) * 1.05];
      }
    }
    return base;
  }

  function load(opts) {
    const {
      dataDir,
      plotDiv: plotDivId,
      statusDiv: statusDivId,
      params,
      filePattern,
      plotType = 'line',
      buildTraces = null,
    } = opts;

    const plotEl   = document.getElementById(plotDivId);
    const statusEl = statusDivId ? document.getElementById(statusDivId) : null;
    if (!plotEl) { console.error('SimViewer: plotDiv not found:', plotDivId); return; }

    /* current parameter values */
    const values = {};
    params.forEach(p => { values[p.name] = p.default !== undefined ? p.default : p.min; });

    /* time scrubber state */
    let currentData  = null;
    let frameIdx     = 0;
    let playing      = false;
    let animHandle   = null;

    /* ── render one frame ── */
    function renderFrame(data, idx) {
      const traces  = buildTraces ? buildTraces(data, idx) : buildDefaultTraces(data, idx, plotType);
      const layout  = buildLayout(data, plotType);
      const t       = data.t ? data.t[data.frames[idx].t_idx ?? idx] : idx;
      layout.title  = { text: `t = ${typeof t === 'number' ? t.toFixed(4) : t}`, font: { size: 12 } };
      Plotly.react(plotEl, traces, layout, { responsive: true, displayModeBar: false });
    }

    /* ── load data for current param values and render ── */
    async function reload() {
      const filename = filePattern(values);
      const url = dataDir.replace(/\/?$/, '/') + filename;
      setStatus(statusEl, `Loading ${filename}…`);
      try {
        const data = await fetchData(url);
        currentData = data;
        frameIdx = 0;
        renderFrame(data, frameIdx);
        setStatus(statusEl, `Loaded ${data.frames.length} frames  ·  nx=${data.params.nx}  ·  Use slider or ← → to scrub`);
        injectTimeControls(data);
      } catch (err) {
        setStatus(statusEl, `⚠ ${err.message}  (run simulations/generate_all.py to generate data)`);
        Plotly.purge(plotEl);
      }
    }

    /* ── build parameter sliders ── */
    const controlsEl = document.getElementById(opts.controlsDiv || 'controls');
    if (controlsEl) {
      params.forEach(p => {
        const group = document.createElement('div');
        group.className = 'slider-group';
        group.innerHTML = `
          <label for="slider-${p.name}">${p.label}</label>
          <div class="slider-row">
            <input type="range" id="slider-${p.name}"
              min="${p.min}" max="${p.max}" step="${p.step}"
              value="${values[p.name]}" />
            <span class="val-display" id="val-${p.name}">${values[p.name]}</span>
          </div>`;
        controlsEl.appendChild(group);

        const input = group.querySelector('input');
        const display = group.querySelector('.val-display');
        input.addEventListener('input', () => {
          values[p.name] = parseFloat(input.value);
          display.textContent = input.value;
          reload();
        });
      });
    }

    /* ── time scrubber (injected after data loads) ── */
    function injectTimeControls(data) {
      const existing = document.getElementById('time-controls');
      if (existing) existing.remove();
      if (data.frames.length <= 1) return;

      const el = document.createElement('div');
      el.id = 'time-controls';
      el.className = 'slider-group';
      el.style.padding = '0.5rem 1.25rem';
      el.innerHTML = `
        <label for="time-slider">Time step</label>
        <div class="slider-row">
          <button id="play-btn" style="font-size:0.8rem;padding:0.1rem 0.6rem;border-radius:4px;border:1px solid #30363d;background:#1c2333;color:#e6edf3;cursor:pointer;">▶ Play</button>
          <input type="range" id="time-slider" min="0" max="${data.frames.length - 1}" step="1" value="0" style="flex:1"/>
          <span class="val-display" id="val-time">0</span>
        </div>`;

      const simSection = plotEl.closest('.sim-section');
      if (simSection) simSection.insertBefore(el, plotEl);

      const slider  = el.querySelector('#time-slider');
      const valDisp = el.querySelector('#val-time');
      const playBtn = el.querySelector('#play-btn');

      slider.addEventListener('input', () => {
        frameIdx = parseInt(slider.value);
        valDisp.textContent = frameIdx;
        renderFrame(data, frameIdx);
      });

      playBtn.addEventListener('click', () => {
        if (playing) {
          playing = false;
          clearInterval(animHandle);
          playBtn.textContent = '▶ Play';
        } else {
          playing = true;
          playBtn.textContent = '⏸ Pause';
          animHandle = setInterval(() => {
            frameIdx = (frameIdx + 1) % data.frames.length;
            slider.value = frameIdx;
            valDisp.textContent = frameIdx;
            renderFrame(data, frameIdx);
          }, 80);
        }
      });
    }

    /* keyboard scrub (← →) */
    document.addEventListener('keydown', (e) => {
      if (!currentData) return;
      if (e.target.tagName === 'INPUT') return;
      if (e.key === 'ArrowLeft'  && e.shiftKey) { frameIdx = Math.max(0, frameIdx - 1); renderFrame(currentData, frameIdx); }
      if (e.key === 'ArrowRight' && e.shiftKey) { frameIdx = Math.min(currentData.frames.length - 1, frameIdx + 1); renderFrame(currentData, frameIdx); }
    });

    reload();
  }

  return { load };
})();
