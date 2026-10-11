import { useEffect, useMemo, useRef, useState } from 'react';
import { api, type WeekTrace } from '../api/client';
import ForecastChart from '../components/charts/ForecastChart';
import RegionsScene, { type RegionState } from '../components/scene/RegionsScene';
import StateBox from '../components/ui/StateBox';
import { useAsync } from '../hooks/useResults';
import { MAIN_MODELS, MODEL_COLOR, MODEL_LABEL, REGION_COLOR } from '../lib/colors';

const NAMES: Record<string, string> = { A: 'Iberian solar', B: 'North-Sea wind', C: 'Mixed Texas' };

function Kpi({ l, v }: { l: string; v: string }) {
  return (
    <div className="kpi">
      <div className="v">{v}</div>
      <div className="l">{l}</div>
    </div>
  );
}

export default function RegionsPage({ visible }: { visible: boolean }) {
  const traces = useAsync(async () => {
    const [a, b, c] = await Promise.all([api.week('A'), api.week('B'), api.week('C')]);
    return { A: a, B: b, C: c } as Record<string, WeekTrace>;
  }, []);
  const [region, setRegion] = useState('C');
  const [tick, setTick] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(6);
  const [models, setModels] = useState<string[]>(['ARIMA (AR(24))', 'XGBoost', 'Proposed: ARIMA + VQK']);
  const stateRef = useRef<Record<string, RegionState>>({
    A: { ghi: 0, wind: 0, stormy: false, cf: 0 }, B: { ghi: 0, wind: 0, stormy: false, cf: 0 }, C: { ghi: 0, wind: 0, stormy: false, cf: 0 },
  });
  const n = traces.data?.C.y.length ?? 0;

  useEffect(() => {
    if (!playing || !visible || n === 0) return;
    const id = window.setInterval(() => setTick((t) => (t + 1) % n), 1000 / speed);
    return () => window.clearInterval(id);
  }, [playing, visible, speed, n]);

  useEffect(() => {
    if (!traces.data) return;
    for (const k of ['A', 'B', 'C']) {
      const t = traces.data[k];
      stateRef.current[k] = { ghi: t.ghi[tick] ?? 0, wind: t.wind[tick] ?? 0, stormy: t.stormy[tick] === 1, cf: t.y[tick] ?? 0 };
    }
  }, [traces.data, tick]);

  const trace = traces.data?.[region];
  const cur = trace ? { t: trace.t[tick] ?? '', y: trace.y[tick] ?? 0, ghi: trace.ghi[tick] ?? 0, wind: trace.wind[tick] ?? 0, storm: trace.stormy[tick] === 1 } : null;
  const names = useMemo(() => NAMES, []);

  if (traces.loading) return <StateBox title="Loading simulated week...">Fetching traces from the API.</StateBox>;
  if (traces.error || !trace || !cur) {
    return (
      <StateBox title="Results not available" error>
        {traces.error ?? 'No data.'} Start the backend and run <code>python -m scripts.run_experiments forecast</code>.
      </StateBox>
    );
  }
  return (
    <div className="twin-layout">
      <div className="twin-left">
        <div className="scene-toolbar">
          <div className="seg" role="group" aria-label="Region">
            {['A', 'B', 'C'].map((k) => (
              <button key={k} aria-pressed={region === k} onClick={() => setRegion(k)} style={region === k ? { background: REGION_COLOR[k] } : undefined}>
                {k} · {NAMES[k]}
              </button>
            ))}
          </div>
          <div className="btn-row">
            <button className="btn primary" onClick={() => setPlaying((p) => !p)}>{playing ? 'Pause' : 'Play'}</button>
            <label className="field" style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }}>
              speed
              <input type="range" min={1} max={24} value={speed} onChange={(e) => setSpeed(Number(e.target.value))} style={{ width: 90 }} />
            </label>
          </div>
        </div>
        <div className="scene-wrap">
          <RegionsScene stateRef={stateRef} selected={region} names={names} />
        </div>
        <div className="legend" aria-label="Scene legend">
          <span><i className="swatch" style={{ background: '#8fe3ff' }} />panel brightness = irradiance</span>
          <span><i className="swatch" style={{ background: '#f1f5ff' }} />rotor speed = wind speed (parked at cut-out)</span>
          <span><i className="swatch" style={{ background: '#3a4468' }} />cloud = stormy regime</span>
        </div>
        <section className="card" aria-label="Forecast chart">
          <h2>One-step-ahead forecasts, region {region} ({NAMES[region]})</h2>
          <p className="caption">Fourteen-day simulated test window; displayed forecast traces are precomputed and saved in the experiment results. Shaded bands are simulated stormy spells. Drag the slider to move through time.</p>
          <input type="range" min={0} max={n - 1} value={tick} onChange={(e) => { setPlaying(false); setTick(Number(e.target.value)); }} aria-label="Time" />
          <ForecastChart trace={trace} models={models} tick={tick} />
          <div className="btn-row" role="group" aria-label="Models">
            {MAIN_MODELS.filter((m) => m !== 'Persistence').map((m) => (
              <button key={m} className="btn" aria-pressed={models.includes(m)}
                style={models.includes(m) ? { borderColor: MODEL_COLOR[m], color: MODEL_COLOR[m] } : undefined}
                onClick={() => setModels((cur2) => (cur2.includes(m) ? cur2.filter((x) => x !== m) : [...cur2, m]))}>
                {MODEL_LABEL[m] ?? m}
              </button>
            ))}
          </div>
        </section>
      </div>
      <aside className="twin-right">
        <section className="card" aria-label="Current state">
          <h2>{cur.t.slice(0, 16)}</h2>
          <div className="kpis">
            <Kpi l="capacity factor" v={cur.y.toFixed(3)} />
            <Kpi l="wind (m/s)" v={cur.wind.toFixed(1)} />
            <Kpi l="irradiance (W/m²)" v={cur.ghi.toFixed(0)} />
            <Kpi l="weather regime" v={cur.storm ? 'stormy' : 'calm'} />
          </div>
        </section>
        <section className="card" aria-label="Forecast at this hour">
          <h2>Forecast for this hour</h2>
          <table className="pred-table">
            <thead><tr><th>Model</th><th>ŷ</th><th>error</th></tr></thead>
            <tbody>
              {MAIN_MODELS.map((m) => {
                const p = trace.pred[m]?.[tick] ?? NaN;
                return (
                  <tr key={m}>
                    <td><i className="swatch" style={{ background: MODEL_COLOR[m] }} />{MODEL_LABEL[m] ?? m}</td>
                    <td className="mono">{p.toFixed(3)}</td>
                    <td className="mono">{(p - cur.y >= 0 ? '+' : '') + (p - cur.y).toFixed(3)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </section>
        <section className="card" aria-label="Notes">
          <h3>About this view</h3>
          <p className="caption">The three scenes show the same simulated hour for all regions. The data come from the region-parameterized synthetic generator, not measured data or a live weather API.</p>
        </section>
      </aside>
    </div>
  );
}
