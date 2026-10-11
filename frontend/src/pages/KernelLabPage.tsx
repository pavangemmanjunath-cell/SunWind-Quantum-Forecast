import { useEffect, useState } from 'react';
import { api, type Embedding, type PredictOut, type Series } from '../api/client';
import KernelScatter from '../components/charts/KernelScatter';
import StateBox from '../components/ui/StateBox';
import { useAsync } from '../hooks/useResults';

const COV: (keyof Series)[] = ['wind_ms', 'gust_ms', 'pressure_hpa', 'temp_c', 'ghi_wm2', 'dp3h_hpa'];

export default function KernelLabPage() {
  const [region, setRegion] = useState('C');
  const [alpha, setAlpha] = useState(0.5);
  const [gamma, setGamma] = useState(0.1);
  const [n, setN] = useState(120);
  const [emb, setEmb] = useState<Embedding | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    const id = window.setTimeout(() => {
      setBusy(true);
      api.embedding(region, alpha, gamma, n)
        .then((e) => alive && (setEmb(e), setErr(null)))
        .catch((e: unknown) => alive && setErr(String(e)))
        .finally(() => alive && setBusy(false));
    }, 350);
    return () => {
      alive = false;
      window.clearTimeout(id);
    };
  }, [region, alpha, gamma, n]);

  // forecast probe -------------------------------------------------------------------------
  const series = useAsync(() => api.series(region, '2025-06-01 00:00', 96), [region]);
  const [hour, setHour] = useState(40);
  const [pred, setPred] = useState<PredictOut | null>(null);
  const [probeErr, setProbeErr] = useState<string | null>(null);
  useEffect(() => {
    const s = series.data;
    if (!s) return;
    const lags = Array.from({ length: 24 }, (_, k) => s.y[hour - k]);
    const cov = COV.map((c) => (s[c] as number[])[hour]);
    let alive = true;
    api.predict(region, lags, cov).then((p) => alive && (setPred(p), setProbeErr(null))).catch((e: unknown) => alive && setProbeErr(String(e)));
    return () => {
      alive = false;
    };
  }, [series.data, hour, region]);
  const actual = series.data?.y[hour + 1];

  return (
    <div className="lab-grid">
      <section className="card" aria-label="Kernel separability" style={{ gridColumn: '1 / -1' }}>
        <h2>Regime separability: quantum circuit kernel vs. RBF kernel</h2>
        <p className="caption">
          Calm-versus-stormy windows from simulated 2024, projected on the two leading kernel principal components. The circuit kernel
          is the exact six-qubit state-vector overlap computed live with PennyLane; angle scale α and RBF width γ are yours to move.
        </p>
        <div className="form-grid">
          <label className="field">Region
            <select value={region} onChange={(e) => setRegion(e.target.value)}>
              <option value="A">A · Iberian solar</option><option value="B">B · North-Sea wind</option><option value="C">C · Mixed Texas</option>
            </select>
          </label>
          <label className="field">Angle scale α = {alpha.toFixed(2)}
            <input type="range" min={0.1} max={3} step={0.05} value={alpha} onChange={(e) => setAlpha(Number(e.target.value))} />
          </label>
          <label className="field">RBF γ = {gamma.toFixed(2)}
            <input type="range" min={0.01} max={1} step={0.01} value={gamma} onChange={(e) => setGamma(Number(e.target.value))} />
          </label>
          <label className="field">Samples per class = {n}
            <input type="range" min={40} max={300} step={10} value={n} onChange={(e) => setN(Number(e.target.value))} />
          </label>
        </div>
        {err && <StateBox title="Could not compute the embedding" error>{err}</StateBox>}
        {emb && (
          <div className="lab-grid" style={{ marginTop: 10, opacity: busy ? 0.55 : 1 }}>
            <KernelScatter points={emb.quantum.points} labels={emb.labels} title="Quantum circuit kernel" fdr={emb.quantum.fdr} />
            <KernelScatter points={emb.rbf.points} labels={emb.labels} title="Radial-basis-function kernel" fdr={emb.rbf.fdr} />
          </div>
        )}
        {!emb && !err && <StateBox title="Computing...">Simulating the circuit.</StateBox>}
      </section>

      <section className="card" aria-label="Forecast probe">
        <h2>Probe the AR(24) + VQK forecaster</h2>
        <p className="caption">Pick an hour in the first days of June 2025. The browser fetches the simulated series from this app&apos;s API, assembles the preceding 24 target values and six covariates at that hour, and sends them to the proposed-model prediction endpoint. This endpoint does not run XGBoost or the LSTM.</p>
        <label className="field">Hour index = {hour} ({series.data?.t[hour]?.slice(0, 16) ?? '...'})
          <input type="range" min={24} max={94} value={hour} onChange={(e) => setHour(Number(e.target.value))} />
        </label>
        {probeErr && <StateBox title="Prediction failed" error>{probeErr}</StateBox>}
        {pred && actual !== undefined && (
          <table className="pred-table" style={{ marginTop: 8 }}>
            <tbody>
              <tr><td>AR(24) forecast</td><td className="mono">{pred.ar.toFixed(4)}</td></tr>
              <tr><td>kernel correction</td><td className="mono">{pred.correction >= 0 ? '+' : ''}{pred.correction.toFixed(4)}</td></tr>
              <tr><td><strong>final forecast</strong></td><td className="mono"><strong>{pred.forecast.toFixed(4)}</strong></td></tr>
              <tr><td>realised y(t+1)</td><td className="mono">{actual.toFixed(4)}</td></tr>
            </tbody>
          </table>
        )}
      </section>
    </div>
  );
}
