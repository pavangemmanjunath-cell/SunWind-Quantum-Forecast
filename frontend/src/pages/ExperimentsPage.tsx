import { useEffect, useState } from 'react';
import { api, type ForecastResults, type KernelResults, type NoiseResults, type SimulatorResults } from '../api/client';
import { ChartCard, mean, std } from '../components/charts/common';
import { FdrBars, ImportanceBars, NoiseLines, NrmseBars, SimulatorLines } from '../components/charts/ExperimentCharts';
import { ALL_MODELS, MODEL_LABEL } from '../lib/colors';
import { useResult } from '../hooks/useResults';

const pm = (v: number[], nd = 4): string => `${mean(v).toFixed(nd)} ± ${std(v).toFixed(nd)}`;
const TRANSFER = ['ARIMA (AR(24))', 'XGBoost', 'LSTM (2 layers)', 'Proposed: ARIMA + VQK'];

const pmOrDash = (values: number[] | undefined): string => values?.length ? pm(values) : '—';

function IndomainTable({ data, region }: { data: ForecastResults; region: string }) {
  const rows = data.indomain[region] ?? {};
  return (
    <table className="pred-table">
      <thead><tr><th>Model</th><th>NRMSE</th><th>CRPS (MAE)</th><th>Regret</th><th>Latency (ms)</th></tr></thead>
      <tbody>
        {ALL_MODELS.filter((m) => rows[m]).map((m) => (
          <tr key={m}>
            <td>{MODEL_LABEL[m] ?? m}</td>
            <td className="mono">{pm(rows[m].nrmse)}</td>
            <td className="mono">{pm(rows[m].crps)}</td>
            <td className="mono">{pm(rows[m].regret)}</td>
            <td className="mono">{rows[m].latency_ms ? rows[m].latency_ms[0].toFixed(3) : '-'}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function TransferTable({ data }: { data: ForecastResults }) {
  const pairs = Object.keys(data.transfer);
  const diag = pairs.filter((p) => p[0] === p[3]);
  const off = pairs.filter((p) => p[0] !== p[3]);
  const per = (m: string, ps: string[], key: 'zero_shot' | 'fine_tune'): number[] => {
    const seeds = data.seeds;
    return Array.from({ length: seeds }, (_, s) => mean(ps.map((p) => data.transfer[p][m]?.[key]?.[s]).filter((x): x is number => x !== undefined)));
  };
  return (
    <table className="pred-table">
      <thead><tr><th>Model</th><th>In-domain (3 pairs)</th><th>Zero-shot (6 pairs)</th><th>Fine-tune (6 pairs)</th></tr></thead>
      <tbody>
        {TRANSFER.map((m) => (
          <tr key={m}>
            <td>{MODEL_LABEL[m] ?? m}</td>
            <td className="mono">{pm(per(m, diag, 'zero_shot'))}</td>
            <td className="mono">{pm(per(m, off, 'zero_shot'))}</td>
            <td className="mono">{pm(per(m, off, 'fine_tune'))}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function TransferPairsTable({ data }: { data: ForecastResults }) {
  return (
    <div className="table-scroll">
      <table className="pred-table pred-table--wide">
        <thead><tr><th>Source → target</th><th>Model</th><th>Zero-shot NRMSE</th><th>Fine-tuned NRMSE</th></tr></thead>
        <tbody>
          {Object.keys(data.transfer).sort().flatMap((pair) => TRANSFER.map((model) => {
            const scores = data.transfer[pair][model];
            return (
              <tr key={`${pair}-${model}`}>
                <td>{pair[0]} → {pair[3]} ({pair[0] === pair[3] ? 'in-domain' : 'cross-region'})</td>
                <td>{MODEL_LABEL[model] ?? model}</td>
                <td className="mono">{pmOrDash(scores?.zero_shot)}</td>
                <td className="mono">{pmOrDash(scores?.fine_tune)}</td>
              </tr>
            );
          }))}
        </tbody>
      </table>
    </div>
  );
}

function KernelTable({ data }: { data: KernelResults }) {
  return (
    <table className="pred-table">
      <thead><tr><th>Region</th><th>Circuit FDR</th><th>Surrogate FDR</th><th>RBF FDR</th><th>Circuit SVM acc.</th><th>RBF SVM acc.</th></tr></thead>
      <tbody>
        {Object.entries(data.regions).map(([r, runs]) => (
          <tr key={r}>
            <td>{r}</td>
            <td className="mono">{pm(runs.map((x) => x.quantum_circuit.fdr_pc1), 2)}</td>
            <td className="mono">{pm(runs.map((x) => x.quantum_surrogate.fdr_pc1), 2)}</td>
            <td className="mono">{pm(runs.map((x) => x.rbf_tuned.fdr_pc1), 2)}</td>
            <td className="mono">{pm(runs.map((x) => x.quantum_circuit.svm_acc ?? NaN), 3)}</td>
            <td className="mono">{pm(runs.map((x) => x.rbf_tuned.svm_acc ?? NaN), 3)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function KernelSummary({ kernel, simulator }: { kernel: KernelResults; simulator: SimulatorResults }) {
  const runs = Object.values(kernel.regions).flat();
  const circuitWins = runs.filter((run) => run.quantum_circuit.fdr_pc1 > run.rbf_tuned.fdr_pc1).length;
  const correlations = Object.values(simulator.surrogate_vs_circuit);
  return (
    <div style={{ gridColumn: '1 / -1' }}>
      <section className="card" aria-label="Kernel study summary">
        <h2>Kernel study summary</h2>
        <p className="caption">
          The exact circuit kernel has a higher leading-component Fisher ratio than tuned RBF in {circuitWins} of {runs.length} region–seed trials.
          The closed-form surrogate/circuit kernel-entry correlation ranges from {Math.min(...correlations).toFixed(2)} to {Math.max(...correlations).toFixed(2)}
          {' '}across the tested angle scales. The feature map has {simulator.n_circuit_parameters} fixed angles; no quantum advantage is claimed.
        </p>
      </section>
    </div>
  );
}

function NoiseTable({ data }: { data: NoiseResults }) {
  const rows = [
    { setting: 'Exact circuit kernel', values: data.exact },
    ...Object.entries(data.shots).map(([shots, values]) => ({ setting: `Shot noise · ${Number(shots).toLocaleString()} shots`, values })),
    ...Object.entries(data.depolarising).map(([p, values]) => ({ setting: `Depolarising · p = ${p}`, values })),
  ];
  return (
    <table className="pred-table">
      <thead><tr><th>Noise setting</th><th>FDR (PC1), mean ± s.d.</th></tr></thead>
      <tbody>{rows.map((row) => <tr key={row.setting}><td>{row.setting}</td><td className="mono">{pm(row.values, 3)}</td></tr>)}</tbody>
    </table>
  );
}

function SimulatorTable({ data }: { data: SimulatorResults }) {
  return (
    <table className="pred-table">
      <thead><tr><th>Region</th><th>Hours</th><th>Mean CF</th><th>s.d.</th><th>ACF(1)</th><th>ACF(24)</th><th>Stormy hours (%)</th><th>Mean storm spell (h)</th></tr></thead>
      <tbody>
        {Object.entries(data.regions).map(([key, region]) => (
          <tr key={key}>
            <td>{key} · {region.name}</td>
            <td className="mono">{region.hours.toLocaleString()}</td>
            <td className="mono">{region.mean_cf.toFixed(3)}</td>
            <td className="mono">{region.std_cf.toFixed(3)}</td>
            <td className="mono">{region.acf1.toFixed(3)}</td>
            <td className="mono">{region.acf24.toFixed(3)}</td>
            <td className="mono">{(region.storm_fraction * 100).toFixed(1)}</td>
            <td className="mono">{region.mean_storm_hours.toFixed(0)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function ImportanceTable({ data, region }: { data: ForecastResults; region: string }) {
  const importance = data.importance[region];
  if (!importance) return null;
  return (
    <table className="pred-table">
      <thead><tr><th>Covariate</th><th>XGBoost forecast</th><th>AR + VQK forecast</th></tr></thead>
      <tbody>
        {importance.covariates.map((covariate, index) => (
          <tr key={covariate}>
            <td>{covariate}</td>
            <td className="mono">{importance.xgboost[index].toFixed(4)}</td>
            <td className="mono">{importance.proposed_forecast[index].toFixed(4)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default function ExperimentsPage() {
  const fc = useResult('forecast');
  const kn = useResult('kernel');
  const nz = useResult('noise');
  const sm = useResult('simulator');
  const [region, setRegion] = useState('C');
  const [text, setText] = useState<string | null>(null);
  useEffect(() => {
    api.explain(region).then((r) => setText(r.text)).catch(() => setText(null));
  }, [region]);
  const common = (s: { loading: boolean; missing: boolean; error: string | null }) => ({ loading: s.loading, missing: s.missing, error: s.error });

  return (
    <div className="exp-grid">
      <div className="banner" style={{ gridColumn: '1 / -1' }} role="note">
        Experimental, ongoing research. These benchmark numbers come from the project&apos;s synthetic generator, not external weather or grid observations; real-world accuracy has not been established.
      </div>
      <div style={{ gridColumn: '1 / -1' }} className="btn-row">
        <strong>Region for tables</strong>
        <div className="seg" role="group" aria-label="Region">
          {['A', 'B', 'C'].map((r) => <button key={r} aria-pressed={region === r} onClick={() => setRegion(r)}>{r}</button>)}
        </div>
      </div>

      <ChartCard title="In-domain forecasting (mean ± s.d. over seeds)" caption="One-step-ahead, fourteen-day simulated test window. CRPS of a point forecast equals its MAE." {...common(fc)} hasData={!!fc.data}>
        {fc.data && <IndomainTable data={fc.data} region={region} />}
      </ChartCard>
      <ChartCard title="NRMSE by region" caption="Lower is better. Error bars are one s.d. over seeds." {...common(fc)} hasData={!!fc.data}>
        {fc.data && <NrmseBars data={fc.data} />}
      </ChartCard>
      <ChartCard title="Cross-region transfer (NRMSE)" caption="Zero-shot applies a source-region model unchanged; fine-tune adapts it on 28 days of target data." {...common(fc)} hasData={!!fc.data}>
        {fc.data && <TransferTable data={fc.data} />}
      </ChartCard>
      <div style={{ gridColumn: '1 / -1' }}>
        <ChartCard title="Cross-region transfer by source and target" caption="All nine source → target pairs. Entries are mean ± s.d. across seeds; fine-tuning uses 28 days of target data and is not applicable to in-domain pairs." {...common(fc)} hasData={!!fc.data}>
          {fc.data && <TransferPairsTable data={fc.data} />}
        </ChartCard>
      </div>
      <ChartCard title="Permutation importance of the covariates" caption={`Eq. (7): mean L1 change of the forecast when one covariate is shuffled, divided by σ_y (region ${region}).`} {...common(fc)} hasData={!!fc.data}>
        {fc.data && <ImportanceBars data={fc.data} region={region} />}
      </ChartCard>
      <ChartCard title={`Permutation-importance values · region ${region}`} caption="All six covariates, first experiment seed; values use the same σ_y-normalised L1 measure as the paper table." {...common(fc)} hasData={!!fc.data}>
        {fc.data && <ImportanceTable data={fc.data} region={region} />}
      </ChartCard>
      <ChartCard title="Calm-versus-stormy separability" caption="Fisher discriminant ratio on the leading kernel principal component (higher = better separated). Each kernel's single hyper-parameter is tuned by cross-validation." {...common(kn)} hasData={!!kn.data}>
        {kn.data && <FdrBars data={kn.data} />}
      </ChartCard>
      <ChartCard title="Kernel separability, details" caption="Mean ± s.d. over seeds; the SVM accuracy is the 4-fold cross-validated accuracy on a precomputed kernel." {...common(kn)} hasData={!!kn.data}>
        {kn.data && <KernelTable data={kn.data} />}
      </ChartCard>
      {kn.data && sm.data && <KernelSummary kernel={kn.data} simulator={sm.data} />}
      <div style={{ gridColumn: '1 / -1' }}>
        <ChartCard title="Hardware-noise emulation of the circuit kernel" caption="Region C, 300 windows, three seeds. Left: each kernel entry estimated from a finite number of shots; right: depolarising channel after every layer." {...common(nz)} hasData={!!nz.data}>
          {nz.data && <NoiseLines data={nz.data} />}
        </ChartCard>
      </div>
      <div style={{ gridColumn: '1 / -1' }}>
        <ChartCard title="Noise-study values" caption="Fisher-ratio mean ± s.d. across the three seeds for every exact, finite-shot, and depolarising-noise condition." {...common(nz)} hasData={!!nz.data}>
          {nz.data && <NoiseTable data={nz.data} />}
        </ChartCard>
      </div>
      <div style={{ gridColumn: '1 / -1' }}>
        <ChartCard title="Simulator checks" caption="Diurnal profile and autocorrelation of the three simulated regions (eight years, hourly)." {...common(sm)} hasData={!!sm.data}>
          {sm.data && <SimulatorLines data={sm.data} />}
        </ChartCard>
      </div>
      <div style={{ gridColumn: '1 / -1' }}>
        <ChartCard title="Simulator summary values" caption="Core distribution, autocorrelation, and weather-regime statistics reported for the three simulated regions." {...common(sm)} hasData={!!sm.data}>
          {sm.data && <SimulatorTable data={sm.data} />}
        </ChartCard>
      </div>
      <section className="card" style={{ gridColumn: '1 / -1' }} aria-label="Interpretation layer">
        <h2>Interpretation layer (stage four)</h2>
        <p className="caption">Template-based, deterministic text generated from the saved experiment result files (not measured observations).</p>
        <p>{text ?? 'Not available until the forecast and kernel experiments have been run.'}</p>
      </section>
    </div>
  );
}
