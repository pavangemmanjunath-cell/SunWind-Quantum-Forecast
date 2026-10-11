import { Bar, BarChart, CartesianGrid, ErrorBar, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import type { ForecastResults, KernelResults, NoiseResults, SimulatorResults } from '../../api/client';
import { ALL_MODELS, MAIN_MODELS, MODEL_COLOR, MODEL_LABEL, REGION_COLOR } from '../../lib/colors';
import { mean, std, tooltipStyle } from './common';

const REGIONS = ['A', 'B', 'C'];

export function NrmseBars({ data }: { data: ForecastResults }) {
  const rows = REGIONS.map((r) => {
    const row: Record<string, number | string> = { region: r };
    for (const m of MAIN_MODELS) {
      const v = data.indomain[r]?.[m]?.nrmse ?? [];
      row[MODEL_LABEL[m] ?? m] = mean(v);
      row[`${m}_sd`] = std(v);
    }
    return row;
  });
  return (
    <div className="chart-box">
      <ResponsiveContainer>
        <BarChart data={rows} margin={{ top: 6, right: 8, bottom: 4, left: 0 }}>
          <CartesianGrid stroke="var(--grid)" strokeDasharray="3 3" />
          <XAxis dataKey="region" stroke="var(--muted)" fontSize={12} />
          <YAxis stroke="var(--muted)" fontSize={11} width={40} />
          <Tooltip contentStyle={tooltipStyle} formatter={(v: number) => v.toFixed(4)} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          {MAIN_MODELS.map((m) => (
            <Bar key={m} dataKey={MODEL_LABEL[m] ?? m} fill={MODEL_COLOR[m]} isAnimationActive={false}>
              <ErrorBar dataKey={`${m}_sd`} stroke="var(--text)" width={3} />
            </Bar>
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function FdrBars({ data }: { data: KernelResults }) {
  const kinds: [keyof KernelResults['regions'][string][number], string, string][] = [
    ['quantum_circuit', 'Quantum circuit kernel', '#f472b6'],
    ['quantum_surrogate', 'Closed-form surrogate (Eq. 5)', '#c084fc'],
    ['rbf_tuned', 'Tuned RBF kernel', '#38bdf8'],
  ];
  const rows = REGIONS.map((r) => {
    const runs = data.regions[r] ?? [];
    const row: Record<string, number | string> = { region: r };
    for (const [k, label] of kinds) {
      const v = runs.map((x) => (x[k] as { fdr_pc1: number }).fdr_pc1);
      row[label] = mean(v);
      row[`${label}_sd`] = std(v);
    }
    return row;
  });
  return (
    <div className="chart-box">
      <ResponsiveContainer>
        <BarChart data={rows} margin={{ top: 6, right: 8, bottom: 4, left: 0 }}>
          <CartesianGrid stroke="var(--grid)" strokeDasharray="3 3" />
          <XAxis dataKey="region" stroke="var(--muted)" fontSize={12} />
          <YAxis stroke="var(--muted)" fontSize={11} width={40} />
          <Tooltip contentStyle={tooltipStyle} formatter={(v: number) => v.toFixed(2)} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          {kinds.map(([, label, c]) => (
            <Bar key={label} dataKey={label} fill={c} isAnimationActive={false}>
              <ErrorBar dataKey={`${label}_sd`} stroke="var(--text)" width={3} />
            </Bar>
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function NoiseLines({ data }: { data: NoiseResults }) {
  const exact = mean(data.exact);
  const shots = Object.entries(data.shots).map(([k, v]) => ({ x: Number(k), fdr: mean(v), sd: std(v), exact }));
  const dep = Object.entries(data.depolarising).map(([k, v]) => ({ x: Number(k) * 100, fdr: mean(v), sd: std(v) }));
  return (
    <div className="exp-grid">
      <div className="chart-box">
        <h3>Finite shots per kernel entry</h3>
        <ResponsiveContainer>
          <LineChart data={shots} margin={{ top: 6, right: 12, bottom: 18, left: 0 }}>
            <CartesianGrid stroke="var(--grid)" strokeDasharray="3 3" />
            <XAxis dataKey="x" scale="log" type="number" domain={['dataMin', 'dataMax']} stroke="var(--muted)" fontSize={11} label={{ value: 'shots', position: 'insideBottom', offset: -8, fill: 'var(--muted)', fontSize: 11 }} />
            <YAxis stroke="var(--muted)" fontSize={11} width={40} />
            <Tooltip contentStyle={tooltipStyle} formatter={(v: number) => v.toFixed(3)} />
            <Line dataKey="fdr" stroke="#f472b6" strokeWidth={2} isAnimationActive={false} name="FDR (PC1)" />
            <Line dataKey="exact" stroke="var(--muted)" strokeDasharray="5 4" dot={false} isAnimationActive={false} name="exact" />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="chart-box">
        <h3>Depolarising channel per qubit per layer</h3>
        <ResponsiveContainer>
          <LineChart data={dep} margin={{ top: 6, right: 12, bottom: 18, left: 0 }}>
            <CartesianGrid stroke="var(--grid)" strokeDasharray="3 3" />
            <XAxis dataKey="x" type="number" domain={[0, 'dataMax']} stroke="var(--muted)" fontSize={11} label={{ value: 'p (%)', position: 'insideBottom', offset: -8, fill: 'var(--muted)', fontSize: 11 }} />
            <YAxis stroke="var(--muted)" fontSize={11} width={40} />
            <Tooltip contentStyle={tooltipStyle} formatter={(v: number) => v.toFixed(3)} />
            <Line dataKey="fdr" stroke="#38bdf8" strokeWidth={2} isAnimationActive={false} name="FDR (PC1)" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export function SimulatorLines({ data }: { data: SimulatorResults }) {
  const diurnal = Array.from({ length: 24 }, (_, h) => {
    const row: Record<string, number> = { h };
    for (const r of REGIONS) row[r] = data.regions[r]?.diurnal[h] ?? 0;
    return row;
  });
  const acf = Array.from({ length: 96 }, (_, l) => {
    const row: Record<string, number> = { lag: l + 1 };
    for (const r of REGIONS) row[r] = data.regions[r]?.acf[l] ?? 0;
    return row;
  });
  const chart = (rows: Record<string, number>[], x: string, label: string) => (
    <div className="chart-box">
      <h3>{label}</h3>
      <ResponsiveContainer>
        <LineChart data={rows} margin={{ top: 6, right: 12, bottom: 4, left: 0 }}>
          <CartesianGrid stroke="var(--grid)" strokeDasharray="3 3" />
          <XAxis dataKey={x} stroke="var(--muted)" fontSize={11} />
          <YAxis stroke="var(--muted)" fontSize={11} width={40} />
          <Tooltip contentStyle={tooltipStyle} formatter={(v: number) => v.toFixed(3)} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          {REGIONS.map((r) => <Line key={r} dataKey={r} stroke={REGION_COLOR[r]} strokeWidth={2} dot={false} isAnimationActive={false} />)}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
  return <div className="exp-grid">{chart(diurnal, 'h', 'Mean diurnal profile of capacity factor')}{chart(acf, 'lag', 'Autocorrelation of capacity factor (hours)')}</div>;
}

export function ImportanceBars({ data, region }: { data: ForecastResults; region: string }) {
  const imp = data.importance[region];
  if (!imp) return null;
  const rows = imp.covariates.map((c, j) => ({ cov: c, 'XGBoost forecast': imp.xgboost[j], 'Proposed forecast': imp.proposed_forecast[j] }));
  return (
    <div className="chart-box">
      <ResponsiveContainer>
        <BarChart data={rows} margin={{ top: 6, right: 8, bottom: 4, left: 0 }}>
          <CartesianGrid stroke="var(--grid)" strokeDasharray="3 3" />
          <XAxis dataKey="cov" stroke="var(--muted)" fontSize={10} />
          <YAxis stroke="var(--muted)" fontSize={11} width={44} />
          <Tooltip contentStyle={tooltipStyle} formatter={(v: number) => v.toFixed(4)} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Bar dataKey="XGBoost forecast" fill={MODEL_COLOR.XGBoost} isAnimationActive={false} />
          <Bar dataKey="Proposed forecast" fill={MODEL_COLOR['Proposed: ARIMA + VQK']} isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export { ALL_MODELS };
