import { useMemo } from 'react';
import { CartesianGrid, Legend, Line, LineChart, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import type { WeekTrace } from '../../api/client';
import { MODEL_COLOR, MODEL_LABEL } from '../../lib/colors';
import { tooltipStyle } from './common';

interface Props { trace: WeekTrace; models: string[]; tick: number }

/** Contiguous index ranges where the latent regime is stormy. */
function stormRuns(flags: number[]): [number, number][] {
  const runs: [number, number][] = [];
  let start = -1;
  flags.forEach((f, i) => {
    if (f && start < 0) start = i;
    if ((!f || i === flags.length - 1) && start >= 0) {
      runs.push([start, f ? i : i - 1]);
      start = -1;
    }
  });
  return runs;
}

export default function ForecastChart({ trace, models, tick }: Props) {
  const data = useMemo(
    () => trace.y.map((y, i) => {
      const row: Record<string, number> = { i, Actual: y };
      for (const m of models) row[MODEL_LABEL[m] ?? m] = trace.pred[m]?.[i] ?? NaN;
      return row;
    }),
    [trace, models],
  );
  const runs = useMemo(() => stormRuns(trace.stormy), [trace]);
  return (
    <div className="chart-box" style={{ height: 320 }}>
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 6, right: 12, bottom: 4, left: 0 }}>
          <CartesianGrid stroke="var(--grid)" strokeDasharray="3 3" />
          {runs.map(([a, b]) => <ReferenceArea key={a} x1={a} x2={b} fill="#6b7bb8" fillOpacity={0.16} strokeOpacity={0} />)}
          <XAxis dataKey="i" tickFormatter={(v: number) => `${Math.floor(v / 24)}d${v % 24}h`} interval={47} stroke="var(--muted)" fontSize={11} />
          <YAxis domain={[0, 1]} stroke="var(--muted)" fontSize={11} width={34} />
          <Tooltip contentStyle={tooltipStyle} formatter={(v: number) => v.toFixed(3)} labelFormatter={(v: number) => trace.t[v]?.slice(0, 16) ?? ''} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <ReferenceLine x={tick} stroke="var(--accent)" strokeWidth={2} />
          <Line dataKey="Actual" stroke="#e6ebff" strokeWidth={2.4} dot={false} isAnimationActive={false} />
          {models.map((m) => (
            <Line key={m} dataKey={MODEL_LABEL[m] ?? m} stroke={MODEL_COLOR[m] ?? '#999'} strokeWidth={1.4} dot={false} isAnimationActive={false} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
