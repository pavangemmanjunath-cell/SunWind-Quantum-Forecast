import { CartesianGrid, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from 'recharts';
import { tooltipStyle } from './common';

interface Props { points: number[][]; labels: number[]; title: string; fdr: number }

export default function KernelScatter({ points, labels, title, fdr }: Props) {
  const calm = points.filter((_, i) => labels[i] === 0).map(([x, y]) => ({ x, y }));
  const storm = points.filter((_, i) => labels[i] === 1).map(([x, y]) => ({ x, y }));
  return (
    <div>
      <div className="btn-row" style={{ justifyContent: 'space-between' }}>
        <h3>{title}</h3>
        <span className="tag" style={{ background: 'var(--accent)' }}>FDR(PC1) = {fdr.toFixed(2)}</span>
      </div>
      <div className="chart-box" style={{ height: 300 }}>
        <ResponsiveContainer>
          <ScatterChart margin={{ top: 8, right: 12, bottom: 18, left: 0 }}>
            <CartesianGrid stroke="var(--grid)" strokeDasharray="3 3" />
            <XAxis type="number" dataKey="x" name="PC1" stroke="var(--muted)" fontSize={11} tickFormatter={(v: number) => v.toFixed(1)} label={{ value: 'kernel PC 1', position: 'insideBottom', offset: -8, fill: 'var(--muted)', fontSize: 11 }} />
            <YAxis type="number" dataKey="y" name="PC2" stroke="var(--muted)" fontSize={11} width={40} tickFormatter={(v: number) => v.toFixed(1)} />
            <ZAxis range={[26, 26]} />
            <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={tooltipStyle} formatter={(v: number) => v.toFixed(3)} />
            <Scatter name="calm" data={calm} fill="#38bdf8" fillOpacity={0.6} isAnimationActive={false} />
            <Scatter name="stormy" data={storm} fill="#f472b6" fillOpacity={0.6} isAnimationActive={false} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
