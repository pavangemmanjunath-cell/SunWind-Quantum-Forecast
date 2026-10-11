import type { ReactNode } from 'react';
import StateBox from '../ui/StateBox';

export const tooltipStyle = { background: 'var(--panel)', border: '1px solid var(--border)', color: 'var(--text)' } as const;
export const SERIES = ['var(--series-1)', 'var(--series-2)', 'var(--series-3)', 'var(--series-4)'] as const;

export const mean = (a: number[]): number => (a.length ? a.reduce((s, v) => s + v, 0) / a.length : 0);
export const std = (a: number[]): number => {
  if (a.length < 2) return 0;
  const m = mean(a);
  return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / (a.length - 1));
};
/** Accept a number, number[] (per seed) and return the list. */
export const asList = (v: number | number[]): number[] => (Array.isArray(v) ? v : [v]);

interface Props {
  title: string;
  caption: string;
  loading: boolean;
  missing: boolean;
  error: string | null;
  hasData: boolean;
  children: ReactNode;
}
export function ChartCard({ title, caption, loading, missing, error, hasData, children }: Props) {
  return (
    <section className="card" aria-label={title}>
      <h2>{title}</h2>
      <p className="caption">{caption}</p>
      {loading ? (
        <StateBox title="Loading...">Fetching results.</StateBox>
      ) : missing ? (
        <StateBox title="Results not generated yet">Run <code>python -m scripts.run_experiments</code> in the backend.</StateBox>
      ) : error ? (
        <StateBox title="Could not load results" error>{error}</StateBox>
      ) : !hasData ? (
        <StateBox title="No data">This result file has no entries for this chart.</StateBox>
      ) : (
        children
      )}
    </section>
  );
}
