export interface Region { key: string; name: string }
export interface WeekTrace {
  t: string[];
  y: number[];
  stormy: number[];
  ghi: number[];
  wind: number[];
  pred: Record<string, number[]>;
}
type MetricLists = Record<string, number[]>;
export interface Importance {
  covariates: string[];
  proposed_forecast: number[];
  proposed_correction: number[];
  xgboost: number[];
  lambda: number;
}
export interface ForecastResults {
  seeds: number;
  indomain: Record<string, Record<string, MetricLists>>;
  transfer: Record<string, Record<string, { zero_shot: number[]; fine_tune: number[] }>>;
  week_traces: Record<string, WeekTrace>;
  importance: Record<string, Importance>;
}
export interface KernelScore { fdr_pc1: number; fdr_pc2: number; fdr_best_of_2: number; svm_acc: number | null }
export interface KernelRun {
  alpha: number;
  gamma: number;
  quantum_circuit: KernelScore;
  quantum_surrogate: KernelScore;
  rbf_tuned: KernelScore;
  quantum_circuit_alpha1: KernelScore;
}
export interface KernelResults { seeds: number; regions: Record<string, KernelRun[]> }
export interface NoiseResults { exact: number[]; shots: Record<string, number[]>; depolarising: Record<string, number[]> }
export interface SimRegion {
  name: string; hours: number; mean_cf: number; std_cf: number; p95_ramp: number; acf1: number; acf24: number; acf72: number;
  storm_fraction: number; mean_storm_hours: number; n_storms: number; mean_wind_ms: number; mean_ghi: number;
  zero_fraction: number; diurnal: number[]; acf: number[]; hist: number[];
}
export interface SimulatorResults {
  software: Record<string, string>;
  regions: Record<string, SimRegion>;
  surrogate_vs_circuit: Record<string, number>;
  n_circuit_parameters: number;
}
export interface ResultMap { forecast: ForecastResults; kernel: KernelResults; noise: NoiseResults; simulator: SimulatorResults }

export interface Embedding {
  labels: number[];
  quantum: { points: number[][]; fdr: number };
  rbf: { points: number[][]; fdr: number };
}
export interface Series {
  region: string; t: string[]; y: number[]; wind_ms: number[]; gust_ms: number[]; pressure_hpa: number[];
  temp_c: number[]; ghi_wm2: number[]; dp3h_hpa: number[]; stormy: number[];
}
export interface PredictOut { region: string; ar: number; correction: number; forecast: number }

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
  /** Backend not reachable, or the result file has not been generated yet. */
  get unavailable(): boolean {
    return this.status === 0 || this.status === 404 || this.status === 503;
  }
}

const BASE: string = (import.meta.env.VITE_API_URL as string | undefined) ?? '';

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, init);
  } catch (e) {
    throw new ApiError(0, `Cannot reach the API (${String(e)})`);
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = ((await res.json()) as { detail?: string }).detail ?? detail;
    } catch { /* not JSON */ }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

export const api = {
  health: () => req<{ status: string; qubits: number; layers: number; parameters: number }>('/api/health'),
  regions: () => req<Region[]>('/api/regions'),
  result: <K extends keyof ResultMap>(name: K) => req<ResultMap[K]>(`/api/results/${name}`),
  week: (region: string) => req<WeekTrace>(`/api/forecast/week?region=${region}`),
  series: (region: string, start: string, hours: number) =>
    req<Series>(`/api/series?region=${region}&start=${encodeURIComponent(start)}&hours=${hours}`),
  embedding: (region: string, alpha: number, gamma: number, n: number) =>
    req<Embedding>(`/api/kernel/embedding?region=${region}&alpha=${alpha}&gamma=${gamma}&n=${n}`),
  predict: (region: string, lags: number[], covariates: number[]) =>
    req<PredictOut>(`/api/predict/${region}`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ lags, covariates }),
    }),
  explain: (region: string) => req<{ text: string }>(`/api/explain?region=${region}`),
};
