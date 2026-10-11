import { Color } from 'three';

export const REGION_COLOR: Record<string, string> = { A: '#fbbf24', B: '#38bdf8', C: '#a78bfa' };
export const MODEL_COLOR: Record<string, string> = {
  'Persistence': '#93a0cc',
  'ARIMA (AR(24))': '#38bdf8',
  'XGBoost': '#34d399',
  'LSTM (2 layers)': '#fb923c',
  'Proposed: ARIMA + VQK': '#f472b6',
  'ARIMA + VQK (circuit kernel)': '#c084fc',
  'ARIMA + RBF kernel': '#fde047',
};
export const MODEL_LABEL: Record<string, string> = {
  'ARIMA (AR(24))': 'AR(24)',
  'Proposed: ARIMA + VQK': 'Proposed: AR(24) + VQK',
  'ARIMA + VQK (circuit kernel)': 'AR(24) + VQK (circuit kernel)',
  'ARIMA + RBF kernel': 'AR(24) + RBF kernel',
};
export const MAIN_MODELS = ['Persistence', 'ARIMA (AR(24))', 'XGBoost', 'LSTM (2 layers)', 'Proposed: ARIMA + VQK'];
export const ALL_MODELS = [...MAIN_MODELS, 'ARIMA + VQK (circuit kernel)', 'ARIMA + RBF kernel'];

const clamp01 = (x: number): number => Math.min(1, Math.max(0, x));
/** dark blue (no irradiance) -> bright cyan-white (full sun) */
export const panelColor = (ghi: number): Color => new Color('#0b2a5b').lerp(new Color('#8fe3ff'), clamp01(ghi / 900));
