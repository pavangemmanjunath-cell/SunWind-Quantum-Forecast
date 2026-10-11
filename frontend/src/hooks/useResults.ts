import { useEffect, useState } from 'react';
import { api, ApiError, type ResultMap } from '../api/client';

export interface LoadState<T> {
  data: T | null;
  loading: boolean;
  /** true when the backend answered 404/503 or is unreachable */
  missing: boolean;
  error: string | null;
}

export function useAsync<T>(load: () => Promise<T>, deps: unknown[]): LoadState<T> {
  const [state, setState] = useState<LoadState<T>>({ data: null, loading: true, missing: false, error: null });
  useEffect(() => {
    let alive = true;
    setState((s) => ({ ...s, loading: true }));
    load()
      .then((data) => alive && setState({ data, loading: false, missing: false, error: null }))
      .catch((e: unknown) => {
        if (!alive) return;
        const err = e instanceof ApiError ? e : new ApiError(0, String(e));
        setState({ data: null, loading: false, missing: err.unavailable, error: err.message });
      });
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return state;
}

export function useResult<K extends keyof ResultMap>(name: K): LoadState<ResultMap[K]> {
  return useAsync(() => api.result(name), [name]);
}
