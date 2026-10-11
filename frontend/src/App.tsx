import { useEffect, useState } from 'react';
import { api } from './api/client';
import RegionsPage from './pages/RegionsPage';
import KernelLabPage from './pages/KernelLabPage';
import ExperimentsPage from './pages/ExperimentsPage';
import AboutPage from './pages/AboutPage';
import ErrorBoundary from './components/ui/ErrorBoundary';

const TABS = [
  { id: 'regions', label: 'Regions' },
  { id: 'lab', label: 'Kernel Lab' },
  { id: 'exp', label: 'Experiments' },
  { id: 'about', label: 'About' },
] as const;
type TabId = (typeof TABS)[number]['id'];

function initialTab(): TabId {
  const h = window.location.hash.replace('#', '');
  return TABS.some((t) => t.id === h) ? (h as TabId) : 'regions';
}

export default function App() {
  const [tab, setTab] = useState<TabId>(initialTab);
  const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  useEffect(() => {
    let active = true;
    const checkApi = (): void => {
      api.health()
        .then(() => active && setApiStatus('online'))
        .catch(() => active && setApiStatus('offline'));
    };
    checkApi();
    const timer = window.setInterval(checkApi, 15000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);
  const select = (t: TabId): void => {
    setTab(t);
    window.history.replaceState(null, '', `#${t}`);
  };
  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true" />
          <span>SunWind<span className="brand-dim">-Quantum-Forecast</span></span>
          <span className="sim-pill" title="All data are simulated">SIMULATED DATA</span>
        </div>
        <nav role="tablist" aria-label="Sections" className="tabs">
          {TABS.map((t) => (
            <button key={t.id} role="tab" aria-selected={tab === t.id} className={`tab ${tab === t.id ? 'active' : ''}`} onClick={() => select(t.id)}>
              {t.label}
            </button>
          ))}
        </nav>
      </header>
      <main className="main">
        <section className="research-intro card" aria-labelledby="project-title">
          <div className="intro-copy">
            <p className="eyebrow">RESEARCH DASHBOARD · CLASSICAL SIMULATION</p>
            <h1 id="project-title">Forecasting across changing weather regimes</h1>
            <p className="intro-lede">
              Compare classical baselines with a quantum-inspired kernel residual model across three simulated renewable regions.
              This dashboard makes data provenance, model behavior, and limitations visible as you explore.
            </p>
            <div className="intro-actions">
              <button className="btn primary" onClick={() => select('regions')}>Open regional replay</button>
              <button className="btn primary" onClick={() => select('exp')}>View simulation results</button>
              <button className="btn" onClick={() => select('about')}>Method and dataset</button>
            </div>
          </div>
          <div className="intro-context">
            <div className="api-indicator" data-status={apiStatus} role="status" aria-live="polite">
              <span className="api-dot" aria-hidden="true" />
              API {apiStatus === 'checking' ? 'checking' : apiStatus === 'online' ? 'connected' : 'unavailable'}
            </div>
            <p><strong>Why this research?</strong> Weather variability challenges renewable forecasting; comparisons against strong baselines help test whether extra kernel structure adds useful signal.</p>
            <p><strong>Research status:</strong> ongoing experimental work. These are region-parameterized synthetic series, not live measurements; real-world forecast performance has not been validated.</p>
            <p><strong>Where to look:</strong> generated series come from <code>backend/app/data/synthetic.py</code>; committed experiment outputs are in <code>results/</code>.</p>
          </div>
        </section>
        {/* the 3-D page stays mounted so the WebGL context survives tab changes */}
        <div hidden={tab !== 'regions'}>
          <ErrorBoundary label="3-D view"><RegionsPage visible={tab === 'regions'} /></ErrorBoundary>
        </div>
        {tab === 'lab' && <ErrorBoundary label="Kernel Lab"><KernelLabPage /></ErrorBoundary>}
        {tab === 'exp' && <ErrorBoundary label="Experiments"><ExperimentsPage /></ErrorBoundary>}
        {tab === 'about' && <AboutPage />}
      </main>
    </div>
  );
}
