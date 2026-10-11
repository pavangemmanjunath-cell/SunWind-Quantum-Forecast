import { Component, type ErrorInfo, type ReactNode } from 'react';

interface Props { children: ReactNode; label?: string }
interface State { error: Error | null }

export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };
  static getDerivedStateFromError(error: Error): State {
    return { error };
  }
  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error('ErrorBoundary', error, info.componentStack);
  }
  render(): ReactNode {
    if (this.state.error) {
      return (
        <div className="state-box err" role="alert" style={{ margin: 16 }}>
          <strong>{this.props.label ?? 'This view'} failed to render</strong>
          <span>{this.state.error.message}</span>
          <div style={{ marginTop: 10 }}>
            <button className="btn" onClick={() => this.setState({ error: null })}>Retry</button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
