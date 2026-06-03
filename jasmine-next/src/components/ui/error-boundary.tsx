'use client';

import { Component, ReactNode } from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) return this.props.fallback;

      return (
        <div className="p-6 rounded-2xl text-center space-y-4"
          style={{ backgroundColor: 'var(--background)', border: '1px solid var(--border)' }}>
          <div className="w-14 h-14 mx-auto rounded-full flex items-center justify-center"
            style={{ backgroundColor: 'var(--risk-high-bg)' }}>
            <AlertTriangle className="w-7 h-7" style={{ color: 'var(--risk-high)' }} />
          </div>
          <div>
            <h3 className="text-lg font-semibold" style={{ color: 'var(--foreground)' }}>Something went wrong</h3>
            <p className="text-sm mt-1" style={{ color: 'var(--text-muted)' }}>
              {this.state.error?.message || 'An unexpected error occurred.'}
            </p>
          </div>
          <button onClick={this.handleReset}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium text-white transition-all hover:scale-[1.02]"
            style={{ backgroundColor: 'var(--primary)' }}>
            <RefreshCw className="w-4 h-4" /> Try Again
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
