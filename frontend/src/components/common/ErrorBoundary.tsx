import { Component, ErrorInfo, ReactNode } from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

interface ErrorBoundaryProps {
  children: ReactNode;
  fallbackTitle?: string;
  fallbackMessage?: string;
  onReset?: () => void;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('ErrorBoundary caught an unhandled error:', error, errorInfo);
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: null });
    if (this.props.onReset) {
      this.props.onReset();
    }
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="p-8 my-6 bg-slate-900/80 border border-rose-500/30 rounded-2xl shadow-xl max-w-2xl mx-auto text-center">
          <div className="w-12 h-12 rounded-full bg-rose-500/20 text-rose-400 flex items-center justify-center mx-auto mb-4 border border-rose-500/30">
            <AlertTriangle size={24} />
          </div>
          <h3 className="text-lg font-bold text-white mb-2">
            {this.props.fallbackTitle || 'Component Rendering Error'}
          </h3>
          <p className="text-xs text-slate-400 mb-4 max-w-lg mx-auto">
            {this.props.fallbackMessage ||
              'An unexpected rendering issue occurred in this section. The rest of the application remains fully functional.'}
          </p>

          {this.state.error?.message && (
            <div className="bg-slate-950 p-3 rounded-lg border border-slate-800 font-mono text-[11px] text-rose-300 mb-5 text-left overflow-x-auto max-h-32">
              {this.state.error.message}
            </div>
          )}

          <button
            onClick={this.handleRetry}
            className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold rounded-xl shadow-lg transition-colors inline-flex items-center gap-2"
          >
            <RefreshCw size={14} />
            <span>Try Again</span>
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
