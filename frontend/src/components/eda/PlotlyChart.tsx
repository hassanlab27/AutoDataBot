import React, { useEffect, useRef } from 'react';
import Plotly from 'plotly.js-dist-min';

interface PlotlyChartProps {
  data: any[];
  layout?: Record<string, any>;
  config?: Record<string, any>;
  style?: React.CSSProperties;
  className?: string;
  chartId?: string;
}

export const PlotlyChart: React.FC<PlotlyChartProps> = ({
  data,
  layout = {},
  config = {},
  style = { width: '100%', height: '400px' },
  className = '',
  chartId
}) => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const mergedLayout = {
      autosize: true,
      paper_bgcolor: 'rgba(17, 24, 39, 0)',
      plot_bgcolor: 'rgba(17, 24, 39, 0.4)',
      font: {
        color: '#94a3b8',
        family: 'Inter, system-ui, sans-serif',
        size: 11
      },
      margin: { l: 50, r: 25, t: 45, b: 45 },
      ...layout
    };

    const mergedConfig = {
      responsive: true,
      displayModeBar: true,
      displaylogo: false,
      modeBarButtonsToRemove: ['lasso2d', 'select2d'] as any,
      ...config
    };

    Plotly.react(el, data, mergedLayout, mergedConfig as any);

    const handleResize = () => {
      if (el) {
        Plotly.Plots.resize(el);
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (el) {
        Plotly.purge(el);
      }
    };
  }, [data, layout, config]);

  return (
    <div
      ref={containerRef}
      id={chartId}
      className={`plotly-chart-container ${className}`}
      style={{ minHeight: '350px', ...style }}
    />
  );
};
