import math
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from app.core.config import settings
from app.validation.schema_inspector import infer_logical_dtype
from app.analysis.statistics import sanitize_float
from app.core.errors import UnsupportedChartError, ColumnNotFoundError

# Dark glassmorphic palette compatible with our frontend theme
DARK_LAYOUT_DEFAULTS = {
    "paper_bgcolor": "rgba(17, 24, 39, 0)",
    "plot_bgcolor": "rgba(17, 24, 39, 0.4)",
    "font": {
        "color": "#94a3b8",
        "family": "Inter, system-ui, sans-serif",
        "size": 12
    },
    "margin": {"l": 50, "r": 30, "t": 50, "b": 50},
    "hoverlabel": {
        "bgcolor": "#1e293b",
        "bordercolor": "#475569",
        "font": {"color": "#f8fafc", "family": "Inter, sans-serif"}
    },
    "xaxis": {
        "gridcolor": "rgba(255, 255, 255, 0.08)",
        "zerolinecolor": "rgba(255, 255, 255, 0.15)",
        "tickfont": {"color": "#94a3b8"}
    },
    "yaxis": {
        "gridcolor": "rgba(255, 255, 255, 0.08)",
        "zerolinecolor": "rgba(255, 255, 255, 0.15)",
        "tickfont": {"color": "#94a3b8"}
    }
}

def build_missingness_chart(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate interactive horizontal bar chart of missing value percentages."""
    cols_data = []
    total_rows = len(df)
    
    for col in df.columns:
        miss_cnt = int(df[col].isna().sum())
        miss_pct = round((miss_cnt / max(total_rows, 1)) * 100, 2)
        cols_data.append({"column": str(col), "missing_count": miss_cnt, "missing_pct": miss_pct})
        
    # Sort ascending for clean horizontal bar display (highest at top)
    cols_data.sort(key=lambda x: x["missing_pct"], reverse=False)
    
    columns = [c["column"] for c in cols_data]
    percentages = [c["missing_pct"] for c in cols_data]
    counts = [c["missing_count"] for c in cols_data]
    
    # Color bar according to severity
    colors = [
        "#ef4444" if p >= 50 else ("#f59e0b" if p > 10 else "#38bdf8")
        for p in percentages
    ]
    
    trace = {
        "type": "bar",
        "orientation": "h",
        "x": percentages,
        "y": columns,
        "customdata": counts,
        "hovertemplate": "<b>%{y}</b><br>Missing: %{x}% (%{customdata} rows)<extra></extra>",
        "marker": {
            "color": colors,
            "line": {"color": "rgba(255, 255, 255, 0.1)", "width": 1}
        }
    }
    
    layout = {
        **DARK_LAYOUT_DEFAULTS,
        "title": {"text": "Missing Data Percentage by Column", "font": {"color": "#f8fafc", "size": 15}},
        "xaxis": {
            **DARK_LAYOUT_DEFAULTS["xaxis"],
            "title": "Missing (%)",
            "range": [0, max(max(percentages or [0]) * 1.1, 10)]
        },
        "yaxis": {
            **DARK_LAYOUT_DEFAULTS["yaxis"],
            "automargin": True
        }
    }
    
    return {
        "chart_type": "bar",
        "title": "Missing Data by Column",
        "data": [trace],
        "layout": layout
    }

def build_univariate_numeric_charts(series: pd.Series, col_name: str) -> Dict[str, Any]:
    """Generate combined histogram and box plot for a numerical column."""
    s_clean = pd.to_numeric(series.replace([np.inf, -np.inf], np.nan), errors="coerce").dropna()
    vals = [round(float(v), 4) for v in s_clean.tolist()]
    
    # Subsample if extremely large to prevent payload bloating
    if len(vals) > settings.EDA_MAX_SCATTER_POINTS * 2:
        np.random.seed(settings.EDA_RANDOM_SEED)
        sampled_vals = np.random.choice(vals, size=settings.EDA_MAX_SCATTER_POINTS * 2, replace=False).tolist()
    else:
        sampled_vals = vals
        
    hist_trace = {
        "type": "histogram",
        "x": sampled_vals,
        "name": "Frequency",
        "marker": {
            "color": "#6366f1",
            "line": {"color": "#818cf8", "width": 1}
        },
        "opacity": 0.85,
        "hovertemplate": "Range: %{x}<br>Count: %{y}<extra></extra>"
    }
    
    box_trace = {
        "type": "box",
        "x": sampled_vals,
        "name": col_name,
        "marker": {"color": "#38bdf8"},
        "boxpoints": "outliers",
        "hovertemplate": "Value: %{x}<extra></extra>"
    }
    
    hist_layout = {
        **DARK_LAYOUT_DEFAULTS,
        "title": {"text": f"Distribution of '{col_name}'", "font": {"color": "#f8fafc", "size": 15}},
        "xaxis": {**DARK_LAYOUT_DEFAULTS["xaxis"], "title": col_name},
        "yaxis": {**DARK_LAYOUT_DEFAULTS["yaxis"], "title": "Frequency"}
    }
    
    box_layout = {
        **DARK_LAYOUT_DEFAULTS,
        "title": {"text": f"Box Plot & Potential Outliers: '{col_name}'", "font": {"color": "#f8fafc", "size": 15}},
        "xaxis": {**DARK_LAYOUT_DEFAULTS["xaxis"], "title": col_name}
    }
    
    return {
        "column": col_name,
        "histogram": {
            "chart_type": "histogram",
            "data": [hist_trace],
            "layout": hist_layout
        },
        "box_plot": {
            "chart_type": "box",
            "data": [box_trace],
            "layout": box_layout
        }
    }

def build_univariate_categorical_chart(
    series: pd.Series,
    col_name: str,
    top_n: int = settings.EDA_MAX_CATEGORIES_PER_CHART
) -> Dict[str, Any]:
    """Generate bar chart for categorical frequencies."""
    s_nonnull = series.dropna().astype(str).str.strip()
    val_counts = s_nonnull.value_counts()
    
    top_items = val_counts.head(top_n)
    categories = [str(k) for k in top_items.index]
    counts = [int(v) for v in top_items.values]
    
    other_count = len(s_nonnull) - sum(counts)
    if other_count > 0:
        categories.append("Other")
        counts.append(other_count)
        
    trace = {
        "type": "bar",
        "x": categories,
        "y": counts,
        "marker": {
            "color": "#8b5cf6",
            "line": {"color": "#a78bfa", "width": 1}
        },
        "hovertemplate": "<b>%{x}</b><br>Count: %{y}<extra></extra>"
    }
    
    layout = {
        **DARK_LAYOUT_DEFAULTS,
        "title": {"text": f"Top Categories for '{col_name}'", "font": {"color": "#f8fafc", "size": 15}},
        "xaxis": {**DARK_LAYOUT_DEFAULTS["xaxis"], "title": col_name, "tickangle": -30},
        "yaxis": {**DARK_LAYOUT_DEFAULTS["yaxis"], "title": "Count"}
    }
    
    return {
        "column": col_name,
        "chart_type": "bar",
        "data": [trace],
        "layout": layout
    }

def build_correlation_heatmap(corr_matrix_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate interactive correlation heatmap trace and layout."""
    cols = corr_matrix_data.get("columns", [])
    matrix = corr_matrix_data.get("matrix", [])
    
    trace = {
        "type": "heatmap",
        "z": matrix,
        "x": cols,
        "y": cols,
        "colorscale": [
            [0.0, "#0284c7"],
            [0.5, "#1e293b"],
            [1.0, "#f43f5e"]
        ],
        "zmin": -1.0,
        "zmax": 1.0,
        "hovertemplate": "<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>"
    }
    
    layout = {
        **DARK_LAYOUT_DEFAULTS,
        "title": {"text": "Pearson Correlation Heatmap", "font": {"color": "#f8fafc", "size": 15}},
        "xaxis": {**DARK_LAYOUT_DEFAULTS["xaxis"], "tickangle": -45, "automargin": True},
        "yaxis": {**DARK_LAYOUT_DEFAULTS["yaxis"], "automargin": True}
    }
    
    return {
        "chart_type": "heatmap",
        "title": "Correlation Heatmap",
        "data": [trace],
        "layout": layout
    }

def recommend_bivariate_chart(
    df: pd.DataFrame,
    col_x: str,
    col_y: Optional[str] = None,
    requested_chart_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Deterministically recommend and construct chart payload for X (and optional Y).
    Sampling: Caps scatter data to EDA_MAX_SCATTER_POINTS with fixed seed.
    """
    if col_x not in df.columns:
        raise ColumnNotFoundError(col_x, available_columns=list(df.columns))
    if col_y and col_y not in df.columns:
        raise ColumnNotFoundError(col_y, available_columns=list(df.columns))
        
    dtype_x = infer_logical_dtype(df[col_x])
    dtype_y = infer_logical_dtype(df[col_y]) if col_y else None
    
    # 1. Single Column Analysis
    if not col_y:
        if dtype_x in ("integer", "float"):
            charts = build_univariate_numeric_charts(df[col_x], col_x)
            # Return histogram by default
            return {
                "chart_type": "histogram",
                "recommended_type": "histogram",
                "title": f"Distribution of {col_x}",
                "data": charts["histogram"]["data"],
                "layout": charts["histogram"]["layout"],
                "box_plot": charts["box_plot"]
            }
        elif dtype_x == "datetime-like":
            s_dt = pd.to_datetime(df[col_x], errors="coerce").dropna()
            trace = {
                "type": "histogram",
                "x": [str(d.isoformat()) for d in s_dt],
                "marker": {"color": "#38bdf8"}
            }
            layout = {
                **DARK_LAYOUT_DEFAULTS,
                "title": {"text": f"Records Over Time ({col_x})", "font": {"color": "#f8fafc"}},
                "xaxis": {**DARK_LAYOUT_DEFAULTS["xaxis"], "title": col_x},
                "yaxis": {**DARK_LAYOUT_DEFAULTS["yaxis"], "title": "Count"}
            }
            return {
                "chart_type": "histogram",
                "recommended_type": "histogram",
                "title": f"Temporal Records for {col_x}",
                "data": [trace],
                "layout": layout
            }
        else:
            bar_chart = build_univariate_categorical_chart(df[col_x], col_x)
            return {
                "chart_type": "bar",
                "recommended_type": "bar",
                "title": f"Top Categories for {col_x}",
                "data": bar_chart["data"],
                "layout": bar_chart["layout"]
            }

    # 2. Bivariate Analysis
    # A) Numeric + Numeric -> Scatter plot
    if dtype_x in ("integer", "float") and dtype_y in ("integer", "float"):
        clean_df = df[[col_x, col_y]].dropna().replace([np.inf, -np.inf], np.nan).dropna()
        clean_df[col_x] = pd.to_numeric(clean_df[col_x], errors="coerce")
        clean_df[col_y] = pd.to_numeric(clean_df[col_y], errors="coerce")
        clean_df = clean_df.dropna()
        
        total_pts = len(clean_df)
        sampled = False
        if total_pts > settings.EDA_MAX_SCATTER_POINTS:
            sampled = True
            clean_df = clean_df.sample(n=settings.EDA_MAX_SCATTER_POINTS, random_state=settings.EDA_RANDOM_SEED)
            
        x_vals = [round(float(v), 4) for v in clean_df[col_x]]
        y_vals = [round(float(v), 4) for v in clean_df[col_y]]
        
        trace = {
            "type": "scatter",
            "mode": "markers",
            "x": x_vals,
            "y": y_vals,
            "marker": {
                "color": "#38bdf8",
                "size": 6,
                "opacity": 0.65,
                "line": {"color": "rgba(255, 255, 255, 0.2)", "width": 0.5}
            },
            "hovertemplate": f"<b>{col_x}</b>: %{{x}}<br><b>{col_y}</b>: %{{y}}<extra></extra>"
        }
        
        # Calculate trendline slope/intercept if points exist
        traces = [trace]
        if len(x_vals) > 2:
            try:
                slope, intercept = np.polyfit(x_vals, y_vals, 1)
                x_min, x_max = min(x_vals), max(x_vals)
                trend_x = [x_min, x_max]
                trend_y = [round(float(slope * x_min + intercept), 4), round(float(slope * x_max + intercept), 4)]
                trend_trace = {
                    "type": "scatter",
                    "mode": "lines",
                    "name": "Linear Trend",
                    "x": trend_x,
                    "y": trend_y,
                    "line": {"color": "#f43f5e", "width": 2, "dash": "dash"},
                    "hoverinfo": "skip"
                }
                traces.append(trend_trace)
            except Exception:
                pass
                
        sub_notice = f" (Sampled {settings.EDA_MAX_SCATTER_POINTS} of {total_pts} points)" if sampled else ""
        layout = {
            **DARK_LAYOUT_DEFAULTS,
            "title": {"text": f"Scatter Plot: {col_x} vs {col_y}{sub_notice}", "font": {"color": "#f8fafc", "size": 15}},
            "xaxis": {**DARK_LAYOUT_DEFAULTS["xaxis"], "title": col_x},
            "yaxis": {**DARK_LAYOUT_DEFAULTS["yaxis"], "title": col_y}
        }
        
        return {
            "chart_type": "scatter",
            "recommended_type": "scatter",
            "title": f"{col_x} vs {col_y}",
            "points_plotted": len(x_vals),
            "sampled": sampled,
            "data": traces,
            "layout": layout
        }
        
    # B) Datetime + Numeric -> Time Series Line
    if (dtype_x == "datetime-like" and dtype_y in ("integer", "float")) or \
       (dtype_y == "datetime-like" and dtype_x in ("integer", "float")):
        dt_col = col_x if dtype_x == "datetime-like" else col_y
        num_col = col_y if dtype_x == "datetime-like" else col_x
        
        clean_df = df[[dt_col, num_col]].dropna()
        clean_df[dt_col] = pd.to_datetime(clean_df[dt_col], errors="coerce")
        clean_df[num_col] = pd.to_numeric(clean_df[num_col], errors="coerce")
        clean_df = clean_df.dropna().sort_values(by=dt_col)
        
        # Subsample if too many points
        total_pts = len(clean_df)
        sampled = False
        if total_pts > settings.EDA_MAX_SCATTER_POINTS:
            sampled = True
            clean_df = clean_df.iloc[::max(len(clean_df) // settings.EDA_MAX_SCATTER_POINTS, 1)]
            
        x_vals = [str(d.isoformat()) for d in clean_df[dt_col]]
        y_vals = [round(float(v), 4) for v in clean_df[num_col]]
        
        trace = {
            "type": "scatter",
            "mode": "lines+markers",
            "x": x_vals,
            "y": y_vals,
            "line": {"color": "#10b981", "width": 1.5},
            "marker": {"size": 4, "color": "#10b981"},
            "hovertemplate": f"Date: %{{x}}<br>{num_col}: %{{y}}<extra></extra>"
        }
        
        layout = {
            **DARK_LAYOUT_DEFAULTS,
            "title": {"text": f"{num_col} Over Time ({dt_col})", "font": {"color": "#f8fafc", "size": 15}},
            "xaxis": {**DARK_LAYOUT_DEFAULTS["xaxis"], "title": dt_col},
            "yaxis": {**DARK_LAYOUT_DEFAULTS["yaxis"], "title": num_col}
        }
        
        return {
            "chart_type": "line",
            "recommended_type": "line",
            "title": f"{num_col} Over Time",
            "data": [trace],
            "layout": layout
        }
        
    # C) Categorical + Numeric -> Grouped Box Plot
    if (dtype_x in ("categorical", "boolean", "text") and dtype_y in ("integer", "float")) or \
       (dtype_y in ("categorical", "boolean", "text") and dtype_x in ("integer", "float")):
        cat_col = col_x if dtype_x in ("categorical", "boolean", "text") else col_y
        num_col = col_y if dtype_x in ("categorical", "boolean", "text") else col_x
        
        clean_df = df[[cat_col, num_col]].dropna()
        clean_df[cat_col] = clean_df[cat_col].astype(str).str.strip()
        clean_df[num_col] = pd.to_numeric(clean_df[num_col], errors="coerce")
        clean_df = clean_df.dropna()
        
        # Limit to top 8 categories
        top_cats = clean_df[cat_col].value_counts().head(8).index.tolist()
        clean_df = clean_df[clean_df[cat_col].isin(top_cats)]
        
        traces = []
        for cat in top_cats:
            sub_vals = clean_df[clean_df[cat_col] == cat][num_col].tolist()
            if len(sub_vals) > settings.EDA_MAX_SCATTER_POINTS:
                sub_vals = np.random.choice(sub_vals, size=settings.EDA_MAX_SCATTER_POINTS, replace=False).tolist()
            traces.append({
                "type": "box",
                "y": [round(float(v), 4) for v in sub_vals],
                "name": str(cat),
                "boxpoints": "outliers"
            })
            
        layout = {
            **DARK_LAYOUT_DEFAULTS,
            "title": {"text": f"Distribution of {num_col} by {cat_col}", "font": {"color": "#f8fafc", "size": 15}},
            "xaxis": {**DARK_LAYOUT_DEFAULTS["xaxis"], "title": cat_col},
            "yaxis": {**DARK_LAYOUT_DEFAULTS["yaxis"], "title": num_col}
        }
        
        return {
            "chart_type": "box",
            "recommended_type": "box",
            "title": f"{num_col} by {cat_col}",
            "data": traces,
            "layout": layout
        }
        
    # D) Categorical + Categorical -> Grouped Count / Heatmap
    if dtype_x in ("categorical", "boolean", "text") and dtype_y in ("categorical", "boolean", "text"):
        clean_df = df[[col_x, col_y]].dropna()
        clean_df[col_x] = clean_df[col_x].astype(str).str.strip()
        clean_df[col_y] = clean_df[col_y].astype(str).str.strip()
        
        top_x = clean_df[col_x].value_counts().head(8).index.tolist()
        top_y = clean_df[col_y].value_counts().head(8).index.tolist()
        
        clean_sub = clean_df[clean_df[col_x].isin(top_x) & clean_df[col_y].isin(top_y)]
        ct = pd.crosstab(clean_sub[col_y], clean_sub[col_x])
        
        # Align columns and index
        ct = ct.reindex(index=top_y, columns=top_x, fill_value=0)
        
        trace = {
            "type": "heatmap",
            "z": ct.values.tolist(),
            "x": [str(c) for c in ct.columns],
            "y": [str(i) for i in ct.index],
            "colorscale": "Blues",
            "hovertemplate": f"<b>{col_x}</b>: %{{x}}<br><b>{col_y}</b>: %{{y}}<br>Count: %{{z}}<extra></extra>"
        }
        
        layout = {
            **DARK_LAYOUT_DEFAULTS,
            "title": {"text": f"Contingency Heatmap: {col_x} vs {col_y}", "font": {"color": "#f8fafc", "size": 15}},
            "xaxis": {**DARK_LAYOUT_DEFAULTS["xaxis"], "title": col_x, "tickangle": -30},
            "yaxis": {**DARK_LAYOUT_DEFAULTS["yaxis"], "title": col_y}
        }
        
        return {
            "chart_type": "heatmap",
            "recommended_type": "heatmap",
            "title": f"Frequency Table: {col_x} vs {col_y}",
            "data": [trace],
            "layout": layout
        }
        
    raise UnsupportedChartError(
        f"Unable to automatically recommend a visualization for columns '{col_x}' ({dtype_x}) and '{col_y}' ({dtype_y})."
    )
