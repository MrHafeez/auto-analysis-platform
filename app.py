import streamlit as st
import pandas as pd
import numpy as np
import io

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoAnalysis Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="st-"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
[data-testid="stAppViewContainer"] {
    background: #0a0a0f;
}
[data-testid="stSidebar"] {
    background: #0d0d14;
    border-right: 1px solid #1e1e2e;
}
[data-testid="stSidebar"] * { color: #c0c0d0 !important; }

h1, h2, h3 { font-family: 'IBM Plex Mono', monospace !important; color: #e0e0ff; }

.upload-zone {
    border: 2px dashed #3333aa;
    border-radius: 12px;
    padding: 40px;
    text-align: center;
    background: linear-gradient(135deg, #0d0d1a 0%, #111128 100%);
    margin: 20px 0;
}

.analysis-card {
    background: #0f0f1e;
    border: 1px solid #1e1e3e;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 10px 0;
    transition: border-color 0.2s;
}
.analysis-card:hover { border-color: #4444cc; }

.analysis-card.active {
    border-left: 4px solid #4c8aff;
    background: #0d0d22;
}
.analysis-card.disabled {
    opacity: 0.45;
    border-left: 4px solid #333;
}

.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    margin: 2px;
    font-family: 'IBM Plex Mono', monospace;
}
.badge-numeric   { background: #1a2a4a; color: #6ab0ff; border: 1px solid #2a4a8a; }
.badge-categ     { background: #2a1a3a; color: #c06aff; border: 1px solid #4a2a6a; }
.badge-datetime  { background: #1a3a2a; color: #6affc0; border: 1px solid #2a6a4a; }
.badge-bool      { background: #3a2a1a; color: #ffb06a; border: 1px solid #6a4a2a; }
.badge-active    { background: #1a2a1a; color: #6aff8a; border: 1px solid #2a5a2a; }
.badge-inactive  { background: #2a1a1a; color: #ff6a6a; border: 1px solid #5a2a2a; }

.stat-box {
    background: #0f0f1e;
    border: 1px solid #1e1e3e;
    border-radius: 8px;
    padding: 14px 18px;
    text-align: center;
}
.stat-number { font-size: 28px; font-weight: 700; color: #4c8aff; font-family: 'IBM Plex Mono', monospace; }
.stat-label  { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }

.why-box {
    background: #0a1a0a;
    border-left: 3px solid #2a6a2a;
    border-radius: 0 6px 6px 0;
    padding: 8px 14px;
    font-size: 12px;
    color: #8aaa8a;
    margin-top: 8px;
    font-family: 'IBM Plex Mono', monospace;
}
.why-box-off {
    background: #1a0a0a;
    border-left: 3px solid #6a2a2a;
    border-radius: 0 6px 6px 0;
    padding: 8px 14px;
    font-size: 12px;
    color: #aa8a8a;
    margin-top: 8px;
    font-family: 'IBM Plex Mono', monospace;
}

.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #4c8aff;
    border-bottom: 1px solid #1e1e3e;
    padding-bottom: 8px;
    margin: 24px 0 16px 0;
}

.result-panel {
    background: #0c0c1e;
    border: 1px solid #2a2a5e;
    border-radius: 12px;
    padding: 24px;
    margin-top: 16px;
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def detect_column_types(df):
    types = {}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            types[col] = "boolean"
        elif pd.api.types.is_datetime64_any_dtype(s):
            types[col] = "datetime"
        elif pd.api.types.is_numeric_dtype(s):
            types[col] = "numeric"
        else:
            # try parse datetime
            try:
                parsed = pd.to_datetime(s, infer_datetime_format=True)
                if parsed.notna().sum() > len(s) * 0.8:
                    types[col] = "datetime"
                    continue
            except Exception:
                pass
            n_unique = s.nunique()
            if n_unique == 2:
                types[col] = "boolean"
            elif n_unique / max(len(s), 1) < 0.3 or n_unique <= 30:
                types[col] = "categorical"
            else:
                types[col] = "text"
    return types

def get_cols_by_type(col_types, *wanted):
    return [c for c, t in col_types.items() if t in wanted]

def build_analysis_registry(df, col_types):
    numeric_cols   = get_cols_by_type(col_types, "numeric")
    categ_cols     = get_cols_by_type(col_types, "categorical", "boolean")
    datetime_cols  = get_cols_by_type(col_types, "datetime")
    n_rows = len(df)

    registry = []

    # 1 ── Distribution Analysis
    ok = len(numeric_cols) >= 1
    registry.append({
        "id": "distribution",
        "name": "📊 Distribution Analysis",
        "category": "Univariate",
        "active": ok,
        "why_on":  f"Found {len(numeric_cols)} numeric column(s). Distributions reveal skewness, outliers, and spread.",
        "why_off": "Requires at least 1 numeric column.",
        "insight": "Understand the shape of your data — normal, skewed, bimodal?",
        "requires": {"numeric": 1},
    })

    # 2 ── Value Counts / Frequency
    ok = len(categ_cols) >= 1
    registry.append({
        "id": "value_counts",
        "name": "📋 Frequency / Value Counts",
        "category": "Univariate",
        "active": ok,
        "why_on":  f"Found {len(categ_cols)} categorical/boolean column(s). Shows category distribution.",
        "why_off": "Requires at least 1 categorical column.",
        "insight": "Identify dominant categories and class imbalance.",
        "requires": {"categorical": 1},
    })

    # 3 ── Outlier Detection
    ok = len(numeric_cols) >= 1 and n_rows >= 10
    registry.append({
        "id": "outliers",
        "name": "🎯 Outlier Detection (IQR + Z-Score)",
        "category": "Univariate",
        "active": ok,
        "why_on":  f"{len(numeric_cols)} numeric cols, {n_rows} rows. IQR and Z-score methods both applicable.",
        "why_off": "Requires numeric columns and at least 10 rows.",
        "insight": "Find anomalous data points that may skew your analysis.",
        "requires": {"numeric": 1, "rows": 10},
    })

    # 4 ── Missing Values
    missing = df.isnull().sum().sum()
    ok = True  # always show
    registry.append({
        "id": "missing",
        "name": "🕳️ Missing Value Analysis",
        "category": "Data Quality",
        "active": ok,
        "why_on":  f"Dataset has {missing} missing value(s) across {df.isnull().any().sum()} columns.",
        "why_off": "",
        "insight": "Understand completeness. Critical before modelling.",
        "requires": {},
    })

    # 5 ── Correlation Heatmap
    ok = len(numeric_cols) >= 2
    registry.append({
        "id": "correlation",
        "name": "🔥 Correlation Heatmap",
        "category": "Bivariate",
        "active": ok,
        "why_on":  f"{len(numeric_cols)} numeric columns detected — Pearson correlation matrix is computable.",
        "why_off": "Requires at least 2 numeric columns.",
        "insight": "Discover linear relationships between numeric features.",
        "requires": {"numeric": 2},
    })

    # 6 ── Scatter Plot
    ok = len(numeric_cols) >= 2
    registry.append({
        "id": "scatter",
        "name": "✦ Scatter Plot Explorer",
        "category": "Bivariate",
        "active": ok,
        "why_on":  f"{len(numeric_cols)} numeric columns — any two can be plotted against each other.",
        "why_off": "Requires at least 2 numeric columns.",
        "insight": "Visualise relationships, clusters, and trends between two variables.",
        "requires": {"numeric": 2},
    })

    # 7 ── Box Plot by Category
    ok = len(numeric_cols) >= 1 and len(categ_cols) >= 1
    registry.append({
        "id": "boxplot",
        "name": "📦 Box Plot by Category",
        "category": "Bivariate",
        "active": ok,
        "why_on":  f"Mix of {len(numeric_cols)} numeric and {len(categ_cols)} categorical columns found.",
        "why_off": "Requires at least 1 numeric AND 1 categorical column.",
        "insight": "Compare distributions across groups — spot group-level differences.",
        "requires": {"numeric": 1, "categorical": 1},
    })

    # 8 ── Pairplot
    ok = len(numeric_cols) >= 3 and n_rows <= 5000
    registry.append({
        "id": "pairplot",
        "name": "🔷 Pair Plot Matrix",
        "category": "Multivariate",
        "active": ok,
        "why_on":  f"{len(numeric_cols)} numeric columns and {n_rows} rows — pairplot feasible.",
        "why_off": f"Requires 3+ numeric columns and ≤5000 rows. You have {len(numeric_cols)} numeric cols and {n_rows} rows.",
        "insight": "See all pairwise relationships at once — great for feature selection.",
        "requires": {"numeric": 3, "max_rows": 5000},
    })

    # 9 ── PCA
    ok = len(numeric_cols) >= 3 and n_rows >= 10
    registry.append({
        "id": "pca",
        "name": "🌀 PCA — Dimensionality Reduction",
        "category": "Multivariate",
        "active": ok,
        "why_on":  f"{len(numeric_cols)} numeric columns — PCA can reduce to 2D for visualisation.",
        "why_off": "Requires 3+ numeric columns and 10+ rows.",
        "insight": "Reduce high-dimensional data to 2D. See hidden structure.",
        "requires": {"numeric": 3, "rows": 10},
    })

    # 10 ── Normality Test
    ok = len(numeric_cols) >= 1 and n_rows >= 8
    registry.append({
        "id": "normality",
        "name": "📐 Normality Test (Shapiro-Wilk)",
        "category": "Statistical",
        "active": ok,
        "why_on":  f"{len(numeric_cols)} numeric columns, {n_rows} rows. Shapiro-Wilk test applicable.",
        "why_off": "Requires 1+ numeric columns and 8+ rows.",
        "insight": "Know if data is normally distributed — required assumption for many models.",
        "requires": {"numeric": 1, "rows": 8},
    })

    # 11 ── ANOVA
    ok = len(numeric_cols) >= 1 and len(categ_cols) >= 1
    registry.append({
        "id": "anova",
        "name": "⚖️ ANOVA — Group Mean Comparison",
        "category": "Statistical",
        "active": ok,
        "why_on":  f"Numeric target(s) + categorical grouping column(s) available.",
        "why_off": "Requires at least 1 numeric + 1 categorical column.",
        "insight": "Test if group means are significantly different — e.g. sales by region.",
        "requires": {"numeric": 1, "categorical": 1},
    })

    # 12 ── Chi-Square
    ok = len(categ_cols) >= 2
    registry.append({
        "id": "chisquare",
        "name": "χ² Chi-Square Test",
        "category": "Statistical",
        "active": ok,
        "why_on":  f"{len(categ_cols)} categorical columns found — independence test between any two.",
        "why_off": "Requires at least 2 categorical columns.",
        "insight": "Test if two categorical variables are independent of each other.",
        "requires": {"categorical": 2},
    })

    # 13 ── Regression Readiness
    ok = len(numeric_cols) >= 2
    registry.append({
        "id": "regression",
        "name": "📈 Regression Analysis (OLS)",
        "category": "ML Ready",
        "active": ok,
        "why_on":  f"{len(numeric_cols)} numeric columns — you can pick any as target for linear regression.",
        "why_off": "Requires at least 2 numeric columns (1 feature + 1 target).",
        "insight": "Predict a continuous target. See coefficients and R² score.",
        "requires": {"numeric": 2},
    })

    # 14 ── Classification Readiness
    ok = len(numeric_cols) >= 1 and len(categ_cols) >= 1
    registry.append({
        "id": "classification",
        "name": "🏷️ Classification (Decision Tree)",
        "category": "ML Ready",
        "active": ok,
        "why_on":  f"Categorical column(s) can serve as target; numeric columns as features.",
        "why_off": "Requires 1+ numeric feature columns and 1 categorical target.",
        "insight": "Classify records into categories. See feature importance.",
        "requires": {"numeric": 1, "categorical": 1},
    })

    # 15 ── Clustering
    ok = len(numeric_cols) >= 2 and n_rows >= 10
    registry.append({
        "id": "clustering",
        "name": "🔵 K-Means Clustering",
        "category": "ML Ready",
        "active": ok,
        "why_on":  f"{len(numeric_cols)} numeric columns and {n_rows} rows — auto-cluster into groups.",
        "why_off": "Requires 2+ numeric columns and 10+ rows.",
        "insight": "Automatically segment records into meaningful groups.",
        "requires": {"numeric": 2, "rows": 10},
    })

    # 16 ── Time Series
    ok = len(datetime_cols) >= 1 and len(numeric_cols) >= 1
    registry.append({
        "id": "timeseries",
        "name": "📅 Time Series Analysis",
        "category": "Temporal",
        "active": ok,
        "why_on":  f"Datetime column(s) detected: {datetime_cols}. Temporal trends are analysable.",
        "why_off": "Requires at least 1 datetime and 1 numeric column.",
        "insight": "Plot trends over time, detect seasonality and patterns.",
        "requires": {"datetime": 1, "numeric": 1},
    })

    return registry

# ── Run analysis functions ────────────────────────────────────────────────────
def run_distribution(df, col_types):
    import plotly.express as px
    numeric_cols = get_cols_by_type(col_types, "numeric")
    col = st.selectbox("Select column", numeric_cols, key="dist_col")
    chart_type = st.radio("Chart type", ["Histogram", "Box Plot", "Violin"], horizontal=True, key="dist_chart")
    if chart_type == "Histogram":
        fig = px.histogram(df, x=col, nbins=40, template="plotly_dark",
                           color_discrete_sequence=["#4c8aff"])
    elif chart_type == "Box Plot":
        fig = px.box(df, y=col, template="plotly_dark", color_discrete_sequence=["#4c8aff"])
    else:
        fig = px.violin(df, y=col, box=True, template="plotly_dark", color_discrete_sequence=["#4c8aff"])
    fig.update_layout(paper_bgcolor="#0c0c1e", plot_bgcolor="#0c0c1e")
    st.plotly_chart(fig, use_container_width=True)
    s = df[col].dropna()
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Mean",    f"{s.mean():.3f}")
    c2.metric("Std Dev", f"{s.std():.3f}")
    c3.metric("Skew",    f"{s.skew():.3f}")
    c4.metric("Kurtosis",f"{s.kurtosis():.3f}")

def run_value_counts(df, col_types):
    import plotly.express as px
    categ_cols = get_cols_by_type(col_types, "categorical", "boolean")
    col = st.selectbox("Select column", categ_cols, key="vc_col")
    top_n = st.slider("Top N categories", 5, 30, 10, key="vc_topn")
    vc = df[col].value_counts().head(top_n).reset_index()
    vc.columns = [col, "count"]
    fig = px.bar(vc, x=col, y="count", template="plotly_dark",
                 color="count", color_continuous_scale="Blues")
    fig.update_layout(paper_bgcolor="#0c0c1e", plot_bgcolor="#0c0c1e")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(vc, use_container_width=True, hide_index=True)

def run_outliers(df, col_types):
    import plotly.express as px
    numeric_cols = get_cols_by_type(col_types, "numeric")
    col = st.selectbox("Select column", numeric_cols, key="out_col")
    method = st.radio("Method", ["IQR", "Z-Score", "Both"], horizontal=True, key="out_method")
    s = df[col].dropna()
    results = {}
    if method in ["IQR","Both"]:
        Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
        IQR = Q3 - Q1
        mask = (s < Q1 - 1.5*IQR) | (s > Q3 + 1.5*IQR)
        results["IQR"] = mask
        st.info(f"IQR → {mask.sum()} outliers ({mask.sum()/len(s)*100:.1f}%) | Range: [{Q1-1.5*IQR:.2f}, {Q3+1.5*IQR:.2f}]")
    if method in ["Z-Score","Both"]:
        z = np.abs((s - s.mean()) / s.std())
        mask = z > 3
        results["Z-Score"] = mask
        st.info(f"Z-Score → {mask.sum()} outliers ({mask.sum()/len(s)*100:.1f}%) | threshold: |z| > 3")
    fig = px.box(df, y=col, template="plotly_dark", color_discrete_sequence=["#ff6a4c"])
    fig.update_layout(paper_bgcolor="#0c0c1e", plot_bgcolor="#0c0c1e")
    st.plotly_chart(fig, use_container_width=True)

def run_missing(df, col_types):
    import plotly.express as px
    miss = df.isnull().sum().reset_index()
    miss.columns = ["Column", "Missing Count"]
    miss["Missing %"] = (miss["Missing Count"] / len(df) * 100).round(2)
    miss["Type"] = miss["Column"].map(col_types)
    miss = miss.sort_values("Missing %", ascending=False)
    fig = px.bar(miss[miss["Missing Count"]>0], x="Column", y="Missing %",
                 template="plotly_dark", color="Missing %",
                 color_continuous_scale="Reds", text="Missing Count")
    fig.update_layout(paper_bgcolor="#0c0c1e", plot_bgcolor="#0c0c1e")
    if miss["Missing Count"].sum() == 0:
        st.success("✅ No missing values found! Dataset is complete.")
    else:
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(miss, use_container_width=True, hide_index=True)

def run_correlation(df, col_types):
    import plotly.graph_objects as go
    numeric_cols = get_cols_by_type(col_types, "numeric")
    selected = st.multiselect("Select columns", numeric_cols, default=numeric_cols[:min(8,len(numeric_cols))], key="corr_cols")
    if len(selected) < 2:
        st.warning("Select at least 2 columns.")
        return
    corr = df[selected].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale="RdBu", zmid=0, text=corr.round(2).values,
        texttemplate="%{text}", hovertemplate="%{x} vs %{y}: %{z:.3f}<extra></extra>"
    ))
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0c0c1e", plot_bgcolor="#0c0c1e", height=500)
    st.plotly_chart(fig, use_container_width=True)
    # Strongest pairs
    pairs = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            pairs.append({"Feature A": corr.columns[i], "Feature B": corr.columns[j], "Correlation": round(corr.iloc[i,j],4)})
    pairs_df = pd.DataFrame(pairs).sort_values("Correlation", key=abs, ascending=False)
    st.markdown("**Top correlated pairs:**")
    st.dataframe(pairs_df.head(10), use_container_width=True, hide_index=True)

def run_scatter(df, col_types):
    import plotly.express as px
    numeric_cols = get_cols_by_type(col_types, "numeric")
    categ_cols   = get_cols_by_type(col_types, "categorical", "boolean")
    c1, c2, c3 = st.columns(3)
    x_col   = c1.selectbox("X axis",   numeric_cols, key="sc_x")
    y_col   = c2.selectbox("Y axis",   numeric_cols, index=min(1,len(numeric_cols)-1), key="sc_y")
    col_col = c3.selectbox("Color by", ["None"] + categ_cols, key="sc_c")
    color = None if col_col == "None" else col_col
    fig = px.scatter(df, x=x_col, y=y_col, color=color, template="plotly_dark",
                     opacity=0.7, trendline="ols" if color is None else None)
    fig.update_layout(paper_bgcolor="#0c0c1e", plot_bgcolor="#0c0c1e")
    st.plotly_chart(fig, use_container_width=True)

def run_boxplot(df, col_types):
    import plotly.express as px
    numeric_cols = get_cols_by_type(col_types, "numeric")
    categ_cols   = get_cols_by_type(col_types, "categorical", "boolean")
    c1, c2 = st.columns(2)
    y_col = c1.selectbox("Numeric (Y)", numeric_cols, key="bp_y")
    x_col = c2.selectbox("Category (X)", categ_cols, key="bp_x")
    fig = px.box(df, x=x_col, y=y_col, template="plotly_dark",
                 color=x_col, color_discrete_sequence=px.colors.qualitative.Bold)
    fig.update_layout(paper_bgcolor="#0c0c1e", plot_bgcolor="#0c0c1e")
    st.plotly_chart(fig, use_container_width=True)

def run_pairplot(df, col_types):
    import plotly.express as px
    numeric_cols = get_cols_by_type(col_types, "numeric")
    categ_cols   = get_cols_by_type(col_types, "categorical", "boolean")
    selected = st.multiselect("Select numeric columns (max 6)", numeric_cols,
                               default=numeric_cols[:min(4,len(numeric_cols))], key="pp_cols")
    color_col = st.selectbox("Color by", ["None"] + categ_cols, key="pp_color")
    if len(selected) < 2:
        st.warning("Select at least 2 columns.")
        return
    color = None if color_col == "None" else color_col
    fig = px.scatter_matrix(df[selected + ([color] if color else [])].dropna(),
                            dimensions=selected, color=color, template="plotly_dark",
                            opacity=0.5)
    fig.update_layout(paper_bgcolor="#0c0c1e", plot_bgcolor="#0c0c1e", height=600)
    st.plotly_chart(fig, use_container_width=True)

def run_pca(df, col_types):
    import plotly.express as px
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    numeric_cols = get_cols_by_type(col_types, "numeric")
    categ_cols   = get_cols_by_type(col_types, "categorical", "boolean")
    selected = st.multiselect("Features for PCA", numeric_cols,
                               default=numeric_cols[:min(6,len(numeric_cols))], key="pca_cols")
    color_col = st.selectbox("Color by", ["None"] + categ_cols, key="pca_color")
    n_comp = st.slider("Components", 2, min(5,len(selected)) if len(selected)>=2 else 2, 2, key="pca_comp")
    if len(selected) < 2:
        st.warning("Select at least 2 features.")
        return
    sub = df[selected].dropna()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(sub)
    pca = PCA(n_components=n_comp)
    Xp = pca.fit_transform(Xs)
    pca_df = pd.DataFrame(Xp, columns=[f"PC{i+1}" for i in range(n_comp)])
    if color_col != "None":
        pca_df[color_col] = df[color_col].values[:len(pca_df)]
    color = None if color_col == "None" else color_col
    fig = px.scatter(pca_df, x="PC1", y="PC2", color=color, template="plotly_dark",
                     opacity=0.7, title="PCA — 2D Projection")
    fig.update_layout(paper_bgcolor="#0c0c1e", plot_bgcolor="#0c0c1e")
    st.plotly_chart(fig, use_container_width=True)
    ev = pca.explained_variance_ratio_
    st.markdown("**Explained Variance:**")
    ev_df = pd.DataFrame({"Component": [f"PC{i+1}" for i in range(n_comp)],
                          "Explained Variance %": (ev*100).round(2),
                          "Cumulative %": (np.cumsum(ev)*100).round(2)})
    st.dataframe(ev_df, use_container_width=True, hide_index=True)

def run_normality(df, col_types):
    from scipy import stats
    numeric_cols = get_cols_by_type(col_types, "numeric")
    selected = st.multiselect("Select columns to test", numeric_cols,
                               default=numeric_cols[:min(5,len(numeric_cols))], key="norm_cols")
    alpha = st.select_slider("Significance level (α)", [0.01, 0.05, 0.10], value=0.05, key="norm_alpha")
    if not selected:
        return
    results = []
    for col in selected:
        s = df[col].dropna()
        if len(s) > 5000:
            s = s.sample(5000, random_state=42)
        stat, p = stats.shapiro(s)
        results.append({
            "Column": col, "Statistic": round(stat,4), "p-value": round(p,4),
            "Normal?": "✅ Yes" if p > alpha else "❌ No",
            "Interpretation": f"p={p:.4f} {'>' if p>alpha else '<'} α={alpha} → {'Likely normal' if p>alpha else 'Not normal'}"
        })
    st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

def run_anova(df, col_types):
    from scipy import stats
    import plotly.express as px
    numeric_cols = get_cols_by_type(col_types, "numeric")
    categ_cols   = get_cols_by_type(col_types, "categorical", "boolean")
    c1, c2 = st.columns(2)
    num_col = c1.selectbox("Numeric (dependent)", numeric_cols, key="anova_num")
    cat_col = c2.selectbox("Category (groups)",   categ_cols,   key="anova_cat")
    groups = [grp[num_col].dropna().values for _, grp in df.groupby(cat_col)]
    if len(groups) < 2:
        st.warning("Need at least 2 groups.")
        return
    f_stat, p_val = stats.f_oneway(*groups)
    alpha = 0.05
    if p_val < alpha:
        st.error(f"F={f_stat:.3f}, p={p_val:.4f} → Significant difference between groups (p < {alpha})")
    else:
        st.success(f"F={f_stat:.3f}, p={p_val:.4f} → No significant difference (p ≥ {alpha})")
    fig = px.box(df, x=cat_col, y=num_col, template="plotly_dark",
                 color=cat_col, title=f"ANOVA: {num_col} by {cat_col}")
    fig.update_layout(paper_bgcolor="#0c0c1e", plot_bgcolor="#0c0c1e")
    st.plotly_chart(fig, use_container_width=True)

def run_chisquare(df, col_types):
    from scipy.stats import chi2_contingency
    categ_cols = get_cols_by_type(col_types, "categorical", "boolean")
    c1, c2 = st.columns(2)
    col1 = c1.selectbox("Column A", categ_cols, key="chi_a")
    col2 = c2.selectbox("Column B", categ_cols, index=min(1,len(categ_cols)-1), key="chi_b")
    if col1 == col2:
        st.warning("Select two different columns.")
        return
    ct = pd.crosstab(df[col1], df[col2])
    chi2, p, dof, expected = chi2_contingency(ct)
    alpha = 0.05
    if p < alpha:
        st.error(f"χ²={chi2:.3f}, p={p:.4f}, dof={dof} → Columns are DEPENDENT (p < {alpha})")
    else:
        st.success(f"χ²={chi2:.3f}, p={p:.4f}, dof={dof} → Columns are INDEPENDENT (p ≥ {alpha})")
    st.markdown("**Contingency Table:**")
    st.dataframe(ct, use_container_width=True)

def run_regression(df, col_types):
    import plotly.express as px
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error
    numeric_cols = get_cols_by_type(col_types, "numeric")
    target = st.selectbox("Target (Y)", numeric_cols, key="reg_target")
    features = st.multiselect("Features (X)", [c for c in numeric_cols if c != target],
                               default=[c for c in numeric_cols if c != target][:min(4,len(numeric_cols)-1)],
                               key="reg_features")
    if not features:
        st.warning("Select at least 1 feature.")
        return
    sub = df[features + [target]].dropna()
    X = sub[features].values
    y = sub[target].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    c1,c2,c3 = st.columns(3)
    c1.metric("R² Score",  f"{r2:.4f}")
    c2.metric("MAE",       f"{mae:.4f}")
    c3.metric("Train rows",f"{len(X_train)}")
    coef_df = pd.DataFrame({"Feature": features, "Coefficient": model.coef_.round(4)})
    coef_df = coef_df.sort_values("Coefficient", key=abs, ascending=False)
    st.markdown("**Coefficients:**")
    st.dataframe(coef_df, use_container_width=True, hide_index=True)
    pred_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    fig = px.scatter(pred_df, x="Actual", y="Predicted", template="plotly_dark",
                     title="Actual vs Predicted", color_discrete_sequence=["#4c8aff"])
    fig.add_shape(type="line", x0=y_test.min(), y0=y_test.min(),
                  x1=y_test.max(), y1=y_test.max(), line=dict(color="#ff4c4c", dash="dash"))
    fig.update_layout(paper_bgcolor="#0c0c1e", plot_bgcolor="#0c0c1e")
    st.plotly_chart(fig, use_container_width=True)

def run_classification(df, col_types):
    import plotly.express as px
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.preprocessing import LabelEncoder
    numeric_cols = get_cols_by_type(col_types, "numeric")
    categ_cols   = get_cols_by_type(col_types, "categorical", "boolean")
    target  = st.selectbox("Target class (Y)", categ_cols, key="clf_target")
    features = st.multiselect("Features (X)", numeric_cols,
                               default=numeric_cols[:min(4,len(numeric_cols))], key="clf_features")
    max_depth = st.slider("Max tree depth", 2, 10, 4, key="clf_depth")
    if not features:
        st.warning("Select at least 1 numeric feature.")
        return
    sub = df[features + [target]].dropna()
    X = sub[features].values
    le = LabelEncoder()
    y = le.fit_transform(sub[target].astype(str))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.metric("Accuracy", f"{acc:.4f}")
    fi_df = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_.round(4)})
    fi_df = fi_df.sort_values("Importance", ascending=False)
    fig = px.bar(fi_df, x="Feature", y="Importance", template="plotly_dark",
                 color="Importance", color_continuous_scale="Blues", title="Feature Importance")
    fig.update_layout(paper_bgcolor="#0c0c1e", plot_bgcolor="#0c0c1e")
    st.plotly_chart(fig, use_container_width=True)
    st.text(classification_report(y_test, y_pred, target_names=le.classes_))

def run_clustering(df, col_types):
    import plotly.express as px
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    numeric_cols = get_cols_by_type(col_types, "numeric")
    features = st.multiselect("Features for clustering", numeric_cols,
                               default=numeric_cols[:min(5,len(numeric_cols))], key="clust_features")
    k = st.slider("Number of clusters (k)", 2, 10, 3, key="clust_k")
    if len(features) < 2:
        st.warning("Select at least 2 features.")
        return
    sub = df[features].dropna()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(sub)
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(Xs)
    pca = PCA(n_components=2)
    Xp = pca.fit_transform(Xs)
    plot_df = pd.DataFrame(Xp, columns=["PC1","PC2"])
    plot_df["Cluster"] = labels.astype(str)
    fig = px.scatter(plot_df, x="PC1", y="PC2", color="Cluster", template="plotly_dark",
                     title=f"K-Means Clustering (k={k}) — PCA projection",
                     color_discrete_sequence=px.colors.qualitative.Bold)
    fig.update_layout(paper_bgcolor="#0c0c1e", plot_bgcolor="#0c0c1e")
    st.plotly_chart(fig, use_container_width=True)
    cluster_counts = pd.Series(labels).value_counts().sort_index().reset_index()
    cluster_counts.columns = ["Cluster", "Count"]
    st.dataframe(cluster_counts, use_container_width=True, hide_index=True)

def run_timeseries(df, col_types):
    import plotly.express as px
    datetime_cols = get_cols_by_type(col_types, "datetime")
    numeric_cols  = get_cols_by_type(col_types, "numeric")
    c1, c2 = st.columns(2)
    dt_col  = c1.selectbox("Date/Time column", datetime_cols, key="ts_dt")
    val_col = c2.selectbox("Value column",     numeric_cols,  key="ts_val")
    agg     = st.radio("Aggregation", ["None","Daily","Weekly","Monthly"], horizontal=True, key="ts_agg")
    sub = df[[dt_col, val_col]].dropna().copy()
    sub[dt_col] = pd.to_datetime(sub[dt_col])
    sub = sub.sort_values(dt_col)
    if agg == "Daily":
        sub = sub.resample("D", on=dt_col)[val_col].mean().reset_index()
    elif agg == "Weekly":
        sub = sub.resample("W", on=dt_col)[val_col].mean().reset_index()
    elif agg == "Monthly":
        sub = sub.resample("ME", on=dt_col)[val_col].mean().reset_index()
    fig = px.line(sub, x=dt_col, y=val_col, template="plotly_dark",
                  title=f"{val_col} over time", color_discrete_sequence=["#4c8aff"])
    fig.update_layout(paper_bgcolor="#0c0c1e", plot_bgcolor="#0c0c1e")
    st.plotly_chart(fig, use_container_width=True)

RUNNERS = {
    "distribution": run_distribution,
    "value_counts":  run_value_counts,
    "outliers":      run_outliers,
    "missing":       run_missing,
    "correlation":   run_correlation,
    "scatter":       run_scatter,
    "boxplot":       run_boxplot,
    "pairplot":      run_pairplot,
    "pca":           run_pca,
    "normality":     run_normality,
    "anova":         run_anova,
    "chisquare":     run_chisquare,
    "regression":    run_regression,
    "classification":run_classification,
    "clustering":    run_clustering,
    "timeseries":    run_timeseries,
}

CATEGORIES = ["Univariate","Data Quality","Bivariate","Multivariate","Statistical","ML Ready","Temporal"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 AutoAnalysis")
    st.markdown("---")
    layout = st.radio("Layout", ["Wide", "Compact"], index=0, key="layout_sel")
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("""
1. Upload any CSV or Excel file
2. Platform detects column types automatically
3. All valid analyses are **unlocked** based on your data
4. Each analysis shows **why** it's available
5. Select any analysis to run it interactively
    """)
    st.markdown("---")
    st.caption("Supports: CSV, XLSX, XLS")

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:30px 0 10px 0'>
<h1 style='font-size:2.2rem; margin:0'>🔬 AutoAnalysis Platform</h1>
<p style='color:#888; font-size:14px; margin-top:8px; font-family:IBM Plex Mono'>
Upload any dataset → Auto-detect columns → Run any analysis
</p>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Drop your CSV or Excel file here",
    type=["csv","xlsx","xls"],
    label_visibility="collapsed"
)

if uploaded is None:
    st.markdown("""
    <div class='upload-zone'>
        <div style='font-size:48px'>📂</div>
        <h3 style='color:#4c8aff; margin:12px 0 6px'>Drop your dataset here</h3>
        <p style='color:#666; font-size:13px'>Supports CSV, XLSX, XLS — any size, any domain</p>
        <p style='color:#444; font-size:12px; margin-top:16px'>Finance · Marketing · HR · Healthcare · Sales · Research</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(file):
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

df = load_data(uploaded)
col_types = detect_column_types(df)
registry  = build_analysis_registry(df, col_types)

numeric_cols  = get_cols_by_type(col_types, "numeric")
categ_cols    = get_cols_by_type(col_types, "categorical","boolean")
datetime_cols = get_cols_by_type(col_types, "datetime")
active_count  = sum(1 for a in registry if a["active"])

# ── Dataset Overview ──────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>DATASET OVERVIEW</div>", unsafe_allow_html=True)

c1,c2,c3,c4,c5 = st.columns(5)
for col_el, num, label in [
    (c1, len(df),           "Rows"),
    (c2, len(df.columns),   "Columns"),
    (c3, len(numeric_cols), "Numeric"),
    (c4, len(categ_cols),   "Categorical"),
    (c5, active_count,      "Analyses Available"),
]:
    col_el.markdown(f"""
    <div class='stat-box'>
        <div class='stat-number'>{num}</div>
        <div class='stat-label'>{label}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Column type badges
st.markdown("**Detected column types:**")
badge_map = {"numeric":"badge-numeric","categorical":"badge-categ",
             "datetime":"badge-datetime","boolean":"badge-bool","text":"badge-categ"}
badge_label = {"numeric":"NUM","categorical":"CAT","datetime":"DATE","boolean":"BOOL","text":"TEXT"}
badges_html = " ".join(
    "<span class='badge {}'>{} [{}]</span>".format(
        badge_map.get(t, "badge-categ"), col, badge_label.get(t, t)
    )
    for col, t in col_types.items()
)
st.markdown(badges_html, unsafe_allow_html=True)

with st.expander("👁️ Preview Data"):
    st.dataframe(df.head(50), use_container_width=True)

# ── Analysis Menu ─────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>AVAILABLE ANALYSES</div>", unsafe_allow_html=True)
st.markdown(f"**{active_count} of {len(registry)} analyses unlocked** based on your dataset structure.")

selected_analysis = st.session_state.get("selected_analysis", None)

for category in CATEGORIES:
    cat_analyses = [a for a in registry if a["category"] == category]
    if not cat_analyses:
        continue
    active_in_cat = sum(1 for a in cat_analyses if a["active"])
    st.markdown(f"**{category}** — {active_in_cat}/{len(cat_analyses)} available")

    cols_per_row = 2 if layout == "Wide" else 1
    for i in range(0, len(cat_analyses), cols_per_row):
        row_analyses = cat_analyses[i:i+cols_per_row]
        cols = st.columns(cols_per_row)
        for j, analysis in enumerate(row_analyses):
            with cols[j]:
                card_class = "analysis-card active" if analysis["active"] else "analysis-card disabled"
                status_badge = "<span class='badge badge-active'>ACTIVE</span>" if analysis["active"] else "<span class='badge badge-inactive'>INACTIVE</span>"
                why_class = "why-box" if analysis["active"] else "why-box-off"
                why_text  = analysis["why_on"] if analysis["active"] else analysis["why_off"]
                st.markdown(f"""
                <div class='{card_class}'>
                    <div style='display:flex; justify-content:space-between; align-items:center'>
                        <b style='font-size:15px'>{analysis["name"]}</b>
                        {status_badge}
                    </div>
                    <div style='color:#888; font-size:12px; margin-top:4px'>💡 {analysis["insight"]}</div>
                    <div class='{why_class}'>{'✅' if analysis["active"] else '❌'} {why_text}</div>
                </div>
                """, unsafe_allow_html=True)
                if analysis["active"]:
                    if st.button(f"▶ Run", key=f"btn_{analysis['id']}"):
                        st.session_state["selected_analysis"] = analysis["id"]

# ── Run Selected Analysis ─────────────────────────────────────────────────────
if "selected_analysis" in st.session_state and st.session_state["selected_analysis"]:
    aid = st.session_state["selected_analysis"]
    analysis_info = next((a for a in registry if a["id"] == aid), None)
    if analysis_info:
        st.markdown("---")
        st.markdown(f"<div class='section-header'>RUNNING: {analysis_info['name']}</div>",
                    unsafe_allow_html=True)
        st.markdown(f"<div class='result-panel'>", unsafe_allow_html=True)
        with st.spinner("Running analysis..."):
            try:
                RUNNERS[aid](df, col_types)
            except Exception as e:
                st.error(f"Error: {e}")
        st.markdown("</div>", unsafe_allow_html=True)
        if st.button("✖ Close Analysis"):
            st.session_state["selected_analysis"] = None
            st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr>
<div style='text-align:center; font-size:12px; color:#444; font-family:IBM Plex Mono'>
AutoAnalysis Platform · Built with Streamlit · 2025
</div>
""", unsafe_allow_html=True)
