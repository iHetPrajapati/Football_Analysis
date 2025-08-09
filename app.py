
import os, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Football Player Performance Dashboard", layout="wide")

# ---------------------
# Helpers
# ---------------------
def pick(colnames, *candidates):
    def norm(x): return re.sub(r"[^a-z0-9]", "", x.lower())
    cnorm = [norm(c) for c in colnames]
    for cand in candidates:
        cc = norm(cand)
        for i, existing in enumerate(cnorm):
            if cc == existing:
                return colnames[i]
    return None

def safe_div(a, b):
    return np.where((b==0) | pd.isna(b) | pd.isna(a), np.nan, a / b)

@st.cache_data(show_spinner=False)
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    data = df.copy()
    data.columns = [c.strip() for c in data.columns]
    lower_map = {c: c.lower().strip().replace(" ", "_") for c in data.columns}
    data.rename(columns=lower_map, inplace=True)
    # attempt numeric coercion for object-like columns
    for c in data.columns:
        if data[c].dtype == "object":
            s = pd.to_numeric(data[c].astype(str).str.replace(",", "").str.extract(r"([+-]?\d+\.?\d*)")[0], errors="coerce")
            if s.notna().mean() > 0.6:
                data[c] = s

    # Detect key columns
    cols = list(data.columns)
    detected = {
        "player": pick(cols, "player", "player_name", "name", "fullname", "full_name"),
        "team": pick(cols, "team", "club", "squad"),
        "position": pick(cols, "position", "pos"),
        "season": pick(cols, "season", "year", "season_year", "yr"),
        "age": pick(cols, "age"),
        "matches": pick(cols, "matches", "apps", "appearances", "games"),
        "minutes": pick(cols, "minutes", "mins", "minutes_played", "time_played"),
        "goals": pick(cols, "goals", "goal", "g"),
        "assists": pick(cols, "assists", "assist", "a"),
    }

    # Feature engineering
    work = data.copy()
    mcol = detected["minutes"]
    if mcol is not None:
        minutes = pd.to_numeric(work[mcol], errors='coerce')
        minutes_90 = minutes / 90.0
        if detected["goals"] is not None:
            work["goals_per_90"] = safe_div(work[detected["goals"]], minutes_90)
        if detected["assists"] is not None:
            work["assists_per_90"] = safe_div(work[detected["assists"]], minutes_90)
        if (detected["goals"] is not None) and (detected["assists"] is not None):
            work["goal_contrib_per_90"] = safe_div(work[detected["goals"]] + work[detected["assists"]], minutes_90)

    if "goals_per_90" not in work.columns and (detected["matches"] is not None) and (detected["goals"] is not None):
        work["goals_per_match"] = safe_div(work[detected["goals"]], work[detected["matches"]])

    if "assists_per_90" not in work.columns and (detected["matches"] is not None) and (detected["assists"] is not None):
        work["assists_per_match"] = safe_div(work[detected["assists"]], work[detected["matches"]])

    work = work.replace([np.inf, -np.inf], np.nan)

    return work, detected

# ---------------------
# Data loading
# ---------------------
DEFAULT_PATH = "player_performance.csv"
filepath = st.sidebar.text_input("CSV Path", value=DEFAULT_PATH, help="Put your CSV in the same folder as app.py or provide an absolute path.")
if not os.path.exists(filepath):
    st.warning(f"CSV not found at '{filepath}'. Please update the path in the sidebar.")
    st.stop()

work, detected = load_data(filepath)

st.title("⚽ Player Performance Dashboard")
st.caption("Interactive BI dashboard for exploring football player metrics.")

# ---------------------
# Sidebar filters
# ---------------------
with st.sidebar:
    st.header("Filters")
    # dynamic filters if columns exist
    filters = {}
    for key in ["team", "position", "season"]:
        col = detected.get(key)
        if col and col in work.columns:
            values = sorted([x for x in work[col].dropna().unique().tolist() if str(x) != "nan"])
            if values:
                sel = st.multiselect(key.capitalize(), values)
                if sel:
                    filters[col] = sel

    # Apply filters
    mask = pd.Series(True, index=work.index)
    for col, sel in filters.items():
        mask &= work[col].isin(sel)
    fdata = work[mask].copy()

    # Metrics selection
    metric_options = [c for c in ["goals", "assists", "goals_per_90", "assists_per_90", "goal_contrib_per_90", "goals_per_match", "assists_per_match"] if c in fdata.columns]
    selected_metric = st.selectbox("Metric to visualize", metric_options or ["goals"], index=0)

# ---------------------
# KPI Row
# ---------------------
kpi_cols = st.columns(4)
def kpi(col, label, value):
    with col:
        st.metric(label, value)

def fmt_num(x):
    try:
        return f"{float(x):,.2f}"
    except Exception:
        return str(x)

if "goals" in fdata.columns:
    kpi(kpi_cols[0], "Total Goals", int(pd.to_numeric(fdata["goals"], errors="coerce").fillna(0).sum()))
if "assists" in fdata.columns:
    kpi(kpi_cols[1], "Total Assists", int(pd.to_numeric(fdata["assists"], errors="coerce").fillna(0).sum()))
if "goals_per_90" in fdata.columns:
    kpi(kpi_cols[2], "Median Goals/90", fmt_num(pd.to_numeric(fdata["goals_per_90"], errors="coerce").median()))
if "assists_per_90" in fdata.columns:
    kpi(kpi_cols[3], "Median Assists/90", fmt_num(pd.to_numeric(fdata["assists_per_90"], errors="coerce").median()))

st.markdown("---")

# ---------------------
# Charts (matplotlib only)
# ---------------------
label_col = detected.get("player") or detected.get("team") or None

# 1) Top 15 bar chart for selected metric
if selected_metric in fdata.columns:
    topn = fdata[[selected_metric]].copy()
    if label_col and label_col in fdata.columns:
        topn[label_col] = fdata[label_col]
    else:
        topn["row"] = fdata.index.astype(str)
    topn = topn.dropna().sort_values(selected_metric, ascending=False).head(15)

    labels = topn[label_col] if label_col and label_col in topn.columns else topn["row"]
    fig1 = plt.figure()
    plt.bar(labels.astype(str), topn[selected_metric].values)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Top 15 by {selected_metric}")
    plt.tight_layout()
    st.pyplot(fig1)

# 2) Scatter: goals vs assists if available
if "goals" in fdata.columns and "assists" in fdata.columns:
    fig2 = plt.figure()
    plt.scatter(pd.to_numeric(fdata["goals"], errors="coerce"), pd.to_numeric(fdata["assists"], errors="coerce"))
    plt.xlabel("Goals")
    plt.ylabel("Assists")
    plt.title("Goals vs Assists")
    plt.tight_layout()
    st.pyplot(fig2)

# 3) Histogram for selected metric
if selected_metric in fdata.columns:
    fig3 = plt.figure()
    vals = pd.to_numeric(fdata[selected_metric], errors="coerce").dropna()
    plt.hist(vals, bins=20)
    plt.title(f"Distribution: {selected_metric}")
    plt.tight_layout()
    st.pyplot(fig3)

# 4) Correlation matrix
num_cols = fdata.select_dtypes(include=[np.number]).columns
if len(num_cols) >= 2:
    corr = fdata[num_cols].corr()
    fig4 = plt.figure()
    plt.imshow(corr, interpolation='nearest', aspect='auto')
    plt.colorbar()
    plt.xticks(range(len(num_cols)), num_cols, rotation=90)
    plt.yticks(range(len(num_cols)), num_cols)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    st.pyplot(fig4)

st.markdown("---")
st.caption("Tip: Use the sidebar filters (team/position/season) to slice the analysis. Upload a new CSV by changing the path in the sidebar.")
