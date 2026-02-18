import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Allocation Engine", layout="wide")
st.title("Flatmate Allocation Engine (Stage 2)")

@st.cache_data
def load_data():
    merged_df = pd.read_csv("merged_df_processed.csv")
    apt_df = pd.read_csv("apt_df_processed.csv")

    # apt_df should be indexed by Apartment
    if "Apartment" in apt_df.columns:
        apt_df = apt_df.set_index("Apartment")

    return merged_df, apt_df

def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / (s.std(ddof=0) + 1e-9)

def align_pc_cols(merged_df: pd.DataFrame, apt_df: pd.DataFrame):
    # apartment mean cols like PC1_mean, PC2_mean...
    pca_mean_cols = [c for c in apt_df.columns if re.match(r"^PC\d+_mean$", c)]
    pca_mean_cols = sorted(pca_mean_cols, key=lambda x: int(re.findall(r"\d+", x)[0]))

    pc_names_from_means = [c.replace("_mean", "") for c in pca_mean_cols]
    pc_names = [pc for pc in pc_names_from_means if pc in merged_df.columns]
    pca_mean_cols = [pc + "_mean" for pc in pc_names]

    # std cols aligned to same PCs
    pca_std_cols = [c for c in apt_df.columns if re.match(r"^PC\d+_std$", c)]
    pca_std_cols = sorted(pca_std_cols, key=lambda x: int(re.findall(r"\d+", x)[0]))
    pca_std_cols = [c for c in pca_std_cols if c.replace("_std", "") in pc_names]

    return pc_names, pca_mean_cols, pca_std_cols

def ensure_apartment_fields(apt_df: pd.DataFrame, pca_std_cols: list[str]):
    # mean dispersion
    if "mean_dispersion" not in apt_df.columns:
        apt_df["mean_dispersion"] = apt_df[pca_std_cols].mean(axis=1) if pca_std_cols else 0.0

    # dominant cluster from shares
    cluster_share_cols = [c for c in apt_df.columns if c.startswith("cluster_share_")]
    if cluster_share_cols and "dominant_cluster" not in apt_df.columns:
        apt_df["dominant_cluster"] = (
            apt_df[cluster_share_cols]
            .idxmax(axis=1)
            .str.replace("cluster_share_", "", regex=False)
            .astype(int)
        )

    return apt_df

def rank_apartments_for_tenant(
    merged_df: pd.DataFrame,
    apt_df: pd.DataFrame,
    tenant_idx: int,
    pc_names: list[str],
    pca_mean_cols: list[str],
    top_n: int = 15,
    require_vacancy: bool = True,
    exclude_current: bool = True,
    max_occupancy_rate: float | None = None,
    min_free_rooms: int | None = None,
    w_dist: float = 1.0,
    w_disp: float = 0.4,
    w_occ: float = 0.3,
    w_match: float = 0.2,
):
    tenant = merged_df.loc[tenant_idx]
    tenant_vec = tenant[pc_names].values.astype(float)
    tenant_cluster = int(tenant["Cluster_Task2"])
    tenant_apartment = tenant["Apartment"]

    cand = apt_df.copy()

    if require_vacancy and "free_rooms" in cand.columns:
        cand = cand[cand["free_rooms"] > 0].copy()

    if max_occupancy_rate is not None and "occupancy_rate" in cand.columns:
        cand = cand[cand["occupancy_rate"] <= max_occupancy_rate].copy()

    if min_free_rooms is not None and "free_rooms" in cand.columns:
        cand = cand[cand["free_rooms"] >= min_free_rooms].copy()

    if exclude_current:
        cand = cand[cand.index != tenant_apartment].copy()

    # PCA distance to apartment centroid
    apt_vecs = cand[pca_mean_cols].values.astype(float)
    cand["fit_distance"] = np.linalg.norm(apt_vecs - tenant_vec, axis=1)

    # cluster match
    if "dominant_cluster" in cand.columns:
        cand["cluster_match"] = (cand["dominant_cluster"] == tenant_cluster).astype(int)
    else:
        cand["cluster_match"] = 0

    # dispersion
    if "mean_dispersion" not in cand.columns:
        cand["mean_dispersion"] = 0.0

    # zscore components (within candidate set)
    cand["dist_z"] = zscore(cand["fit_distance"])
    cand["disp_z"] = zscore(cand["mean_dispersion"])
    cand["occ_z"] = zscore(cand["occupancy_rate"]) if "occupancy_rate" in cand.columns else 0.0

    # final score
    cand["fit_score"] = (
        -w_dist * cand["dist_z"]
        -w_disp * cand["disp_z"]
        -w_occ  * cand["occ_z"]
        +w_match * cand["cluster_match"]
    )

    out = cand.sort_values("fit_score", ascending=False).head(top_n)

    cols_show = [
        "n_tenants", "ApartmentSize", "free_rooms", "occupancy_rate",
        "fit_distance", "mean_dispersion", "dominant_cluster", "cluster_match", "fit_score"
    ]
    cols_show = [c for c in cols_show if c in out.columns]
    return out[cols_show]

def radar_tenant_vs_apartment(merged_df: pd.DataFrame, tenant_idx: int, apartment_id: str, features: list[str]):
    t = merged_df.loc[tenant_idx, features].astype(float)
    a = merged_df.loc[merged_df["Apartment"] == apartment_id, features].astype(float).mean(axis=0)

    mins = merged_df[features].astype(float).min(axis=0)
    maxs = merged_df[features].astype(float).max(axis=0)
    denom = (maxs - mins).replace(0, 1e-9)

    t_scaled = (t - mins) / denom
    a_scaled = (a - mins) / denom

    categories = features + [features[0]]
    t_vals = t_scaled.tolist() + [t_scaled.iloc[0]]
    a_vals = a_scaled.tolist() + [a_scaled.iloc[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=t_vals, theta=categories, fill="toself", name="Tenant"))
    fig.add_trace(go.Scatterpolar(r=a_vals, theta=categories, fill="toself", name=f"Apt {apartment_id} mean"))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Tenant vs Apartment Profile (normalized 0–1)"
    )
    return fig

# -----------------------------
# Load & prep data
# -----------------------------
merged_df, apt_df = load_data()
pc_names, pca_mean_cols, pca_std_cols = align_pc_cols(merged_df, apt_df)
apt_df = ensure_apartment_fields(apt_df, pca_std_cols)

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Controls")

tenant_idx = st.sidebar.number_input("Tenant index", 0, len(merged_df)-1, 0, 1)
top_n = st.sidebar.slider("Top N recommendations", 5, 30, 15)

require_vacancy = st.sidebar.checkbox("Require vacancy", True)
exclude_current = st.sidebar.checkbox("Exclude current apartment", True)

max_occ_enabled = st.sidebar.checkbox("Max occupancy constraint", False)
max_occ = st.sidebar.slider("Max occupancy rate", 0.2, 1.0, 0.8) if max_occ_enabled else None

min_free_enabled = st.sidebar.checkbox("Min free rooms constraint", False)
min_free = st.sidebar.slider("Min free rooms", 1, 6, 1) if min_free_enabled else None

st.sidebar.subheader("Weights")
w_dist = st.sidebar.slider("Distance weight", 0.0, 2.0, 1.0, 0.1)
w_disp = st.sidebar.slider("Dispersion penalty", 0.0, 2.0, 0.4, 0.1)
w_occ  = st.sidebar.slider("Crowding penalty", 0.0, 2.0, 0.3, 0.1)
w_match = st.sidebar.slider("Cluster match bonus", 0.0, 2.0, 0.2, 0.1)

# -----------------------------
# Recommendations
# -----------------------------
recs = rank_apartments_for_tenant(
    merged_df, apt_df, tenant_idx,
    pc_names, pca_mean_cols,
    top_n=top_n,
    require_vacancy=require_vacancy,
    exclude_current=exclude_current,
    max_occupancy_rate=max_occ,
    min_free_rooms=min_free,
    w_dist=w_dist, w_disp=w_disp, w_occ=w_occ, w_match=w_match
)

def add_why_column(recs: pd.DataFrame) -> pd.DataFrame:
    if len(recs) == 0:
        return recs

    out = recs.copy()

    # robust thresholds (within top-N list)
    dist_good = out["fit_distance"].quantile(0.25) if "fit_distance" in out else None
    disp_good = out["mean_dispersion"].quantile(0.25) if "mean_dispersion" in out else None

    why = []
    for _, r in out.iterrows():
        reasons = []
        if dist_good is not None and r.get("fit_distance", np.inf) <= dist_good:
            reasons.append("Very similar profile")
        if disp_good is not None and r.get("mean_dispersion", np.inf) <= disp_good:
            reasons.append("Low internal diversity")
        if "free_rooms" in out.columns and r.get("free_rooms", 0) >= 2:
            reasons.append("More space (2+ free rooms)")
        if "occupancy_rate" in out.columns and r.get("occupancy_rate", 1.0) <= 0.75:
            reasons.append("Less crowded")
        if r.get("cluster_match", 0) == 1:
            reasons.append("Matches dominant tenant type")
        why.append(", ".join(reasons) if reasons else "Balanced option")

    out["why_recommended"] = why
    return out

recs = add_why_column(recs)

left, right = st.columns([1.1, 1.4])

with left:
    st.subheader("Top Recommendations")
    st.dataframe(recs, use_container_width=True)
    st.download_button(
    "Download recommendations (CSV)",
    data=recs.reset_index().to_csv(index=False).encode("utf-8"),
    file_name=f"recs_tenant_{tenant_idx}.csv",
    mime="text/csv"
)

with right:
    st.subheader("Apartment Map (PCA means)")
    plot_df = apt_df.reset_index()
    plot_df["is_recommended"] = plot_df["Apartment"].isin(recs.index)

    # attach fit_score for recommended apartments
    plot_df["fit_score"] = np.nan
    plot_df.loc[plot_df["is_recommended"], "fit_score"] = recs["fit_score"].values

    # tenant location
    tenant_row = merged_df.loc[tenant_idx]
    tenant_x = float(tenant_row.get("PC1", np.nan))
    tenant_y = float(tenant_row.get("PC2", np.nan))

    fig = px.scatter(
        plot_df,
        x="PC1_mean", y="PC2_mean",
        size="n_tenants" if "n_tenants" in plot_df.columns else None,
        symbol="is_recommended",
        hover_data=[
            "Apartment", "n_tenants", "ApartmentSize", "free_rooms",
            "occupancy_rate", "mean_dispersion", "dominant_cluster",
            "fit_score", "is_recommended"
        ],
        title="Apartments in PCA mean space; recommendations highlighted"
    )
    fig.add_trace(go.Scatter(
        x=[tenant_x], y=[tenant_y],
        mode="markers",
        marker=dict(size=14, symbol="x"),
        name="Selected tenant",
        hovertext=[f"Tenant idx: {tenant_idx}"],
        hoverinfo="text"
    ))
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Radar explainability
# -----------------------------
st.subheader("Explainability: Tenant vs Apartment (Radar)")

default_apt = recs.index[0] if len(recs) else merged_df.loc[tenant_idx, "Apartment"]
apt_choice = st.selectbox("Compare with apartment", options=list(recs.index) if len(recs) else [default_apt])

radar_features = [
    "Emotional_Stability", "Extraversion", "Openness", "Conscientiousness", "Agreeableness",
    "Noise_sensitivity_num", "Cleanliness_num", "Cleanliness_2_num",
    "Cooking_at_home_num", "Guests_over_num", "Alcohol_num", "Vibe_num",
    "Chores_num", "Compatibility_Importance_num", "Smoking_num"
]
radar_features = [f for f in radar_features if f in merged_df.columns]

st.plotly_chart(radar_tenant_vs_apartment(merged_df, tenant_idx, apt_choice, radar_features), use_container_width=True)
