import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import plotly.express as px
import plotly.graph_objects as go
import textwrap

# =========================
# Configuração da página
# =========================

st.set_page_config(
    page_title="Export Strategy Dashboard", # O título que mostra na tab do browser
    layout="wide" # A opção "centered" coloca a página numa coluna central
)

st.title("🍷 Wine Export Strategy – UK Market 🇬🇧")

st.markdown(
"""
Data-driven dashboard to support the commercial decision
to export wine to the United Kingdom.
"""
)

# =========================
# Carregamento dos dados
# =========================

df = pd.read_csv("dataset__Wine_Trabalho_Segmentação.csv")

df = df.drop(columns=["Id","quality"])

df["bound_sulfur_dioxide"] = df["total sulfur dioxide"] - df["free sulfur dioxide"]

df = df.drop(columns=["free sulfur dioxide","total sulfur dioxide"])

# =========================
# Definir as cores dos Clusters
# =========================

cluster_colors = {
    "Cluster 0": "#1f77b4",
    "Cluster 1": "#FFD700",  # dourado para destacar
    "Cluster 2": "#2ca02c"
}

# =========================
# Normalização dos dados
# =========================

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df)

X_scaled = pd.DataFrame(X_scaled, columns=df.columns)

# =========================
# K-MEANS
# =========================

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

X_scaled["cluster"] = clusters

# =========================
# KPI's
# =========================

st.header("Executive Summary")

col1,col2,col3 = st.columns(3)

col1.metric("Wines analysed",len(df))
col2.metric("Clusters",3)
col3.metric("Variables",len(df.columns))

st.divider()

# =========================
# PCA VISUALIZATION
# =========================

pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled.drop(columns=["cluster"]))

plot_df = X_scaled.copy()
plot_df["pca1"] = pca_result[:, 0]
plot_df["pca2"] = pca_result[:, 1]
plot_df["cluster_label"] = plot_df["cluster"].apply(lambda x: f"Cluster {x}")

centroids = plot_df.groupby("cluster_label")[["pca1", "pca2"]].mean().reset_index()

fig = px.scatter(
    plot_df,
    x="pca1",
    y="pca2",
    color="cluster_label",
    color_discrete_map=cluster_colors,
    title="Wine Segmentation - Portfolio View",
    opacity=0.60,
    hover_data={"pca1": False, "pca2": False, "cluster_label": True}
)

fig.update_traces(marker=dict(size=9, line=dict(width=0.5, color="white")))

fig.add_trace(
    go.Scatter(
        x=centroids["pca1"],
        y=centroids["pca2"],
        mode="markers+text",
        text=centroids["cluster_label"],
        textposition="top center",
        marker=dict(size=20, color="black", symbol="diamond"),
        name="Cluster Centre"
    )
)

fig.update_layout(
    template="plotly_white",
    height=600,
    legend_title="Portfolio Clusters",
    xaxis_title="Principal Component 1",
    yaxis_title="Principal Component 2"
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# =========================
# CLUSTER PROFILE
# =========================

cluster_profile = X_scaled.groupby("cluster").mean().round(3)

st.header("Cluster Profiles")
st.caption("Average normalized values (0–1) for each variable by cluster.")

# =========================
# UK TARGET PROFILE
# =========================

uk_target = {
"alcohol":0.35,
"fixed acidity":0.65,
"citric acid":0.60,
"sulphates":0.55
}

scores = {}

for cluster in cluster_profile.index:

    profile = cluster_profile.loc[cluster]

    score = (
        abs(profile["alcohol"]-uk_target["alcohol"])
        + abs(profile["fixed acidity"]-uk_target["fixed acidity"])
        + abs(profile["citric acid"]-uk_target["citric acid"])
        + abs(profile["sulphates"]-uk_target["sulphates"])
    )

    scores[cluster] = score

best_cluster = min(scores,key=scores.get)

# =========================
# 1) CARDS RESUMO POR CLUSTER
# =========================
st.subheader("Executive Snapshot")

cols = st.columns(len(cluster_profile))

for i, cluster in enumerate(cluster_profile.index):
    profile = cluster_profile.loc[cluster]

    cluster_name = f"Cluster {cluster}"
    cluster_color = cluster_colors.get(cluster_name, "#888888")
    border_style = "4px solid gold" if cluster == best_cluster else "none"
    badge_html = (
        '<div style="font-size:13px; font-weight:700; margin-bottom:10px; '
        'background: rgba(255,255,255,0.18); display:inline-block; padding:6px 10px; '
        'border-radius:999px;">Recommended for UK</div>'
        if cluster == best_cluster else ""
    )

    alcohol = profile.get("alcohol", 0)
    acidity = profile.get("fixed acidity", 0)
    citric = profile.get("citric acid", 0)
    sulphates = profile.get("sulphates", 0)

    card_html = f"""
<div style="
    background-color:{cluster_color};
    padding:20px;
    border-radius:18px;
    color:white;
    box-shadow:0 6px 18px rgba(0,0,0,0.18);
    min-height:240px;
    border:{border_style};
    font-family:Arial, sans-serif;
">
    <div style="font-size:20px; font-weight:700; margin-bottom:14px;">
        {cluster_name}
    </div>

    {badge_html}

    <div style="
        background:rgba(255,255,255,0.94);
        color:#1f2937;
        padding:16px 18px;
        border-radius:12px;
        line-height:1.8;
        font-size:15px;
        margin-top:8px;
    ">
        <div><strong>Alcohol:</strong> {alcohol:.2f}</div>
        <div><strong>Fixed Acidity:</strong> {acidity:.2f}</div>
        <div><strong>Citric Acid:</strong> {citric:.2f}</div>
        <div><strong>Sulphates:</strong> {sulphates:.2f}</div>
    </div>
</div>
"""

    with cols[i]:
        components.html(card_html, height=260, scrolling=False)

st.divider()

# =========================
# 2) HEATMAP INTERATIVO
# =========================
import plotly.express as px

heatmap_data = cluster_profile.reset_index().melt(
    id_vars="cluster",
    var_name="Variable",
    value_name="Value"
)

fig_heatmap = px.imshow(
    cluster_profile,
    text_auto=".2f",
    aspect="auto",
    color_continuous_scale="RdYlGn",
    title="Heatmap of Cluster Profiles"
)

fig_heatmap.update_layout(
    xaxis_title="Variables",
    yaxis_title="Cluster",
    height=500
)

st.plotly_chart(fig_heatmap, use_container_width=True)

st.divider()

# =========================
# 3) LEITURA EXECUTIVA AUTOMÁTICA
# =========================
st.subheader("Executive Interpretation")

for cluster in cluster_profile.index:
    profile = cluster_profile.loc[cluster]

    top_features = profile.sort_values(ascending=False).head(3).index.tolist()
    low_features = profile.sort_values(ascending=True).head(2).index.tolist()

    st.markdown(
        f"""
        **Cluster {cluster}**  
        - Strongest characteristics: **{", ".join(top_features)}**
        - Lowest relative characteristics: **{", ".join(low_features)}**
        """
    )

st.divider()  

# =========================
# RADAR CHART
# =========================

st.header("Cluster Comparison")

features = cluster_profile.columns.tolist()

fig = go.Figure()

for cluster in cluster_profile.index:

    cluster_name = f"Cluster {cluster}"
    values = cluster_profile.loc[cluster].values.tolist()
    values += [values[0]]
    theta = features + [features[0]]

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=theta,
        fill='toself',
        name=cluster_name,
        line=dict(color=cluster_colors[cluster_name])
    ))

fig.update_layout(
polar=dict(radialaxis=dict(visible=True, range=[0,1])),
showlegend=True
)

st.plotly_chart(fig,use_container_width=True)

st.divider()

# =========================
# UK Market Fit
# =========================

st.header("UK Market Fit 🇬🇧")

st.success(f"Cluster {best_cluster} best matches the UK market profile")

# =========================
# GAUGE SCORE
# =========================

score_value = 1 - scores[best_cluster]

fig = go.Figure(go.Indicator(
mode="gauge+number",
value=score_value,
title={'text':"UK Market Fit Score"},
gauge={'axis':{'range':[0,1]}}
))

st.plotly_chart(fig,use_container_width=True)

st.divider()

# =========================
# COMMERCIAL STRATEGY
# =========================

st.header("Commercial Strategy")

st.markdown(
"""
Recommended actions:

• Focus on wines belonging to the selected cluster  
• Position as fresh modern wines  
• Target UK importers and wine bars  
• Competitive pricing strategy  

This cluster best matches UK consumer trends.
"""
)
