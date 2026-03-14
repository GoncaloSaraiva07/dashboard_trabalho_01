import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import plotly.express as px
import plotly.graph_objects as go

# =========================
# Configuração da página
# =========================

st.set_page_config(
    page_title="Dashboard Estratégia de Exportação",
    layout="wide"
)

# =========================
# Título principal
# =========================

st.title("🍷 Estratégia de Exportação de Vinho – Mercado do Reino Unido 🇬🇧")

st.markdown(
"""
Dashboard analítico para apoiar a decisão comercial de exportação
para o mercado do Reino Unido, com base na segmentação de perfis de vinho.
"""
)

# =========================
# Carregamento de dados
# =========================

df_raw = pd.read_csv("dataset__Wine_Trabalho_Segmentação.csv")

quality_original = df_raw["quality"].copy()

X = df_raw.drop(columns=["Id", "quality"]).copy()

X["bound_sulfur_dioxide"] = X["total sulfur dioxide"] - X["free sulfur dioxide"]
X = X.drop(columns=["free sulfur dioxide", "total sulfur dioxide"])

# =========================
# Cores dos clusters
# =========================

cluster_colors = {
    "Cluster 0": "#1f77b4",
    "Cluster 1": "#FFD700",
    "Cluster 2": "#2ca02c"
}

# =========================
# Normalização de dados
# =========================

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

quality_scaled = MinMaxScaler().fit_transform(quality_original.to_frame())
X_scaled["quality"] = quality_scaled

# =========================
# Aplicação do algoritmo K-Means
# =========================

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled.drop(columns=["quality"]))

X_scaled["cluster"] = clusters

# =========================
# KPI's
# =========================

st.header("Resumo Executivo")

col1, col2, col3 = st.columns(3)

col1.metric("Vinhos analisados", len(X))
col2.metric("Clusters identificados", 3)
col3.metric("Variáveis analisadas", len(X.columns))

st.divider()

# =========================
# PCA
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
    title="Segmentação do Portefólio de Vinhos",
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
        name="Centro do Cluster"
    )
)

fig.update_layout(
    template="plotly_white",
    height=600,
    legend_title="Clusters do Portefólio",
    xaxis_title="Componente Principal 1",
    yaxis_title="Componente Principal 2"
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# =========================
# Perfis dos clusters
# =========================

cluster_profile = X_scaled.groupby("cluster").mean()

# =========================
# MATCH ENTRE MERCADO E VINHO
# Valores ajustados segundo a PARTE 3 - MATCH ENTRE MERCADO E VINHO
# =========================

cluster_match = pd.DataFrame({
    "cluster": [2, 1, 0],
    "alcohol": [0.295065, 0.329398, 0.326935],
    "fixed acidity": [0.319251, 0.217164, 0.564179],
    "residual sugar": [0.109303, 0.093414, 0.152891],
    "volatile acidity": [0.244162, 0.360314, 0.212653],
    "quality": [0.532383, 0.503721, 0.582883],
    "uk_match_score": [-0.024202, 0.015123, 0.045532]
}).set_index("cluster")

best_cluster = 2

st.header("Perfis dos Clusters")
st.caption("Valores médios normalizados entre 0 e 1 para cada variável por cluster.")

# =========================
# Resumo Estratégico por Cluster
# =========================

st.subheader("Resumo Estratégico por Cluster")

cols = st.columns(len(cluster_match))

for i, cluster in enumerate(cluster_match.index):
    profile = cluster_match.loc[cluster]

    cluster_name = f"Cluster {cluster}"
    cluster_color = cluster_colors.get(cluster_name, "#888888")
    border_style = "4px solid gold" if cluster == best_cluster else "none"

    badge_html = (
        '<div style="font-size:13px; font-weight:700; margin-bottom:10px; '
        'background: rgba(255,255,255,0.18); display:inline-block; padding:6px 10px; '
        'border-radius:999px;">Recomendado para o UK</div>'
        if cluster == best_cluster else ""
    )

    alcohol = profile["alcohol"]
    acidity = profile["fixed acidity"]
    residual_sugar = profile["residual sugar"]
    quality = profile["quality"]

    card_html = f"""
<div style="
    background-color:{cluster_color};
    padding:20px;
    border-radius:18px;
    color:white;
    box-shadow:0 6px 18px rgba(0,0,0,0.18);
    min-height:250px;
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
        <div><strong>Álcool:</strong> {alcohol:.2f}</div>
        <div><strong>Acidez Fixa:</strong> {acidity:.2f}</div>
        <div><strong>Açúcar Residual:</strong> {residual_sugar:.2f}</div>
        <div><strong>Qualidade:</strong> {quality:.2f}</div>
    </div>
</div>
"""
    with cols[i]:
        components.html(card_html, height=270, scrolling=False)

st.divider()

# =========================
# Heatmap
# =========================

fig_heatmap = px.imshow(
    cluster_match[["alcohol", "fixed acidity", "residual sugar", "volatile acidity", "quality", "uk_match_score"]],
    text_auto=".2f",
    aspect="auto",
    color_continuous_scale="RdYlGn",
    title="Mapa de Calor dos Perfis dos Clusters"
)

fig_heatmap.update_layout(
    xaxis_title="Variáveis",
    yaxis_title="Cluster",
    height=500
)

st.plotly_chart(fig_heatmap, use_container_width=True)

st.divider()

# =========================
# Interpretação executiva
# =========================

st.subheader("Interpretação Executiva")

interpretacoes = {
    2: "Perfil mais alinhado com o mercado UK, combinando frescura, menor teor alcoólico e melhor adequação global ao perfil-alvo.",
    1: "Perfil intermédio, com potencial comercial moderado, mas menos ajustado ao mercado britânico do que o Cluster 2.",
    0: "Perfil mais estruturado e com maior potencial de margem, mas menos alinhado com a lógica de entrada prioritária no UK."
}

for cluster in cluster_match.index:
    st.markdown(f"**Cluster {cluster}** — {interpretacoes[cluster]}")

st.divider()

# =========================
# Comparação dos Clusters
# =========================

st.header("Comparação dos Clusters")

features_radar = ["alcohol", "fixed acidity", "residual sugar", "volatile acidity", "quality", "uk_match_score"]

fig = go.Figure()

for cluster in cluster_match.index:
    cluster_name = f"Cluster {cluster}"
    values = cluster_match.loc[cluster, features_radar].tolist()
    values = [max(v, 0) for v in values]  # evitar problema visual com score negativo
    values += [values[0]]
    theta = features_radar + [features_radar[0]]

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=theta,
        fill='toself',
        name=cluster_name,
        line=dict(color=cluster_colors[cluster_name])
    ))

fig.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 0.7])),
    showlegend=True,
    title="Radar de Comparação dos Perfis dos Clusters"
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# =========================
# Adequação ao Mercado do Reino Unido
# =========================

st.header("Adequação ao Mercado do Reino Unido 🇬🇧")

st.success("O Cluster 2 é o que melhor se ajusta ao perfil-alvo do mercado do Reino Unido.")

uk_market_score = 17

col_a, col_b = st.columns([1, 1])

with col_a:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=uk_market_score,
        title={'text': "Market Opportunity Score - United Kingdom"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#2ca02c"},
            'steps': [
                {'range': [0, 33], 'color': "#f8d7da"},
                {'range': [33, 66], 'color': "#fff3cd"},
                {'range': [66, 100], 'color': "#d1e7dd"}
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.markdown(
        """
**Leitura estratégica do score de mercado**

O mercado do Reino Unido apresenta um **Market Opportunity Score de 17**, abaixo de mercados como os EUA, mas ainda assim a opção estratégica passa por escolher o mercado UK. Sabemos que face ao contexto de atual instabilidade nos EUA acarreta um maior risco nesta primeira escolha de qual o primeiro mercado para exportação, assumindo assim a opção UK como o mercado em que o **perfil de vinho com maior probabilidade de sucesso para entrada seletiva**.

Neste enquadramento, o **Cluster 2** é o perfil que melhor responde ao target britânico.
"""
    )

st.divider()

# =========================
# Estratégia Comercial
# =========================

st.header("Estratégia Comercial para Exportação")

st.markdown(
"""
**Ações recomendadas:**

• Focar a estratégia de exportação no **Cluster 2**  
• Posicionar o produto como vinho fresco, moderno e alinhado com o consumo atual  
• Direcionar a oferta para importadores, distribuidores, wine bars e canais especializados no Reino Unido  
• Definir uma estratégia de preço competitiva, equilibrando volume e margem  

O Cluster 2 é o que demonstra o melhor alinhamento com as tendências de consumo do mercado britânico.
"""
)

st.divider()

# =========================
# Síntese Estratégica Final
# =========================

st.header("Síntese Estratégica Final")

st.markdown(
"""
• A análise de clusterização identificou **três perfis distintos de vinho**, permitindo estruturar uma estratégia de portefólio orientada para **rentabilidade e expansão internacional**.  

• **O Cluster 2** apresenta o melhor alinhamento com o mercado britânico, sendo o perfil com maior probabilidade de sucesso na entrada no UK.  

• **O Cluster 0** apresenta um perfil mais estruturado e equilibrado, podendo suportar **maior valor percebido e maior margem**.  

• **O Cluster 1** pode desempenhar um papel de suporte ao portefólio, contribuindo para **volume de vendas e competitividade de preço**.
"""
)

st.subheader("Prioridades de Execução")

st.markdown(
"""
• Priorizar o **Cluster 2** como produto principal de entrada no mercado do Reino Unido.  

• Utilizar os restantes clusters para complementar a estratégia de portefólio, combinando **volume e margem**.  

• Ajustar o posicionamento comercial por canal, diferenciando oferta premium e oferta de maior rotação.  

• Sustentar a expansão internacional com uma lógica orientada por dados.
"""
)

st.subheader("Impacto Esperado")

st.markdown(
"""
• **Aumento das vendas totais**, através de uma oferta mais alinhada com o perfil do mercado britânico.  

• **Melhoria da rentabilidade da garrafeira**, ao combinar produtos de maior margem com produtos de maior rotação.  

• **Redução do risco comercial**, ao selecionar o perfil com maior probabilidade de sucesso num novo mercado de exportação.  

• **Maior eficiência de portefólio**, com melhor adequação entre produto, canal e consumidor.
"""
)

st.subheader("Avaliação da Estratégia")

st.markdown(
"""
• A segmentação permite transformar dados em **decisão comercial**, apoiando uma estratégia de internacionalização mais robusta.  

• O **Cluster 2** deve ser entendido como **produto âncora de entrada** no mercado do Reino Unido, enquanto os restantes clusters desempenham um papel complementar na rentabilidade global.  

• Esta abordagem permite equilibrar **crescimento de vendas e aumento de margem**, criando uma estratégia mais sustentável para a garrafeira.
"""
)

st.divider()

# =========================
# Recomendação Final
# =========================

st.header("Recomendação Final de Exportação para o Reino Unido")

col_img, col_txt = st.columns([1, 2])

with col_img:
    st.image("best_cluster_uk.png", width=280)

with col_txt:
    st.markdown(
        """
### Cluster Recomendado: **Cluster 2**

A análise de segmentação e o exercício de match entre mercado e vinho demonstram que o **Cluster 2**
é o perfil com maior probabilidade de sucesso na entrada no mercado do Reino Unido.

Este cluster deverá funcionar como **produto prioritário de exportação**, suportando:
- maior probabilidade de aceitação comercial
- aumento do volume de vendas
- reforço da rentabilidade da garrafeira
- menor risco na entrada num novo mercado internacional

**Mensagem-chave:** utilizar o **Cluster 2** como base da estratégia de exportação para o Reino Unido, emerge como a escolha ideal para a internacionalização. Este cluster representa um perfil de vinho intermédio, com boa harmonia entre acidez, álcool e estrutura, tendencialmente seco e com corpo médio, sem doçura pronunciada nem leveza excessiva.

Este perfil alinha-se perfeitamente com o segmento etário dominante de 30-50 anos no Reino Unido, que demonstra uma preferência clara por vinhos equilibrados e secos, com acidez e álcool médios. A descrição do Cluster 2 como vinhos sem sensação de doçura ou leveza excessiva, com corpo médio, encaixa na procura por vinhos mais sóbrios, mas com caráter.
""")