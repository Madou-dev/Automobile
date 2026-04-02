import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import pearsonr, spearmanr

# ─── CONFIG ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoViz · Automobile Dataset",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── SESSION STATE : thème ────────────────────────────────────────────────────
if "light_mode" not in st.session_state:
    st.session_state.light_mode = True

# ─── PALETTES ─────────────────────────────────────────────────────────────────
DARK = {
    "app_bg":        "#0a0a0f",
    "sidebar_bg":    "#111118",
    "sidebar_border":"#2a2a3a",
    "card_bg":       "linear-gradient(135deg, #16161f 0%, #1e1e2e 100%)",
    "card_border":   "#2a2a3a",
    "text_primary":  "#e8e8f0",
    "text_secondary":"#606080",
    "text_muted":    "#a0a0c0",
    "tab_bg":        "#111118",
    "tab_border":    "#2a2a3a",
    "tab_color":     "#606080",
    "plot_bg":       "#0e0e16",
    "plot_paper":    "#0e0e16",
    "plot_font":     "#a0a0c0",
    "plot_grid":     "#1e1e2e",
    "plot_line":     "#2a2a3a",
    "legend_bg":     "#111118",
    "legend_border": "#2a2a3a",
    "df_border":     "#2a2a3a",
    "pie_line":      "#0a0a0f",
    "hist_line":     "#0a0a0f",
    "toggle_label":  "☀️ Mode Clair",
}
LIGHT = {
    "app_bg":        "#f5f5f7",
    "sidebar_bg":    "#ffffff",
    "sidebar_border":"#e0e0e8",
    "card_bg":       "linear-gradient(135deg, #ffffff 0%, #f0f0f8 100%)",
    "card_border":   "#d0d0e0",
    "text_primary":  "#1a1a2e",
    "text_secondary":"#606080",
    "text_muted":    "#505070",
    "tab_bg":        "#ffffff",
    "tab_border":    "#d0d0e0",
    "tab_color":     "#808090",
    "plot_bg":       "#ffffff",
    "plot_paper":    "#f5f5f7",
    "plot_font":     "#505070",
    "plot_grid":     "#e8e8f0",
    "plot_line":     "#d0d0e0",
    "legend_bg":     "#ffffff",
    "legend_border": "#e0e0e8",
    "df_border":     "#d0d0e0",
    "pie_line":      "#f5f5f7",
    "hist_line":     "#f5f5f7",
    "toggle_label":  "🌙 Mode Sombre",
}

T = LIGHT if st.session_state.light_mode else DARK

# ─── CSS DYNAMIQUE ────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {{ font-family: 'DM Sans', sans-serif; }}

.stApp {{
    background: {T["app_bg"]};
    color: {T["text_primary"]};
    transition: background 0.3s ease, color 0.3s ease;
}}
[data-testid="stSidebar"] {{
    background: {T["sidebar_bg"]};
    border-right: 1px solid {T["sidebar_border"]};
}}
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] label {{
    color: {T["text_muted"]} !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}}
.hero-title {{
    font-family: 'Bebas Neue', cursive;
    font-size: 4.5rem;
    letter-spacing: 0.05em;
    background: linear-gradient(135deg, {T["text_primary"]} 0%, #ff4d4d 50%, #ff9900 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    margin-bottom: 0;
}}
.hero-sub {{
    color: {T["text_secondary"]};
    font-size: 0.9rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 4px;
}}
.hero-divider {{
    height: 2px;
    background: linear-gradient(90deg, #ff4d4d, #ff9900, transparent);
    margin: 1rem 0 2rem 0;
    border: none;
}}
.kpi-card {{
    background: {T["card_bg"]};
    border: 1px solid {T["card_border"]};
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: background 0.3s ease;
}}
.kpi-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #ff4d4d, #ff9900);
}}
.kpi-label {{
    color: {T["text_secondary"]};
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.4rem;
}}
.kpi-value {{
    font-family: 'Bebas Neue', cursive;
    font-size: 2.2rem;
    color: {T["text_primary"]};
    letter-spacing: 0.04em;
    line-height: 1;
}}
.kpi-unit {{
    color: #ff9900;
    font-size: 0.85rem;
    font-weight: 500;
    margin-top: 0.2rem;
}}
.section-title {{
    font-family: 'Bebas Neue', cursive;
    font-size: 1.8rem;
    letter-spacing: 0.08em;
    color: {T["text_primary"]};
    margin: 2.5rem 0 0.5rem 0;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}}
.section-title::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, {T["card_border"]}, transparent);
    margin-left: 0.8rem;
}}
[data-testid="stDataFrame"] {{
    border: 1px solid {T["df_border"]};
    border-radius: 10px;
}}
.stTabs [data-baseweb="tab-list"] {{
    background: {T["tab_bg"]};
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid {T["tab_border"]};
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent;
    color: {T["tab_color"]};
    border-radius: 8px;
    padding: 8px 18px;
    font-size: 0.8rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}}
.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, #ff4d4d, #ff9900) !important;
    color: white !important;
}}
.stDownloadButton button {{
    background: linear-gradient(135deg, #ff4d4d, #ff9900);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    letter-spacing: 0.05em;
    padding: 0.5rem 1.2rem;
}}
.stSelectbox label {{
    color: {T["text_muted"]} !important;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}
</style>
""", unsafe_allow_html=True)

# ─── PLOTLY THEME DYNAMIQUE ───────────────────────────────────────────────────
def apply_theme(fig, title=None):
    fig.update_layout(
        plot_bgcolor  = T["plot_bg"],
        paper_bgcolor = T["plot_paper"],
        font          = dict(color=T["plot_font"], family="DM Sans"),
        colorway      = ["#ff4d4d","#ff9900","#4daaff","#a855f7","#22d3ee","#86efac"],
        title         = dict(text=title, font=dict(size=14, color=T["text_primary"]), x=0.02) if title else {},
        margin        = dict(l=20, r=20, t=40 if title else 20, b=20),
        legend        = dict(bgcolor=T["legend_bg"], bordercolor=T["legend_border"],
                             borderwidth=1, font=dict(color=T["text_primary"])),
    )
    fig.update_xaxes(gridcolor=T["plot_grid"], linecolor=T["plot_line"],
                     tickcolor=T["plot_line"], tickfont=dict(color=T["plot_font"]))
    fig.update_yaxes(gridcolor=T["plot_grid"], linecolor=T["plot_line"],
                     tickcolor=T["plot_line"], tickfont=dict(color=T["plot_font"]))
    return fig

# ─── DATA ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('data/Automobile_data.csv', na_values='?')
    df['pertes_normalisees'] = df['pertes_normalisees'].fillna(df['pertes_normalisees'].median())
    df['alesage']            = df['alesage'].fillna(df['alesage'].median())
    df['course']             = df['course'].fillna(df['course'].median())
    df['puissance']          = df['puissance'].fillna(df['puissance'].median())
    df['regime_max']         = df['regime_max'].fillna(df['regime_max'].median())
    df['prix']               = df['prix'].fillna(df['prix'].median())
    df['nombre_portes']      = df['nombre_portes'].fillna(df['nombre_portes'].mode()[0])
    return df

df = load_data()

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Apparence")
    if st.button(T["toggle_label"], use_container_width=True):
        st.session_state.light_mode = not st.session_state.light_mode
        st.rerun()

    st.markdown("---")
    st.markdown("### 🔧 Filtres")

    marques = st.multiselect(
        "Marque", options=sorted(df["marque"].unique()),
        default=sorted(df["marque"].unique())
    )
    carburant = st.multiselect(
        "Carburant", options=df["type_carburant"].unique(),
        default=df["type_carburant"].unique()
    )
    carrosserie = st.multiselect(
        "Carrosserie", options=df["type_carrosserie"].unique(),
        default=df["type_carrosserie"].unique()
    )
    prix_range = st.slider(
        "Fourchette de prix ($)",
        int(df["prix"].min()), int(df["prix"].max()),
        (int(df["prix"].min()), int(df["prix"].max()))
    )
    puissance_range = st.slider(
        "Puissance (ch)",
        int(df["puissance"].min()), int(df["puissance"].max()),
        (int(df["puissance"].min()), int(df["puissance"].max()))
    )
    st.markdown("---")
    st.download_button(
        "⬇️ Télécharger les données",
        df.to_csv(index=False),
        'Automobile_data.csv',
        use_container_width=True
    )

# ─── FILTRE ───────────────────────────────────────────────────────────────────
df_f = df[
    df["marque"].isin(marques) &
    df["type_carburant"].isin(carburant) &
    df["type_carrosserie"].isin(carrosserie) &
    df["prix"].between(*prix_range) &
    df["puissance"].between(*puissance_range)
]

# ─── HERO ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">AutoViz</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Automobile Dataset · Analyse & Exploration</p>', unsafe_allow_html=True)
st.markdown('<hr class="hero-divider">', unsafe_allow_html=True)

# ─── KPIs ─────────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
kpis = [
    (c1, "🏎️ Voitures",         f"{len(df_f)}",                               "dans la sélection"),
    (c2, "💰 Prix Moyen",        f"${df_f['prix'].mean():,.0f}",                "USD"),
    (c3, "⚡ Puissance Moy.",    f"{df_f['puissance'].mean():.0f}",             "chevaux"),
    (c4, "⛽ Conso. Ville",      f"{df_f['consommation_ville_mpg'].mean():.1f}","mpg"),
    (c5, "🛣️ Conso. Autoroute", f"{df_f['consommation_autoroute_mpg'].mean():.1f}", "mpg"),
]
for col, label, value, unit in kpis:
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-unit">{unit}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Vue d'ensemble", "💡 Analyse", "🔗 Corrélations", "📋 Données"])

# ══════════════════════════════════════════════════════════════════════
# TAB 1 — VUE D'ENSEMBLE
# ══════════════════════════════════════════════════════════════════════
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        counts = df_f["marque"].value_counts().reset_index()
        counts.columns = ["marque", "count"]
        fig = px.bar(counts, x="count", y="marque", orientation="h",
                     color="count", color_continuous_scale=["#ff4d4d","#ff9900"])
        fig.update_coloraxes(showscale=False)
        apply_theme(fig, "Voitures par marque")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.histogram(df_f, x="prix", nbins=25, color_discrete_sequence=["#ff9900"])
        fig.update_traces(marker_line_width=0.5, marker_line_color=T["hist_line"])
        apply_theme(fig, "Distribution des prix")
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        pie_data = df_f["type_carrosserie"].value_counts().reset_index()
        pie_data.columns = ["carrosserie", "count"]
        fig = px.pie(pie_data, names="carrosserie", values="count", hole=0.6,
                     color_discrete_sequence=["#ff4d4d","#ff9900","#4daaff","#a855f7","#22d3ee"])
        fig.update_traces(textposition='outside', textinfo='percent+label',
                          marker=dict(line=dict(color=T["pie_line"], width=2)))
        apply_theme(fig, "Répartition par carrosserie")
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        fig = px.box(df_f, x="type_carburant", y="prix", color="type_carburant",
                     color_discrete_sequence=["#ff4d4d","#4daaff"])
        apply_theme(fig, "Distribution des prix par carburant")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(df_f, x="puissance", y="prix", color="marque", size="poids_a_vide",
                     hover_data=["marque","type_carrosserie","nombre_cylindres"],
                     opacity=0.85, size_max=20)
    apply_theme(fig, "Puissance vs Prix  (taille = poids à vide)")
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 2 — ANALYSE
# ══════════════════════════════════════════════════════════════════════
with tab2:
    c1, c2 = st.columns(2)
    with c1:
        avg_price = df_f.groupby("marque")["prix"].mean().sort_values(ascending=True).reset_index()
        fig = px.bar(avg_price, x="prix", y="marque", orientation="h",
                     color="prix", color_continuous_scale="Oryel")
        fig.update_coloraxes(showscale=False)
        apply_theme(fig, "Prix moyen par marque")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        risk = df_f.groupby("marque")["indice_risque"].mean().sort_values().reset_index()
        colors = ["#22d3ee" if v >= 0 else "#ff4d4d" for v in risk["indice_risque"]]
        fig = go.Figure(go.Bar(x=risk["indice_risque"], y=risk["marque"],
                               orientation="h", marker_color=colors))
        apply_theme(fig, "Indice de risque moyen par marque")
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        conso = df_f.groupby("type_carrosserie")[
            ["consommation_ville_mpg","consommation_autoroute_mpg"]
        ].mean().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Ville",     x=conso["type_carrosserie"],
                             y=conso["consommation_ville_mpg"],     marker_color="#ff4d4d"))
        fig.add_trace(go.Bar(name="Autoroute", x=conso["type_carrosserie"],
                             y=conso["consommation_autoroute_mpg"], marker_color="#4daaff"))
        fig.update_layout(barmode="group")
        apply_theme(fig, "Consommation : Ville vs Autoroute par carrosserie")
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        cyl_order = ["two","three","four","five","six","eight","twelve"]
        cyl_data  = df_f.groupby("nombre_cylindres")["puissance"].median().reset_index()
        cyl_data["nombre_cylindres"] = pd.Categorical(
            cyl_data["nombre_cylindres"], categories=cyl_order, ordered=True)
        cyl_data = cyl_data.sort_values("nombre_cylindres")
        fig = px.line(cyl_data, x="nombre_cylindres", y="puissance",
                      markers=True, color_discrete_sequence=["#ff9900"])
        fig.update_traces(line_width=2.5, marker_size=8)
        apply_theme(fig, "Puissance médiane par nombre de cylindres")
        st.plotly_chart(fig, use_container_width=True)

    fig = px.violin(df_f, x="type_carrosserie", y="poids_a_vide",
                    color="type_carrosserie", box=True,
                    color_discrete_sequence=["#ff4d4d","#ff9900","#4daaff","#a855f7","#22d3ee"])
    fig.update_layout(showlegend=False)
    apply_theme(fig, "Distribution du poids à vide par carrosserie")
    st.plotly_chart(fig, use_container_width=True)

    treemap_data = df_f.groupby(["marque","type_carrosserie"])["prix"].mean().reset_index()
    fig = px.treemap(treemap_data, path=["marque","type_carrosserie"], values="prix",
                     color="prix", color_continuous_scale="Oryel",
                     hover_data={"prix":":.0f"})
    apply_theme(fig, "Prix moyen : Marque → Carrosserie")
    fig.update_traces(textfont_size=12)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 3 — CORRÉLATIONS
# ══════════════════════════════════════════════════════════════════════
with tab3:
    col_num = df_f.select_dtypes(include='number').columns.tolist()

    corr_matrix = df_f[col_num].corr()
    fig = px.imshow(corr_matrix, text_auto=".2f", aspect="auto",
                    color_continuous_scale=["#4daaff", T["plot_bg"], "#ff4d4d"],
                    zmin=-1, zmax=1)
    apply_theme(fig, "Matrice de corrélation")
    fig.update_traces(textfont_size=9)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Analyse ciblée entre deux variables</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        v1 = st.selectbox("Variable X", col_num,
                          index=col_num.index("puissance") if "puissance" in col_num else 0)
    with c2:
        v2 = st.selectbox("Variable Y", col_num,
                          index=col_num.index("prix") if "prix" in col_num else 1)

    data_clean = df_f[[v1, v2]].dropna()
    fig = px.scatter(data_clean, x=v1, y=v2,
                     trendline="ols", trendline_color_override="#ff9900",
                     color_discrete_sequence=["#4daaff"], opacity=0.75)
    apply_theme(fig, f"{v1}  vs  {v2}")
    st.plotly_chart(fig, use_container_width=True)

    pearson_r,  pearson_p  = pearsonr(data_clean[v1], data_clean[v2])
    spearman_r, spearman_p = spearmanr(data_clean[v1], data_clean[v2])

    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Corrélation de Pearson</div>
            <div class="kpi-value">{pearson_r:.4f}</div>
            <div class="kpi-unit">p-value : {pearson_p:.4f} — {'✅ significative' if pearson_p < 0.05 else '⚠️ non significative'}</div>
        </div>""", unsafe_allow_html=True)
    with mc2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Corrélation de Spearman</div>
            <div class="kpi-value">{spearman_r:.4f}</div>
            <div class="kpi-unit">p-value : {spearman_p:.4f} — {'✅ significative' if spearman_p < 0.05 else '⚠️ non significative'}</div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 4 — DONNÉES BRUTES
# ══════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown(f"**{len(df_f)} voitures** correspondent aux filtres sélectionnés.")
    st.dataframe(df_f.reset_index(drop=True), use_container_width=True, height=500)
