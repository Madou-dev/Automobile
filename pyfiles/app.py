import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
st.set_page_config(
    page_title="Automobile dataset",
    page_icon="🏎️",
    layout="wide"
)

st.header('Automobile dataset')

@st.cache_data
def load_data ():
    df =  pd.read_csv('/Users/alphaamadoudiallo/Desktop/data_analysis/data/Automobile_data.csv', na_values='?')
    df = df.replace("?", np.nan)
    df = pd.read_csv("/Users/alphaamadoudiallo/Desktop/data_analysis/data/Automobile_data.csv", na_values="?")
    for col in df.columns:
        if df[col].dtypes == 'str':
            print(f'{df[col]} : {df[col].unique()}')
            print(f'{df[col]} : {df[col].value_counts()}')

    df['pertes_normalisees'] = df["pertes_normalisees"].fillna(df["pertes_normalisees"].median())

    df['alesage'] = df['alesage'].fillna(df['alesage'].median())

    df['course'] = df['course'].fillna(df['course'].median())

    df['puissance'] = df['puissance'].fillna(df['course'].median())

    df['regime_max'] = df['regime_max'].fillna(df['regime_max'].median())

    df['prix'] = df['prix'].fillna(df['prix'].median())

    mode_nombre_portes = df['nombre_portes'].mode()[0]

    df['nombre_portes'] = df['nombre_portes'].fillna(mode_nombre_portes)

    return df
    
df = load_data()
df['nombre_portes'].unique()
df.isna().sum()

#st.dataframe(df)

# Just add it after st.sidebar:
# a = st.sidebar.radio('Choose:',[1,2,3])
# if a == 1 :
#     st.dataframe(df)
# elif a == 2:
#     st.multiselect("Selectionnez les marques", df["marque"].unique(), default="bmw")
# else:
#     pass

st.download_button('Download the dataset', df.to_csv(index=False), 'Automobile_data.csv')

st.markdown("Explorez les données du dataset de voitures issues de différentes marques.")


st.sidebar.header('Filtres')
marques =st.sidebar.multiselect(
    label = "Marque Voiture",
    options=df["marque"].unique(),
    default=df["marque"].unique()
)
carburant = st.sidebar.multiselect(
    label="Type de carburant",
    options=df["type_carburant"].unique(),
    default=df["type_carburant"].unique()
)

carrosserie = st.sidebar.multiselect(
    label="Type de carrosserie",
    options=df["type_carrosserie"].unique(),
    default=df["type_carrosserie"].unique()
)

prix = st.sidebar.multiselect(
    label="Prix",
    options=df["prix"].unique(),
    default=df["prix"].unique()
)
nombre_cylindres = st.sidebar.multiselect(
    label="Nombre de cylindre",
    options=df["nombre_cylindres"].unique(),
    default=df["nombre_cylindres"].unique()
)

aspiration = st.sidebar.multiselect(
    label="Aspiration",
    options=df["aspiration"].unique(),
    default=df["aspiration"].unique()
)

nombre_portes = st.sidebar.multiselect(
    label="Nombre de portes",
    options=df["nombre_portes"].unique(),
    default=df["nombre_portes"].unique()
)

roues_motrices = st.sidebar.multiselect(
     label="Roues motrices",
    options=df["roues_motrices"].unique(),
    default=df["roues_motrices"].unique()
)

emplacement_moteur = st.sidebar.multiselect(
    label="Emplacement moteur",
    options=df["emplacement_moteur"].unique(),
    default=df["emplacement_moteur"].unique()
)

empattement = st.sidebar.multiselect(
    label="Empattement",
    options=df["empattement"].unique(),
    default=df["empattement"].unique()
)

longueur = st.sidebar.multiselect(
    label="Longueur",
    options=df["longueur"].unique(),
    default=df["longueur"].unique()
)

largeur = st.sidebar.multiselect(
    label="Largeur",
    options=df["largeur"].unique(),
    default=df["largeur"].unique()
)

hauteur = st.sidebar.multiselect(
    label="Hauteur",
    options=df["hauteur"].unique(),
    default=df["hauteur"].unique()
)

poids_a_vide = st.sidebar.multiselect(
    label="Poids a vide",
    options=df["poids_a_vide"].unique(),
    default=df["poids_a_vide"].unique()
)

type_moteur = st.sidebar.multiselect(
    label="Type de moteur",
    options=df["type_moteur"].unique(),
    default=df["type_moteur"].unique()
)

taille_moteur = st.sidebar.multiselect(
    label="Taille moteur",
    options=df["taille_moteur"].unique(),
    default=df["taille_moteur"].unique()
)

systeme_carburant = st.sidebar.multiselect(
    label="Systeme de carburant",
    options=df["systeme_carburant"].unique(),
    default=df["systeme_carburant"].unique()
)

alesage = st.sidebar.multiselect(
    label="Alesage",
    options=df["alesage"].unique(),
    default=df["alesage"].unique()
)

course = st.sidebar.multiselect(
    label="Course",
    options=df["course"].unique(),
    default=df["course"].unique()
)

puissance = st.sidebar.multiselect(
    label="Puissance",
    options=df["puissance"].unique(),
    default=df["puissance"].unique()
)

regime_max = st.sidebar.multiselect(
    label="Regime max",
    options=df["regime_max"].unique(),
    default=df["regime_max"].unique()
)

pertes_normalisees = st.sidebar.multiselect(
    label="Pertes normalisees",
    options=df["pertes_normalisees"].unique(),
    default=df["pertes_normalisees"].unique()
)

consommation_ville_mpg = st.sidebar.multiselect(
    label="Consommation ville mpg",
    options=df["consommation_ville_mpg"].unique(),
    default=df["consommation_ville_mpg"].unique()
)

consommation_autoroute_mpg = st.sidebar.multiselect(
    label="Consommation autoroute mpg",
    options=df["consommation_autoroute_mpg"].unique(),
    default=df["consommation_autoroute_mpg"].unique()
)

taux_compression = st.sidebar.multiselect(
    label="Taux de compression",
    options=df["taux_compression"].unique(),
    default=df["taux_compression"].unique()
)

indice_risque = st.sidebar.multiselect(
    label="Indice de risque",
    options=df["indice_risque"].unique(),
    default=df["indice_risque"].unique()
)

df_filtre = df[
    (df["marque"].isin(marques)) &
    (df["type_carburant"].isin(carburant)) &
    (df["type_carrosserie"].isin(carrosserie))&
    (df["prix"].isin(prix))&
    (df["nombre_cylindres"].isin(nombre_cylindres))&
    (df["aspiration"].isin(aspiration))&
    (df["nombre_portes"].isin(nombre_portes))&
    (df["roues_motrices"].isin(roues_motrices))&
    (df["emplacement_moteur"].isin(emplacement_moteur))&
    (df["empattement"].isin(empattement))&
    (df["longueur"].isin(longueur))&
    (df["largeur"].isin(largeur))&
    (df["hauteur"].isin(hauteur))&
    (df["poids_a_vide"].isin(poids_a_vide))&
    (df["type_moteur"].isin(type_moteur))&
    (df["taille_moteur"].isin(taille_moteur))&
    (df["systeme_carburant"].isin(systeme_carburant))&
    (df["alesage"].isin(alesage))&
    (df["course"].isin(course))&
    (df["puissance"].isin(puissance))&
    (df["regime_max"].isin(regime_max))&
    (df["pertes_normalisees"].isin(pertes_normalisees))&
    (df["consommation_ville_mpg"].isin(consommation_ville_mpg))&
    (df["consommation_autoroute_mpg"].isin(consommation_autoroute_mpg))&
    (df["taux_compression"].isin(taux_compression))&
    (df["indice_risque"].isin(indice_risque))

]


col1, col2, col3, col4 = st.columns(4)
col1.metric("🏎️ Voiture", len(df_filtre))
col2.metric("Prix Moyen", f"{df_filtre['prix'].mean(): ,.0f}$")
col3.metric("Puissance Moyenne", f"{df_filtre['puissance'].mean() : .0f} ch")
col4.metric("Consommation ville en moy", f"{df_filtre['consommation_ville_mpg'].mean() : .1f} mpg")

# st.divider()

# --- TABLEAU ---
st.subheader("Donnees filtrees")

st.dataframe(df_filtre, use_container_width=True)

st.divider()

# --- GRAPHIQUES ---
st.subheader("Visualisation")
col1, col2 = st.columns(2)

with col1:
        st.markdown("**Nombre de Voitures par marque**")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.patch.set_facecolor("#000000")
        ax.set_facecolor("#838383")

        df_filtre["marque"].value_counts().plot(kind="bar", ax=ax, color="mediumseagreen")
        ax.set_xlabel("Marque")
        ax.set_ylabel("Nombre")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)

with col2:
   
    st.markdown("**Distribution des prix**")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax.patch.set_facecolor("#000000")
    ax.set_facecolor("#838383")
    ax2.hist(df_filtre["prix"].dropna(), bins=20, color="coral", edgecolor="white")
    
    ax2.set_xlabel("Prix ($)")
    ax2.set_ylabel("Frequence")
    plt.tight_layout()
    st.pyplot(fig2)

col3, col4 = st.columns(2)

with col3:
    st.markdown("**Prix moyen par marque**")
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    df_filtre.groupby("marque")["prix"].mean().sort_values().plot(kind="barh", ax=ax3, color="mediumseagreen")
    ax3.set_xlabel("Prix Moyen ($)")
    plt.tight_layout()
    st.pyplot(fig3)

with col4:
    st.markdown("**Puissance VS Prix**")
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    ax4.scatter(df_filtre["puissance"],df_filtre["prix"],alpha=0.6, color="mediumpurple")
    ax4.set_xlabel("Puissance (ch)")
    ax4.set_ylabel("Prix ($)")
    plt.tight_layout()
    st.pyplot(fig4)



# Couleur de fond souhaitee
bg_color = "#000000" 

#box plot pour la visualisation des outliers
df = df.sort_values('marque', ascending=False)

fig0 = px.bar(df, 
             x='marque', 
             y='indice_risque', 
             color='indice_risque', 
             title="Indice de risque moyen par marque",
             color_continuous_scale='RdYlGn_r') 


fig0.update_yaxes(
    tickmode = 'linear',
    tick0 = -3,
    dtick = 1,
    range = [-3.5, 3.5] 
)

st.plotly_chart(fig0, use_container_width=True)



st.markdown("---")



st.title("Analyse de correlation")

#Selection des colonnes numeriques 
col_num = df.select_dtypes(include=['number']).columns

#Heatmap de correlation
st.subheader("Carte de chaleur des correlations")

correlation_matrix = df[col_num].corr()

fig = px.imshow(
     correlation_matrix,
     text_auto=True,
     title="Correlation Heatmap"
)

st.plotly_chart(fig, use_container_width=True)

# Analyse ciblee entre 2 variables
st.subheader("Analyse de correlation entre deux variables")

col1 = st.selectbox("Choisir la premiere variable", col_num, index=list(col_num).index("hauteur") if "hauteur" in col_num else 0)
col2 = st.selectbox("Choisir la deuxieme variable", col_num, index=list(col_num).index("taux_compression") if "taux_compression" in col_num else 1)

#Nettoyage 
data = df[[col1, col2]].dropna()

#Pearson
pearson_corr, pearson_p = pearsonr(data[col1], data[col2])

st.write(f"**Correlation de Pearson** : {pearson_corr:.4f}")
st.write(f"**p-value** : {pearson_p:.4f}")

if pearson_p < 0.05:
    st.success("Correlation statistiquement significative (Pearson)")
else:
    st.warning("Pas de correlation significative (Pearson)")

#Spearman
spearman_corr, spearman_p = spearmanr(data[col1], data[col2])

st.write(f"**Correlation de Spearman** : {spearman_corr:.4f}")
st.write(f"**p-value** : {spearman_p:.4f}")

if spearman_p < 0.05:
    st.success("Correlation statistiquement significative (Spearman)")
else:
    st.warning("Pas de correlation significative (Spearman)")
# st.write(df.columns.to_list())
# st.write(df.head())
