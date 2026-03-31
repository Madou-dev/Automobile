import pandas as pd 
import numpy as np
import plotly.express as px
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
df = pd.read_csv("/Users/alphaamadoudiallo/Desktop/data_analysis/data/Automobile_data.csv")
df = df.replace("?", np.nan)

df.head(50)

df.info() 
df.shape
df.describe(include = 'all')
df.isna().sum() 

df = df[df.duplicated(keep=False)]
print(f"Doublons : {df.duplicated().sum()}")

df = pd.read_csv("/Users/alphaamadoudiallo/Desktop/data_analysis/data/Automobile_data.csv", na_values="?")
for col in df.columns:
    if df[col].dtypes == 'object':
        print(f'{df[col]} : {df[col].unique()}') #Afficher les valeurs uniques
        print(f'{df[col]} : {df[col].value_counts()}') #Afficher le nombre de valeurs uniques

df['pertes_normalisees'] = df["pertes_normalisees"].fillna(df["pertes_normalisees"].median())

df['alesage'] = df['alesage'].fillna(df['alesage'].median())

df['course'] = df['course'].fillna(df['course'].median())

df['puissance'] = df['puissance'].fillna(df['course'].median())

df['regime_max'] = df['regime_max'].fillna(df['regime_max'].median())

df['prix'] = df['prix'].fillna(df['prix'].median())

#Calcul du mode
mode_nombre_portes = df['nombre_portes'].mode()[0]

df['nombre_portes'] = df['nombre_portes'].fillna(mode_nombre_portes)

col_num = df.select_dtypes(include=np.number).columns
for col in col_num:
    fig = px.box(df, y = col)
    print(col)
    fig.show()


#Calcul du upper bound et du lower bound

for col in col_num:
    Q1 = np.percentile(df[col],25)
    Q3 = np.percentile(df[col],75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

#moyenne et ecart type pour les colonnes numeriques
#methode du z_score 

for col in col_num:
    moyenne = np.mean(df[col])
    ecart_type = np.std(df[col])
    threshold = 3
    z_score = (df[col] - moyenne) / ecart_type
    z_score_outliers = np.abs(z_score) > threshold


fig1 = px.box(df, x='type_carrosserie', y = 'prix', color = 'type_carrosserie', title= " Prix en fonction du type de carrosserie")
fig1.show()

fig2 = px.box(df, x='marque', y='prix', color='marque', title= " Prix en fonction des marques")
fig2.show()

fig3 = px.box(df, x='marque', y='puissance', color='marque', title= " Puissance en fonction des marques")
fig3.show()

fig4 = px.box(df, x='consommation_ville_mpg', y='marque', color='marque', title= "Les marques qui consomment le plus en ville")
fig4.show()

fig5 = px.box(df, x='consommation_autoroute_mpg', y='marque', color='marque', title= "Les marques qui consomment le plus en autoroute")
fig5.show()

sns.catplot(df, x="consommation_ville_mpg", y="marque", kind="swarm")

sns.catplot(df, x="consommation_autoroute_mpg", y="marque", kind="swarm")

#Correlation et carte thermique
correlation = df[col_num].corr()
fig6 = px.imshow(correlation, text_auto=True, title='Correlation Heatmap')
fig6.show()

correlation, p_value = pearsonr(df['hauteur'], df['taux_compression'])
print(f"La correlation entre hauteur et taux de compression est de {correlation} avec un p_value de {p_value}")

if p_value < 0.05:
    print("Il existe une correlation statistiquement significative entre hauteur et taux de compression")
else:
    print("Il n'existe pas de correlation statistiquement significative entre hauteur et taux de compression")
    
correlation, p_value = spearmanr(df['hauteur'], df['taux_compression'])
print(f"La correlation entre hateur et taux de compression est de {correlation} avec p_value de {p_value}")

if p_value < 0.05:
    print("Il existe une correlation statistiquement significative entre hauteur et taux de compression")
else:
    print("Il n'existe pas de correlation statistiquement significative entre hauteur et taux de compression")
    



