import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump, load

import modeles_titanic
import intro_titanic
import visu_titanic
import demo_titanic

# Variables globales
PAGES = {
    "Je monte à bord !": demo_titanic,
    "Présentation du sujet": intro_titanic,
    "Visualisation des données": visu_titanic,
    "Les modèles" : modeles_titanic


}

# Gestion des chemins

# Récupération du dossier courant
# current_folder = os.path.dirname(__file__)
# Récupération du dossier der données (dataset, images, ...)
# data_path = os.path.join(current_folder, "Dataframe")


# Fonction pour charger les données
@st.cache
def load_data():
    train = pd.read_csv('https://raw.githubusercontent.com/bruhua/Titanic-app/main/Dataframe/train.csv',
                        sep=',', header=0, index_col=0, error_bad_lines=False)

    valid = pd.read_csv('https://raw.githubusercontent.com/bruhua/Titanic-app/main/Dataframe/test.csv',
                        sep=',', header=0, index_col=0, error_bad_lines=False)
    train['df'] = 'train'
    valid['df'] = 'valid'
    df = pd.concat([train, valid], axis=0)
    return df


# Chargement des données
df = load_data()


# Chargement des modeles pour la partie demo
logistic_reg = load('logistic_reg.joblib')




# Variables globales

st.sidebar.title("Titanic - Prédiction de survie des passagers")

# Choix de la page
selection = st.sidebar.radio("Menu", list(PAGES.keys()),key=range(0,4) )
page = PAGES[selection]
page.app(df ) #, data_path)

st.sidebar.markdown("""<hr style="height:2px;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
st.sidebar.markdown("Auteur :")
st.sidebar.markdown("[Bruno Huart](https://www.linkedin.com/in/bruno-huart-051459107/) ")
st.sidebar.markdown("""<hr style="height:2px;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

