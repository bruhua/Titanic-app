import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt

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
current_folder = os.path.dirname(__file__)
# Récupération du dossier der données (dataset, images, ...)
data_path = os.path.join(current_folder, "Dataframe")


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







# Variables globales

st.sidebar.title("Menu")

# Choix de la page
selection = st.sidebar.radio("", list(PAGES.keys()))
page = PAGES[selection]
page.app(df, data_path)

