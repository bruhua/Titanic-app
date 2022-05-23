import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os


# Page d'introduction
def app(df, data_path):
    st.title("Titanic - Prédiction de survie des passagers")

    st.subheader("Introduction")
    st.markdown("""
    Voici un projet de machine learning classique pour toute personne débutant dans la datascience.
    L'objectif est de prédire la survie des passagers du Titanic en fonction des informations dont on dispose 
    sur eux. 
    
    """)

    st.subheader('La méthodologie')
    st.markdown("""
    En machine learning, nos modèles vont s'entrainer sur une partie des données seulement. 
    \n Ensuite, nous fournissons au modèle des données qu'il n'a jamais vu : le modèle doit donc réussir à obtenir de 
    bonnes prédictions sur ces nouvelles données.
    \n Dans le cas du Titanic, une bonne prédiction correspond à prédire la survie si le passager a effectivement survécu (et 
    le décès si le passager n'a pas survécu)

    """)


    st.subheader('Le jeu de données')
    st.markdown("""
     Pour chaque passager, nous possédons différentes données : 
     - La survie : 1 ou 0
     - La classe : 1er/2e/3e
     - Le nom
     - Le sexe
     - L'âge
     - Le nombre de frères/soeurs ou conjoints
     - Le nombre d'enfants ou de parents
     - Le prix de leur billet
     - Le numéro de cabine
     - Le port d'embarquement : Cherbourg / Queenstown / Southampton

     """)
    st.write(df.drop('df',axis=1).head(10))


