import streamlit as st
import pandas as pd
import seaborn as sns
sns.set(font_scale = 0.7)
import numpy as np
import matplotlib.pyplot as plt


# Page de visualisation des données
def app(df) : #, data_path):
    st.title("Visualisation des données")
    st.markdown("""

    """)

    st.markdown("""
    La visualisation des données permet de connaitre la répartition des données, les valeurs prises ainsi que les valeurs extrêmes s’il y en a. 
    C’est également l’occasion de créer des visualisations permettant d’établir des liens entre nos variables et la variable cible : la survie.

    """)

    st.write("")
    st.subheader('Répartition des survivants')
    st.write("38% de survivants")
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.countplot(x='Survived',  data=df, ax=ax)
    ax.set(title = "Variable survie")
    st.pyplot(fig)

    st.write("")
    st.write("")
    st.write("")
    st.subheader('Répartition des classes')
    st.write("On observe qu'il y a beaucoup plus de passagers en 3e classe...")
    fig2, ax = plt.subplots(figsize=(5, 3))
    sns.countplot(x='Pclass',  data=df, ax=ax)
    ax.set(title = "Variable Classe")
    st.pyplot(fig2)



    st.subheader('Croisement entre la survie et les classes')
    st.write("...mais les chances de survie sont plus élevées dans les 1eres et 2e classes")
    fig3, ax = plt.subplots(figsize=(5, 3))
    sns.countplot(x='Pclass', hue='Survived', data=df)
    ax.set(title = "Croisement entre classe et survie")
    st.pyplot(fig3)




    st.write("")
    st.write("")
    st.write("")
    st.subheader('Répartition des genres')
    fig2, ax = plt.subplots(figsize=(5, 3))
    sns.countplot(x='Sex',  data=df, ax=ax)
    ax.set(title = "Variable sex")
    st.pyplot(fig2)

    st.subheader('Croisement entre la survie et le genre')
    st.write("Beaucoup plus d'hommes sur le Titanic mais plus de chances de survivre pour les femmes")
    fig3, ax = plt.subplots(figsize=(5, 3))
    sns.countplot(x='Sex', hue='Survived', data=df)
    ax.set(title = "Croisement entre genre et survie")
    st.pyplot(fig3)

