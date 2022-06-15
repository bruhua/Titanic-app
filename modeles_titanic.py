import streamlit as st
import os
import pandas as pd

import matplotlib.pyplot as plt





def app(df):
    st.title("Les modèles")
    st.markdown("""
    Pour prédire la survie des passagers, 4 modèles sont testés. """)
    st.markdown("""Leur fonctionnement est différent mais leur objectif est toujours le même : bien classer les individus et donc bien prédire ce 
    qu'il leur est arrivé.
    """)




    st.subheader('Résultat avec une régression logistique')
    st.write("Déf : Ce modèle se rapproche d'une régression linéaire classique (y = a + bx...) si ce n'est qu'il permet de prédire la probabilité "
             "qu'un évenèment arrive (1) ou non (0)")

    st.write("==> Le score avec ce modèle est de : 0.821")
    st.write("Cela veut dire que le modèle a réalisé 82.1% de bonnes prédictions ")

    st.success("==> C'est ce modèle qui est utilisé dans la démonstration")


    st.write("")
    st.write("")
    st.write("")



    # RF
    st.subheader('Résultat avec un random forest')
    st.write("Déf : Ce modèle est basé sur le même principe qu'un arbre de décision sauf qu'au lieu d'avoir un seul arbre (qui a davantage de chance de se tromper), on créé une forêt remplies d'arbres (par défaut 100)."
            "Les prédictions sont donc beaucoup plus robustes et fiables")


    st.write("==> Le score avec ce modèle est de 0.838")
    st.write("Le random forest a réalisé 83.8% de bonnes prédictions")


    st.write("")
    st.write("")
    st.write("")

    # KNN
    st.subheader('Résultat avec un KNN')
    st.write("""Déf : Le KNN (pour k-nearest neighbors ou méthode des plus proches voisins) regroupe les individus ayant des points communs.
            On est un peu dans le principe du "qui se ressemble s'assemble".""")


    st.write("==> Le score avec ce modèle est de : 0.838")
    st.write("Le KNN a réalisé 83.8% de bonnes prédictions")

    st.write("")
    st.write("")
    st.write("")

    # GBC
    st.subheader('Résultat avec un Gradient Boosting Classifier')
    st.write("Déf : Le GBC repose également sur des arbres sauf qu'il va augmenter le poids observations des difficiles à classer."
            "\n Le but est d'améliorer au fur et à mesure les arbres")

    st.write("==> Le score avec ce modèle est de 0.815")
    st.write("Le GBC a réalisé 81.5% de bonnes prédictions")

    st.write("")
    st.write("")
    st.write("")

    st.subheader('Comparaison des performances des modèles')
    st.write("Le graphique ci-dessous compare les performances obtenues par nos 4 modèles : les résultats sont assez proches ")



    # Creation d'un dataframe avec l'ensemble des resultats
    result_perf_modele = pd.DataFrame({'Résultat Reg Log': 0.821,
                                       'Résultat RF': 0.838,
                                       'Résultat KNN': 0.838,
                                       'Résultat GBC': 0.815
                                       },
                                      index=[0],
                                      columns=['Résultat Reg Log', 'Résultat KNN', 'Résultat RF', 'Résultat GBC'])

    # Graphique pour comparer les résultats
    barWidth = 0.7
    x = range(1, 5)

    y1 = result_perf_modele['Résultat Reg Log']
    y2 = result_perf_modele['Résultat KNN']
    y3 = result_perf_modele['Résultat RF']
    y4 = result_perf_modele['Résultat GBC']

    fig_comparaison, ax = plt.subplots(figsize=(5, 3))
    plt.title("Comparaison des performances des modèles")
    plt.bar(x[0], y1, label='Résultat Régression logistique', width=barWidth)
    plt.bar(x[1], y2, label='Résultat KNN', width=barWidth)
    plt.bar(x[2], y3, label='Résultat Random Forest', width=barWidth)
    plt.bar(x[3], y4, label="Résultat GBC", width=barWidth)

    plt.ylim(0, 1)
    plt.ylabel('%')
    plt.xticks(x, ['Régression Log', 'KNN', 'Random Forest', 'GBC'])

    for i in range(len(y1)):
        plt.annotate(str(y1[i]), xy=(x[0] - 0.1, y1[i]))

    for i in range(len(y2)):
        plt.annotate(str(y2[i]), xy=(x[1] - 0.1, y2[i]))

    for i in range(len(y3)):
        plt.annotate(str(y3[i]), xy=(x[2] - 0.1, y3[i]))

    for i in range(len(y4)):
        plt.annotate(str(y4[i]), xy=(x[3] - 0.1, y4[i]))

    st.pyplot(fig_comparaison)
