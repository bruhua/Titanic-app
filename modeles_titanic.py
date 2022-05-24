import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
from sklearn.ensemble import GradientBoostingClassifier


# Fonction pour pré-processer les données d'entraînement
@st.cache
def nettoyage(df):
    # On garde uniquement certaines variables :
    df_clean = df[['Survived', 'Age', 'Sex', 'Embarked', 'Fare', 'Pclass', 'SibSp','Parch','df']]

    # Remplacement pour la colonne embarked
    df_clean['Embarked'] = df_clean['Embarked'].replace(['Cherbourg', 'Queenstown', 'Southampton'], ['C', 'Q', 'S'])
    df_clean['Sex'] = df_clean['Sex'].replace(['Homme', 'Femme'], ['male', 'female'])

    # Remplacements vides
    df_clean['Age'].replace({np.NaN: df_clean['Age'].median()}, inplace=True)
    df_clean['Embarked'].replace({np.NaN: df_clean['Embarked'].mode()[0]}, inplace=True)
    df_clean['Fare'].replace({np.NaN: df_clean['Fare'].median()}, inplace=True)

    df_clean['Survived'] = df_clean['Survived'].astype('float64')

    return df_clean




def preprocessing_df(df):
    # Get dummy
    df_discret = pd.get_dummies(data=df)

    # Séparation des df
    train2 = df_discret[df_discret['df_train'] == 1]
    valid2 = df_discret[df_discret['df_valid'] == 1]

    # Nettoyage des colonnes devenues inutiles
    train2 = train2.drop(['df_train', 'df_valid'], axis=1)
    valid2 = valid2.drop(['df_train', 'df_valid', 'Survived'], axis=1)

    # On crée un dataframe target avec uniquement la variable qu'on cherche à prédire et un dataframe features avec les variables explicatives

    target = train2['Survived']
    features = train2.drop('Survived', axis=1)

    # Split

    X_train, X_test, y_train, y_test = train_test_split(features, target, stratify=target, test_size=0.2,
                                                        random_state=50)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    valid_scaled = scaler.transform(valid2)

    return df, X_train, X_train_scaled, X_test_scaled, y_train, y_test, valid_scaled



def app(df) : #, data_path):
    st.title("Les modèles")
    st.markdown("""
    Pour prédire la survie des passagers, 4 modèles sont testés. """)
    st.markdown("""Leur fonctionnement est différent mais leur objectif est toujours le même : bien classer les individus et donc bien prédire ce 
    qu'il leur est arrivé.
    """)
    df_clean = nettoyage(df)

    X_train_scaled = preprocessing_df(df_clean)[2]
    X_test_scaled = preprocessing_df(df_clean)[3]
    y_train = preprocessing_df(df_clean)[4]
    y_test = preprocessing_df(df_clean)[5]

    st.subheader('Résultat avec une régression logistique')
    st.write("Déf : Ce modèle se rapproche d'une régression linéaire classique (y = a + bx...) si ce n'est qu'il permet de prédire la probabilité "
             "qu'un évenèment arrive (1) ou non (0)")
    # REG LOG
    clf_lr= LogisticRegression()
    clf_lr.fit(X_train_scaled,y_train)
    y_pred_test_lr = clf_lr.predict(X_test_scaled)
    resultat_log_reg = round(clf_lr.score(X_test_scaled, y_test),3)
    st.write("==> Le score avec ce modèle est de :", resultat_log_reg, ".")
    st.write("Cela veut dire que le modèle a réalisé", round(resultat_log_reg*100,3),"% de bonnes prédictions ")

    st.success("==> C'est ce modèle qui est utilisé dans la démonstration")


    st.write("")
    st.write("")
    st.write("")



    # RF
    st.subheader('Résultat avec un random forest')
    st.write("Déf : Ce modèle est basé sur le même principe qu'un arbre de décision sauf qu'au lieu d'avoir un seul arbre (qui a davantage de chance de se tromper), on créé une forêt remplies d'arbres (par défaut 100)."
            "Les prédictions sont donc beaucoup plus robustes et fiables")

    rf = ensemble.RandomForestClassifier()
    rf.fit(X_train_scaled, y_train)
    y_pred_test_rf = rf.predict(X_test_scaled)
    resultat_rf = round(rf.score(X_test_scaled, y_test), 3)
    st.write("==> Le score avec ce modèle est de :", resultat_rf, ".")
    st.write("Le random forest a réalisé", round(resultat_rf*100,3) ,"% de bonnes prédictions")


    st.write("")
    st.write("")
    st.write("")

    # KNN
    st.subheader('Résultat avec un KNN')
    st.write("""Déf : Le KNN (pour k-nearest neighbors ou méthode des plus proches voisins) regroupe les individus ayant des points communs.
            On est un peu dans le principe du "qui se ressemble s'assemble".""")

    knn =  neighbors.KNeighborsClassifier()
    knn.fit(X_train_scaled, y_train)
    y_pred_test_knn = knn.predict(X_test_scaled)
    resultat_knn = round(knn.score(X_test_scaled, y_test), 3)
    st.write("==> Le score avec ce modèle est de :", resultat_knn, ".")
    st.write("Le KNN a réalisé", round(resultat_knn*100,3) ,"% de bonnes prédictions")

    st.write("")
    st.write("")
    st.write("")

    # GBC
    st.subheader('Résultat avec un Gradient Boosting Classifier')
    st.write("Déf : Le GBC repose également sur des arbres sauf qu'il va augmenter le poids observations des difficiles à classer."
            "\n Le but est d'améliorer au fur et à mesure les arbres")
    clf = GradientBoostingClassifier()
    clf.fit(X_train_scaled, y_train)
    y_pred_test_clf = clf.predict(X_test_scaled)
    resultat_clf = round(clf.score(X_test_scaled, y_test), 3)
    st.write("==> Le score avec ce modèle est de :", resultat_clf, ".")
    st.write("Le GBC a réalisé", round(resultat_clf*100,3) ,"% de bonnes prédictions")

    st.write("")
    st.write("")
    st.write("")

    st.subheader('Comparaison des performances des modèles')
    st.write("Le graphique ci-dessous compare les performances obtenues par nos 4 modèles : les résultats sont assez proches ")



    # Creation d'un dataframe avec l'ensemble des resultats
    result_perf_modele = pd.DataFrame({'Résultat Reg Log': resultat_log_reg,
                                       'Résultat RF': resultat_rf,
                                       'Résultat KNN': resultat_knn,
                                       'Résultat GBC': resultat_clf
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
