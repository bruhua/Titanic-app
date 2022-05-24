import streamlit as st
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression




# Fonction pour pré-processer les données d'entraînement
@st.cache
def nettoyage(df):
    # On garde uniquement certaines variables / on a supprimer la colonne Pclass peu impactante :
    df_clean = df[['Survived', 'Age', 'Sex', 'Embarked', 'Fare', 'df']]

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
    st.title("Embarquement immédiat !")
    st.markdown("""
    Bienvenue à bord du Titanic ! 
    \n Vous avez embarqué à bord du Titanic malheureusement vous savez déjà ce qu'il va arriver :( 
    
    Mais tout n'est pas perdu ! Vous avez (peut être) des chances de survivre à la catastrophe !
    
    Grâce au machine learning, vous pouvez tenter de déjouer le mauvais sort et d'estimer vos chances de survie.
    """)

    st.subheader('A vous de jouer !')
    st.write("Remplissez les quelques informations suivantes et découvrez ce que le destin vous réserve :")


    st.write("")
    st.write("")
    def user_input() :

        age=st.slider('Votre âge :', min_value=0 ,max_value=80, value=30, step=1)
        sexe = st.radio("Sexe :",['Homme','Femme'])
        port = st.radio("Port d'embarquement :", ['Cherbourg', 'Queenstown', 'Southampton'])
        prix = st.slider('Prix du billet :', min_value=0, max_value=512, value=33, step=1)


        data = {'Age': age,
                'Sexe': sexe,
                "Port d'embarquement": port,
                'Prix du billet': prix
                }

        titanic_parametres=pd.DataFrame(data, index=[0])
        return titanic_parametres

    df_user=user_input()

# Affichage en df des infos utilisateurs
    st.write(df_user)



# Ajout des colonnes survived et df  / Puis concatenation avec la base

    df_user=df_user.rename(columns = {'Age': 'Age', 'Sexe': 'Sex', "Port d'embarquement" : 'Embarked', 'Prix du billet':'Fare'})
    df_user.insert(0, "Survived", 'NaN', allow_duplicates=False)
    df_user.insert(5, "df", 'valid', allow_duplicates=False)
    df_clean = nettoyage(df)
    df_clean2 = pd.concat([df_clean, df_user], axis=0)

    # Nettoyage et preprocessing
    df_clean3 = nettoyage(df_clean2)
    preprocessing_df(df_clean3)

    X_train_scaled2 = preprocessing_df(df_clean3)[2]
    X_test_scaled2 = preprocessing_df(df_clean3)[3]
    y_train2 = preprocessing_df(df_clean3)[4]
    y_test2 = preprocessing_df(df_clean3)[5]
    valid_scaled2 = preprocessing_df(df_clean3)[6]

    # REG LOG
    clf_lr = LogisticRegression()
    clf_lr.fit(X_train_scaled2, y_train2)
    y_pred_test_lr = clf_lr.predict(X_test_scaled2)

    # Prediction
    resultat = clf_lr.predict(valid_scaled2[-1:])
    resultat2 = clf_lr.predict_proba(valid_scaled2[-1:])

    st.write("")
    st.write("")
    if (st.button('JE VALIDE ')):
        st.subheader("""
            Votre résultat
            """)

        if resultat2[0,1] <= 0.2 :
            st.error("Désolé, vous finissez au fond de l'eau !")
            st.write("Seulement ", round(resultat2[0,1]*100,2),"% de survie estimée par le modèle."
                    "\n Vous n'auriez pas du monter à bord.")

        elif (resultat2[0,1] > 0.2) & (resultat2[0,1] < 0.4) :
            st.warning("Vous êtes Jack : malgré tous vos efforts, vos chances de survie sont assez faibles.")
            st.write(round(resultat2[0,1]*100,2),"% de survie estimée par le modèle.")

        elif resultat2[0,1] > 0.6 :
            st.success("Vous êtes Rose : vous avez toutes les chances d'être sauvé !")
            st.write( round(resultat2[0,1]*100,2),"% de survie estimée par le modèle.")
        else :
            st.info("Croisez les doigts ! Vous êtes quasiment à 50-50. ")
            st.write(round(resultat2[0,1]*100,2),"% de survie estimée par le modèle. "
                    "\n Il y a peut être encore une place dans un canot de sauvetage ?")


        st.subheader('Envie de comprendre comment ça fonctionne ? ')
        st.write("N'hésitez pas à explorer l'app à l'aide du menu à gauche")
