# Importation des bibliothèques
import streamlit as st
import pandas as pd
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

# Bibliotheque Pour la regression
from pycaret.regression import setup as setup_reg
from pycaret.regression import compare_models as compare_models_reg
from pycaret.regression import save_model as save_model_reg
from pycaret.regression import plot_model as plot_model_reg

# Bibliotheque Pour la regression
from pycaret.classification import setup as setup_class
from pycaret.classification import compare_models as compare_models_class
from pycaret.classification import save_model as save_model_class
from pycaret.classification import plot_model as plot_model_class

# Caching pour enregistré les données en memoir
@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    return data

# Fonction pour l'Application
def main():
    
    # Infos sur l'app
    st.title('CedricAutoML')
    st.sidebar.write("[Author: Cedric Anderson ](%s)")
    st.sidebar.markdown(
        "***Cette application Web est un outil no code pour l'analyse exploratoire des données et la création d'un modèle d'apprentissage automatique pour R \n"
        "1. Telechargez votre DataSet au format CSV; \n"
        "2. Cliquez sur le bouton *Profile Dataset* afin de générer le profilage pandas du jeu de données; \n"
        "3. Choisissez votre colonne cible; \n"
        "4. Choisissez la tâche d'apprentissage automatique (Régression ou Classification); \n"
        "5. Cliquez sur *Exécuter la modélisation* pour démarrer le processus de formation. \n"
        "\n"
        "Lorsque le modèle est construit, vous pouvez afficher les résultats comme le modèle de pipeline, le tracé des résidus, la courbe ROC,\n. Téléchargez le modèle Pipeline sur votre ordinateur local."
    )

# Permetre a l'utilisateur de chargez ses données    
file = st.file_uploader("Veuillez téléchargez le fichier csv pour les prédictions", type=["csv"])
    
if file is not None:
    data = load_data(file)
    st.dataframe(data.head())
    
    # Création du bouton de prediction
    if st.button('Profile DataSet'):
        profile_df = data.profile_report()
        st_profile_report(profile_df)
    
    # Permetre a l'utilisateur de selectionner ses variable pour la prediction    
    target = st.selectbox("selectionne les variables cibles", data.columns)
    task = st.selectbox("selectionnez la tache ML", ["Regression", "lassification"])
    data = data.dropna(subset = [target])
    
    # Si l'utilisateur choisi regression...
    if task == "Regression":
        if st.button("Run Modelling"):
            exo_reg = setup_reg(data, target=target)
            model_reg = compare_models_reg()
            save_model_reg(model_reg, "best_reg_model")
            st.success("Model de regression executer avec succes !")
            
            # Afficher le graphique des residus
            st.write("Residue")
            plot_model_reg(model_reg, plot = "residuals", save=True)
            st.image("Residuals.png")
            
            # Afficher le graphique des residus
            st.write("feature importantce")
            plot_model_reg(model_reg, plot = "feature", save=True)
            st.image("feature importantce.png")
            
            # Données la possibilité a l'utilisateur de telecharger le meilleur model
            with open('best_reg_model.pkl', 'rb') as f:
                st.download_button('Download Peppline model', f, file_name='best_reg_model.pkl')
                
                
    
    # Si l'utilisateur choisi classification...
    if task == "Classification":
        if st.button("Run Modelling"):
            exo_class = setup_class(data, target=target)
            model_class = compare_models_class()
            save_model_class(model_class, "best_reg_model")
            st.success("Model de classification executer avec succes !")
            
            #Afficher les resultats sur deux colonnes
            col5, col6 = st.columns(2)
            with col5:
                st.write('ROC Cuve')
                plot_model_class(model_class, save=True)
                st.image("AUC.png")
                
            with col6:
                st.write('Classification Report')
                plot_model_class(model_class, plot = 'class_report', save=True)
                st.image("Class Report.png")
                
            col7, col8 = st.columns(2)
            with col7:
                st.write('Confusion Matrix')
                plot_model_class(model_class, plot = 'Confusion Matrix', save=True)
                st.image("Confusion Matrix.png")
                
            with col8:
                st.write('Feature Importantce')
                plot_model_class(model_class, plot = 'feature', save=True)
                st.image("Feature Importantce.png")
                
            # Telechargement du model
            with open('best_class_model.pkl', 'rb') as f:
                st.download_button('Download Model', f, file_name= 'best_class_model.pkl')
                
else:
    st.image("https://tse1.mm.bing.net/th?id=OIP.12rN58CkEuHhkhvO8-cpgAHaEx&pid=Api&P=0", use_column_width=True)       

           
        
if __name__=='__main__':
    main()