import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("Prédiction de la Consommation de Drogues")

st.write("Sélectionnez une drogue et entrez les informations pour prédire si une personne est consommatrice.")

drug_options = {
    "Nicotine": "nicotine", "Alcool": "alcohol", "Amphétamines": "amphet", "Amyl Nitrite": "amyl", 
    "Benzodiazépines": "benzos", "Caféine": "caff", "Cannabis": "cannabis", 
    "Chocolat": "choc", "Cocaïne": "coke", "Crack": "crack", "Ecstasy": "ecstasy", 
    "Héroïne": "heroin", "Kétamine": "ketamine", "Drogues légales": "legalh", 
    "LSD": "lsd", "Méthamphétamines": "meth", "Champignons": "mushrooms" 
    
}
selected_drug_label = st.selectbox("Choisissez la drogue à prédire", list(drug_options.keys()))
selected_drug = drug_options[selected_drug_label]

try:
    with open(f'logistic_model_{selected_drug}.pkl', 'rb') as model_file:
        logistic_model = pickle.load(model_file)
    with open(f'scaler_{selected_drug}.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except FileNotFoundError:
    st.error(f"Modèle ou scaler pour {selected_drug_label} non trouvé. Veuillez vérifier les fichiers.")
    st.stop()

st.subheader("Informations démographiques et personnalité")
age = st.slider("Âge (années)", 18, 80, 30)
gender = st.selectbox("Sexe", ["Femme", "Homme"])
education = st.selectbox("Niveau d'éducation", [
    "Aucun diplôme", "Certificat d'école secondaire", "Diplôme d'études secondaires",
    "Certificat universitaire", "Diplôme universitaire", "Master", "Doctorat"
])
country = st.selectbox("Pays", ["Royaume-Uni", "États-Unis", "Canada", "Australie", "Autre"])
ethnicity = st.selectbox("Ethnicité", ["Blanche", "Noire", "Asiatique", "Mixte", "Autre"])
nscore = st.slider("Score de neuroticisme (0-100)", 0, 100, 10)
oscore = st.slider("Score d'ouverture à l'expérience (0-100)", 0, 100, 10)
impulsive = st.slider("Score d'impulsivité (0-100)", 0, 100, 10)
ss = st.slider("Score de recherche de sensations (0-100)", 0, 100, 10)

st.subheader("Fréquence de consommation d'autres substances")
drug_freq_options = {
    "Jamais utilisé": 0,
    "Utilisé il y a plus de 10 ans": 1,
    "Utilisé dans les 10 dernières années": 2,
    "Utilisé dans la dernière année": 3,
    "Utilisé dans le dernier mois": 4,
    "Utilisé dans la dernière semaine": 5,
    "Utilisé dans la dernière journée": 6
}

drug_frequencies = {}
for drug_label, drug_key in drug_options.items():
    if drug_key != selected_drug:  # Exclure la drogue prédite
        drug_frequencies[drug_key] = st.selectbox(
            f"Fréquence de consommation de {drug_label}",
            list(drug_freq_options.keys()),
            key=drug_key
        )

def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val) * 6 - 3  # Normalisation entre -3 et 3

# Créer le DataFrame d'entrée
input_data_dict = {
    'Age': normalize(age, 18, 80),
    'Gender': -0.48246 if gender == "Femme" else 0.48246,
    'Education': normalize({
        "Aucun diplôme": 0, "Certificat d'école secondaire": 1, "Diplôme d'études secondaires": 2,
        "Certificat universitaire": 3, "Diplôme universitaire": 4, "Master": 5, "Doctorat": 6
    }[education], 0, 6),
    'Country': normalize({
        "Royaume-Uni": 0, "États-Unis": 1, "Canada": 2, "Australie": 3, "Autre": 4
    }[country], 0, 4),
    'Ethnicity': normalize({
        "Blanche": 0, "Noire": 1, "Asiatique": 2, "Mixte": 3, "Autre": 4
    }[ethnicity], 0, 4),
    'Nscore': normalize(nscore, 0, 100),
    'Escore': normalize(10, 0, 100),  
    'Oscore': normalize(oscore, 0, 100),
    'AScore': normalize(10, 0, 100),  # Valeur par défaut fixée à 10, non visible
    'Cscore': normalize(10, 0, 100),  
    'Impulsive': normalize(impulsive, 0, 100),
    'SS': normalize(ss, 0, 100)
}

for drug_key, freq_label in drug_frequencies.items():
    input_data_dict[drug_key.capitalize()] = drug_freq_options[freq_label]

input_data = pd.DataFrame([input_data_dict])

expected_columns = [
    'Age', 'Gender', 'Education', 'Country', 'Ethnicity', 'Nscore', 'Escore', 
    'Oscore', 'AScore', 'Cscore', 'Impulsive', 'SS', 'Alcohol', 'Amphet', 
    'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc', 'Coke', 'Crack', 'Ecstasy', 
    'Heroin', 'Ketamine', 'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine'
]
expected_columns = [col for col in expected_columns if col.lower() != selected_drug]
input_data = input_data.reindex(columns=expected_columns, fill_value=0)

if st.button("Prédire"):
    try:
        scaled_input = scaler.transform(input_data)
        
        prediction = logistic_model.predict(scaled_input)
        prediction_prob = logistic_model.predict_proba(scaled_input)[0]
        
        if prediction[0] == 1:
            st.error(f"Le modèle prédit que cette personne est un **consommateur de {selected_drug_label}**.")
            st.write(f"Probabilité d'être consommateur : {prediction_prob[1]:.2%}")
        else:
            st.success(f"Le modèle prédit que cette personne **n'est pas consommateur de {selected_drug_label}**.")
            st.write(f"Probabilité de ne pas être consommateur : {prediction_prob[0]:.2%}")
    
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {str(e)}")