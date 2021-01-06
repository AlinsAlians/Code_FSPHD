from preprocessing_reg import preprocess
from epa_lstm_reg import lstm
from epa_cnn_reg import cnn
from ML_features_reg import ml
from ML_late_reg import ml_late

## variables explicatives :
# Séries temporelles de smt, pluie, prix mais,
# occupation du sol
# nombres de centres de santé et d'écoles pour 1000 habitants
# nombre d'événements violents pour 1000 habitants
# variables météo
# variables population
# ndvi moyens de l'année n et n-1
# variables économiques annuelles World Bank
# densités de population par pixel de 100m

for r_split in [1, 2, 3, 4, 5]:
    for rep in ['sda', 'sca']:
        print(rep, " / ", r_split)
        # preprocessing des variables
        preprocess(rep, r_split)

        # création des features avec 2 réseaux de neurones
        lstm(rep, r_split) # LSTM sur les séries temporelles
        cnn(rep, r_split) # CNN sur les pixels de densités de population et occupation du sol (cultures, forêts, constructions)
        # Random forest sur les variables initiales et sur les features
        ml(rep, r_split)
        # Régression ridge sur les réponses des 3 modèles
        ml_late(rep, r_split)


