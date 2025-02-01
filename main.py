
"""
Questo script rappresenta l'entry point del progetto per il caricamento e la gestione del dataset.

Funzionalità principali:
1. Carica un dataset da file con diverse estensioni supportate (CSV, JSON, XML).
2. Utilizza una Factory per selezionare automaticamente il caricatore appropriato in base al tipo di file.
3. Esegue il caricamento del dataset in un formato standard (pandas DataFrame).
4. Mostra i dati caricati per confermarne il corretto caricamento.

Come usarlo:
- Specificare il percorso del file da caricare nella variabile `file_path`.
- Lo script selezionerà il caricatore corretto e restituirà i dati pronti per l'elaborazione successiva.

Prerequisiti:
- Librerie richieste: pandas
- Il file deve essere presente nella directory indicata da `file_path`.

"""
from scripts.data_preprocessing.loader.factory import Factory, load_data
from scripts.data_preprocessing.pulizia_dataset.dummy_variables import crea_dummy_variables
from scripts.data_preprocessing.pulizia_dataset.feature_scaling import scala_features
from scripts.data_preprocessing.pulizia_dataset.gestisci_colonne import elimina_colonne
from scripts.data_preprocessing.pulizia_dataset.gestione_valori_Nan import gestisci_valori_mancanti
from scripts.data_preprocessing.Target_Features.ClassLabel_Selector import classlabel_selector

import pandas as pd

from KNN.Classificatore_Knn import Classificatore_KNN

if __name__ == "__main__":
    
    

    dataset = load_data()  # Carica i dati assegnandoli a un pandas dataframe

    dataset = elimina_colonne(dataset) #elimina le colonne che non si desiderano
    
    dataset = crea_dummy_variables(dataset) #converte le colonne che sono del tipo string in valori numerici usando le dummy variables

    [Features, colonne_label] = classlabel_selector(dataset) #divide il dataframe in due sotto dataframe: feature e label

    Features = gestisci_valori_mancanti(dataset)
    
    scala_features(Features)

    print(Features.dtypes)
    print(Features)
    print(colonne_label)


    #--------------------------------parte aggiunntiva a titolo di prova

    #inizializzazione del valore dei k vicini da usare per il classificatore
    #k=input("Inserire il valore dei k vicini da voler usare per costruire il Classificatore KNN: ")

    # Supponiamo che il DataFrame si chiami Features
    num_righe = len(Features)
    train_size = int(num_righe * 0.9)  # Calcola il 70% delle righe

    # Creazione dei dataset di training e test
    Features_train_set = Features.iloc[:train_size]  # Primi 70%
    Features_test_set = Features.iloc[train_size:]  # Ultimi 30%

    Labels_train_set = colonne_label.iloc[:train_size]  # Primi 70%
    Labels_test_set = colonne_label.iloc[train_size:]  # Ultimi 30%

    Knn=Classificatore_KNN(Features_train_set,Labels_train_set)

    lista_predizioni=Knn.predizione(Features_test_set)

    print(lista_predizioni)

    print(Labels_test_set)

    c = 0
    for predizione, valore in zip(lista_predizioni, Labels_test_set.iloc[:, 0]):
        if predizione == valore:
            c += 1

    print(c)



   
   
    


