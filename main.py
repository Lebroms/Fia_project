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

import pandas as pd



from KNN.Classificatore_Knn import Classificatore_KNN

if __name__ == "__main__":
   
    dataset = load_data()  # Carica i dati assegnandoli a un pandas dataframe
    
    col2=["Irrelevant_Feature1","Irrelevant_Feature2","Sample code number"]
    col3=["Random_String","Irrelevant_Numeric","Sam!"]
    col4=["sample_code_number","randomfeature2","col_11"]
    col5=["irrelevant_col_1","col_0","irrelevant_col_2"]
    
    dataset = elimina_colonne(dataset,col2)

    
    dataset = crea_dummy_variables(dataset)    
    
    dataset = gestisci_valori_mancanti(dataset)
    
    scala_features(dataset)
    
    
    print(dataset.dtypes)   
    
    print(dataset.columns)
    print(dataset)


    #---------------------parte aggiuntiva solo di prova per il momento

    feature_set = dataset.drop(columns=["Class"])

    label_set= dataset["Class"]

    feature_train_set = feature_set.iloc[:int(0.7 * len(feature_set))]

    label_train_set= label_set.iloc[:int(0.7 * len(label_set))]


    feature_test_set=feature_set.iloc[int(0.7 * len(feature_set)):]
    label_test_set=label_set.iloc[int(0.7 * len(label_set)):]


    Knn=Classificatore_KNN(5,feature_train_set,label_train_set)
    lista_predizioni=Knn.predizione(feature_test_set)


    print(lista_predizioni)

    print(type(lista_predizioni))

    print(label_test_set)


    def precision(y_true, y_pred):
        TP = sum((y_t == 1 and y_p == 1) for y_t, y_p in zip(y_true, y_pred))  # Veri Positivi
        FP = sum((y_t == 0 and y_p == 1) for y_t, y_p in zip(y_true, y_pred))  # Falsi Positivi

        return TP / (TP + FP) if (TP + FP) > 0 else 0  # Evita divisione per zero


    test_labels_list = label_test_set.tolist()

    # Trova gli indici in cui test_labels e lista_predizioni sono diversi
    error_indices = [i for i, (y_t, y_p) in enumerate(zip(test_labels_list, lista_predizioni)) if y_t != y_p]

    # Calcola la precisione
    prec = precision(test_labels_list, lista_predizioni)

    # Stampa i risultati
    print(f"Precisione: {prec:.2f}")

    if error_indices:
        print("Indici delle predizioni errate:", error_indices)
    else:
        print("Tutte le predizioni sono corrette!")





   
   
    

