
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

if __name__ == "__main__":
    
    

    dataset = load_data()  # Carica i dati assegnandoli a un pandas dataframe
    
    col2=["Irrelevant_Feature1","Irrelevant_Feature2","Sample code number"]
    col3=["Random_String","Irrelevant_Numeric","Sam!"]
    col4=["sample_code_number","randomfeature2","col_11"]
    col5=["irrelevant_col_1","col_0","irrelevant_col_2"]
    col_lab_2_3=["Class"]
    col_lab_4=["class"]
    col_lab_5=["col_10"]

    """queste variabili servono solo per provare il codice con tutti e 5 i file più velocemente,
       nella versione finale queste variabili saranno inserite tramite riga di comando"""

    
    
    


    dataset = elimina_colonne(dataset) #elimina le colonne che non si desiderano
    
    dataset = crea_dummy_variables(dataset) #converte le colonne che sono del tipo string in valori numerici usando le dummy variables

    [Features, colonne_label] = classlabel_selector(dataset) #divide il dataframe in due sotto dataframe: feature e label

    
    
    Features = gestisci_valori_mancanti(dataset)
    
    scala_features(Features)

    print(Features.dtypes)
    print(Features)
    print(colonne_label)
   
   
    


