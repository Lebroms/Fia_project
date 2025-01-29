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
from scripts.data_preprocessing.pulizia_dataset.gestione_colonne import elimina_colonne_levenshtein
from scripts.data_preprocessing.pulizia_dataset.gestione_valori_Nan import gestisci_valori_mancanti

if __name__ == "__main__":
   
    dataset = load_data()  # Carica i dati assegnandoli a un pandas dataframe
    
    a=[]
    
    [dataset, a] = elimina_colonne_levenshtein(dataset)
    
    dataset = gestisci_valori_mancanti(dataset)
    
    
    #string_columns = dataset.select_dtypes(include=['object', 'category']).columns
    
    #dataset = crea_dummy_variables(dataset)
    
    #dataset = gestisci_valori_mancanti(dataset)
    
    
    print(dataset.columns)
    #print(string_columns)
    