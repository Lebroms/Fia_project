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
from scripts.data_preprocessing.loader.factory import load_data
from prova_codice_levenshtain import elimina_colonne_levenshtein

if __name__ == "__main__":
   
    dataset = load_data()  # Carica i dati assegnandoli a un pandas dataframe
    

    print(dataset.dtypes)  
     #restituisce le prime cinque colonne del dataset per accertare che il caricamento sia avvenuto


    #dataset,matched_columns, unmatched_columns_to_match= elimina_colonne_levenshtein(dataset)

    #print(dataset)
    #print(matched_columns)
    #print(unmatched_columns_to_match)
    #print("Intestazioni finali del DataFrame:", dataset.columns.tolist())


    