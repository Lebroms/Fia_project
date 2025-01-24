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
from scripts.data_preprocessing.loader.factory import Factory
if __name__ == "__main__":
    file_path = "C:/Users/lscor/OneDrive/Magistrale/F_Intelligenza_artificiale/Fia_project/data/version_4.json"

    # Creazione del loader usando la Factory
    loader = Factory.get_loader(file_path)

    # Caricamento del dataset
    dataset = loader.load(file_path)

    print("\nDataset caricato:")
    print(dataset)


processed_path = "C:/Users/lscor/OneDrive/Magistrale/F_Intelligenza_artificiale/Fia_project/data/dati_prepocessati.csv"
dataset.to_csv(processed_path, index=False)  # Salva senza includere l'indice
print(f"Dataset preprocessato salvato in: {processed_path}")