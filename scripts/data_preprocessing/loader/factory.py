from .classe_loader import DataLoader
from .csv_loader import CsvLoader
from .xml_loader import XmlLoader
from .json_loader import JsonLoader

class Factory:
    """
    Classe Factory che fornisce un metodo per ottenere dinamicamente un loader appropriato 
    in base all'estensione del file. Supporta i formati CSV, XLSX, XLS e JSON.

    Metodo:
        - get_loader(file_path: str): Determina e restituisce il loader corretto 
          basandosi sull'estensione del file fornito.
    """

    @staticmethod
    def get_loader(file_path: str) -> DataLoader:
        """
        Determina il loader corretto per il file specificato in base alla sua estensione.

        Args:
            file_path (str): Il percorso completo del file di input.

        Returns:
            DataLoader: Un'istanza del loader appropriato per il tipo di file.

        Raises:
            ValueError: Sollevato se l'estensione del file non è supportata.

        Esempio:
            loader = Factory.get_loader("dataset.csv")
            dataset = loader.load("dataset.csv")
        """
        # Estrai l'estensione del file (in minuscolo) dall'input
        file_extension = file_path.split('.')[-1].lower()

        # Mappa delle estensioni supportate ai rispettivi loader
        loaders = {
            'csv': CsvLoader,     # Loader per file CSV
            'xlsx': XmlLoader,    # Loader per file Excel (formato .xlsx)
            'xls': XmlLoader,     # Loader per file Excel (formato .xls)
            'json': JsonLoader,   # Loader per file JSON
        }

        # Recupera la classe del loader appropriato dalla mappa
        loader_class = loaders.get(file_extension)

        if loader_class:
            # Restituisci un'istanza del loader trovato
            return loader_class()
        else:
            # Solleva un'eccezione se l'estensione non è supportata
            raise ValueError(f"Formato file non supportato: {file_path}")

import json


#funzione per il caricamento del file tramite l'uso di un file di configurazione in cui è salvato 
#il file di partenza

def load_data():
    """
    Carica i dati da un file specificato nel file di configurazione.
    Returns:
        pd.DataFrame: Il dataset caricato.
    """
    # Leggi il file di configurazione
    with open("dati/config.json", "r") as config_file:
        config = json.load(config_file)

    input_path = config["input_file"]

    try:
        # Usa la Factory per ottenere il loader corretto
        loader = Factory.get_loader(input_path)
        dataset = loader.load(input_path)  # Carica il dataset
        print("\nDataset caricato con successo.")
        return dataset
    except ValueError as e:
        print(f"Errore: {e}")
        return None
    except Exception as e:
        print(f"Errore imprevisto: {e}")
        return None

