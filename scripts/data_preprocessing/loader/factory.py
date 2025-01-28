from .classe_loader import DataLoader
from .csv_loader import CsvLoader
from .xml_loader import XmlLoader
from .json_loader import JsonLoader

class Factory:
    @staticmethod
    def get_loader(file_path: str) -> DataLoader:
        # Estrai l'estensione del file in minuscolo
        file_extension = file_path.split('.')[-1].lower()

        # Mappa delle estensioni ai loader
        loaders = {
            'csv': CsvLoader,
            'xlsx': XmlLoader,
            'xls': XmlLoader,
            'json': JsonLoader,
        }

        # Recupera il loader corrispondente o restituisce un'errore se l'estensione non viene trovata
        loader_class = loaders.get(file_extension)
        if loader_class:
            return loader_class()
        else:
            raise ValueError(f"Formato file non supportato: {file_path}")


import json


#funzione per il caricamento del file tramite l'uso di un file di configurazione

def load_data():
    """
    Carica i dati da un file specificato nel file di configurazione.
    Returns:
        pd.DataFrame: Il dataset caricato.
    """
    # Leggi il file di configurazione
    with open("data/config.json", "r") as config_file:
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

