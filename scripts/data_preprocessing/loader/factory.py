from .classe_loader import DataLoader
from .csv_loader import CsvLoader
from .xml_loader import XmlLoader
from .json_loader import JsonLoader
from .xlsx_loader import ExcelLoader
from .txt_loader import TxtLoader

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
            'tsv':CsvLoader,      #Loader per file TSV
            'txt': TxtLoader,     #Loader per file txt
            'xlsx': ExcelLoader,    # Loader per file Excel (formato .xlsx)
            'xls': ExcelLoader,     # Loader per file Excel (formato .xls)
            'xml':XmlLoader,      #Loader per file XML
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
import argparse

def parse_arguments():
    """
    Definisce e analizza gli argomenti della riga di comando.
    
    Returns:
        argparse.Namespace: Oggetto con gli argomenti forniti dall'utente.
    """
    parser = argparse.ArgumentParser(description="Caricamento e pulizia dei dati.")

    # Argomento obbligatorio: percorso del file di input
    parser.add_argument("-i", "--input", required=True, help="Percorso del file di input")

    return parser.parse_args()




def load_data():
    """
    Carica i dati da un file specificato da riga di comando.
    Returns:
        pd.DataFrame: Il dataset caricato.
    """
    # Parse degli argomenti dalla riga di comando
    args = parse_arguments()
    input_path = args.input

    try:
        # Usa la Factory per ottenere il loader corretto
        loader = Factory.get_loader(input_path)
        dataset = loader.load(input_path)  # Carica il dataset

        dataset=convert_comma_to_dot(dataset)
        print("\nDataset caricato con successo.")
        return dataset
    except ValueError as e:
        print(f"Errore: {e}")
        return None
    except Exception as e:
        print(f"Errore imprevisto: {e}")
        return None



import pandas as pd

def convert_comma_to_dot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte tutte le colonne di tipo 'object' che contengono numeri con la virgola 
    in numeri con il punto e le trasforma in float, se possibile.
    
    Args:
        df (pd.DataFrame): Il DataFrame da modificare.
    
    Returns:
        pd.DataFrame: Il DataFrame con i valori corretti.
    """
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.replace(',', '.', regex=True)
        
        # Prova a convertire in float, se tutti i valori possono essere convertiti
        try:
            df[col] = df[col].astype(float)
        except ValueError:
            pass  # Se ci sono errori, mantiene il tipo object
    
    return df
