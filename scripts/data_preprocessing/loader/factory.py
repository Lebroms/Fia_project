from .classe_loader import DataLoader
from .csv_loader import CsvLoader
from .xml_loader import XmlLoader
from .json_loader import JsonLoader
from .txt_loader import TxtLoader
from .Excel_loader import ExcelLoader

import pandas as pd

class Factory:
    """
    Classe Factory per la gestione dinamica dei caricamenti di file.

    Questa classe fornisce un metodo per ottenere un'istanza del loader corretto in base
    all'estensione del file, supportando diversi formati tra cui CSV, TSV, Excel, JSON, XML e TXT.

    Methods:
        get_loader(file_path: str) -> DataLoader:
            Determina e restituisce il loader appropriato in base all'estensione del file.
    """
    @staticmethod
    def get_loader(file_path: str) -> DataLoader:
        """
        Determina il loader corretto per il file specificato in base alla sua estensione.

        Args:
            file_path (str): Il percorso completo del file di input.

        Returns:
            DataLoader: Un'istanza del loader appropriato per il tipo di file.

        """
        # Estrai l'estensione del file (in minuscolo) dall'input
        file_extension = file_path.split('.')[-1].lower()

        # Mappa delle estensioni supportate ai rispettivi loader
        loaders = {
            'csv': CsvLoader,     # Loader per file CSV
            'tsv':CsvLoader,      #Loader per file Tsv
            'txt':TxtLoader,        # Loader per file Txt
            'xlsx': ExcelLoader,    # Loader per file Excel (formato .xlsx)
            'xls': ExcelLoader,     # Loader per file Excel (formato .xls)
            'json': JsonLoader,     # Loader per file JSON
            'xml':XmlLoader
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
    Carica i dati da un file specificato tramite riga di comando.

    Questa funzione:
    - Analizza gli argomenti passati alla riga di comando per determinare il file di input.
    - Utilizza la classe `Factory` per ottenere il loader corretto in base all'estensione.
    - Converte eventuali numeri scritti con la virgola in numeri con il punto.
    - Restituisce il DataFrame risultante.

    Returns:
        pd.DataFrame: Il dataset caricato.

    """
    # Parse degli argomenti dalla riga di comando
    args = parse_arguments()
    input_path = args.input

    print(f"\n \U0001F4C2 Sto caricando il file '{input_path}' \n")



    try:
        # Usa la Factory per ottenere il loader corretto
        loader = Factory.get_loader(input_path)
        dataset = loader.load(input_path)  # Carica il dataset

        dataset=convert_comma_to_dot(dataset)
        print("\n \U00002705 Dataset caricato con successo. \n")
        return dataset
    except ValueError as e:
        print(f"Errore: {e}")
        return None
    except Exception as e:
        print(f"Errore imprevisto: {e}")
        return None


def convert_comma_to_dot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte i numeri con la virgola in numeri con il punto all'interno di un DataFrame.

    Questa funzione:
    - Analizza tutte le colonne di tipo 'object' nel DataFrame.
    - Sostituisce le virgole (`','`) con punti (`'.'`) nei valori.
    - Tenta di convertire le colonne in numeri float, se possibile.

    Args:
        df (pd.DataFrame): Il DataFrame da elaborare.

    Returns:
        pd.DataFrame: Il DataFrame con i valori numerici corretti.

    """
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.replace(',', '.', regex=True)
        
        # Prova a convertire in float, se tutti i valori possono essere convertiti
        try:
            df[col] = df[col].astype(float)
        except ValueError:
            pass  # Se ci sono errori, mantiene il tipo object
    
    return df



import argparse

def parse_arguments():
    """
    Definisce e analizza gli argomenti passati dalla riga di comando.

    Questa funzione permette di specificare:
    - Il percorso del file di input da caricare (`-i` o `--input`).

    Returns:
        argparse.Namespace: Un oggetto contenente gli argomenti forniti dall'utente.

    """
    parser = argparse.ArgumentParser(description="Caricamento e pulizia dei dati.")

    # Argomento obbligatorio: percorso del file di input
    parser.add_argument("-i","--input", type=str, default="dati/version_1.csv", help="Nome del file di input")


    

    return parser.parse_args()