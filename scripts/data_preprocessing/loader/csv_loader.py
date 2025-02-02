import pandas as pd
from .classe_loader import DataLoader

class CsvLoader(DataLoader):
    """
    Sottoclasse di DataLoader specializzata nel caricamento di file CSV e TSV.
    Il metodo load verifica l'estensione del file per scegliere il separatore.

    Args:
        file_path (str): Il percorso del file da caricare.

    Returns:
        pd.DataFrame: I dati caricati come DataFrame.
    """

    def load(self, file_path: str) -> pd.DataFrame:
        """
        Carica un file CSV o TSV in base all'estensione o al contenuto.

        Args:
            file_path (str): Il percorso del file da caricare.

        Returns:
            pd.DataFrame: I dati caricati come DataFrame.
        """
        # Imposta un separatore di default
        separator = ','

        # Se il file è un TSV, usa il separatore di tabulazione
        if file_path.endswith('.tsv'):
            separator = '\t'
        elif file_path.endswith('.csv'):
            # Controlla se potrebbe essere un CSV con ';' come separatore
            with open(file_path, "r", encoding="utf-8") as file:
                first_line = file.readline()
                if ";" in first_line:
                    separator = ";"

        
        return pd.read_csv(file_path, sep=separator)
