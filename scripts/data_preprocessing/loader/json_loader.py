import pandas as pd
from .classe_loader import DataLoader



class JsonLoader(DataLoader):
    """
    Sottoclasse di DataLoader per il caricamento di file Json.

    Questa classe implementa il metodo `load` per leggere file Json in un DataFrame.
    
    """
    def load(self, file_path: str) -> pd.DataFrame:
        """
        Carica un file Json in un DataFrame.

        Args:
            file_path (str): Il percorso del file da caricare.

        Returns:
            pd.DataFrame: Il contenuto del file caricato in un DataFrame.
        """


        print(f"Caricamento del file JSON: {file_path}")
        return pd.read_json(file_path)