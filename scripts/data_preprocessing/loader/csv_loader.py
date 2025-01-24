import pandas as pd
from loader.classe_loader import DataLoader

#sottoclasse di loader specializzata nel caricamento del formato csv

class CsvLoader(DataLoader):
    """
        Metodo per caricare un file in formato csv 

        Args:
            file_path (str): Il percorso del file da caricare.

        Returns:
            pd.DataFrame: I dati caricati come DataFrame.
        """
    def load(self, file_path: str) -> pd.DataFrame:
        print(f"Caricamento del file CSV: {file_path}")
        return pd.read_csv(file_path)