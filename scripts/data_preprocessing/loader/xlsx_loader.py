import pandas as pd
from .classe_loader import DataLoader

class ExcelLoader(DataLoader):
    def load(self, file_path: str) -> pd.DataFrame:
        """
        Carica un file Excel (.xlsx) e lo converte in un DataFrame.

        Args:
            file_path (str): Il percorso del file Excel.

        Returns:
            pd.DataFrame: I dati caricati come DataFrame.
        """
        print(f"Caricamento del file Excel: {file_path}")
        return pd.read_excel(file_path)
